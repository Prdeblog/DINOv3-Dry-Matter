"""Frozen DINOv3 feature encoder for constrained biomass regression."""

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .DualPoolAttention import DualPoolingAttention
from .Mscale import MultiScaleHybridCrossBlock


DEFAULT_DINOV3_BACKBONE = str(
    (Path(__file__).resolve().parents[1] / "checkpoints" / "dinov3").resolve()
)


def _load_huggingface_backbone(
    model_name: str,
    pretrained: bool,
    backbone_config: Optional[Dict[str, Any]],
    local_files_only: bool,
) -> nn.Module:
    try:
        from transformers import AutoConfig, AutoModel
    except ImportError as error:
        raise ImportError(
            "DINOv3 requires transformers>=4.56.0; install the project requirements"
        ) from error

    try:
        if pretrained:
            return AutoModel.from_pretrained(
                model_name,
                local_files_only=local_files_only,
            )
        if backbone_config is not None:
            config_values = dict(backbone_config)
            model_type = config_values.pop("model_type", "dinov3_vit")
            config = AutoConfig.for_model(model_type, **config_values)
        else:
            config = AutoConfig.from_pretrained(
                model_name,
                local_files_only=local_files_only,
            )
        return AutoModel.from_config(config)
    except (KeyError, ValueError) as error:
        raise RuntimeError(
            "Unable to construct DINOv3. Use transformers>=4.56.0 and a "
            "DINOv3 model/config supported by Hugging Face."
        ) from error


class FrozenDINOv3Encoder(nn.Module):
    """Return spatial patch maps from selected frozen transformer layers."""

    def __init__(
        self,
        model_name: str = DEFAULT_DINOV3_BACKBONE,
        feature_layers: Sequence[int] = (4, 8, 12),
        pretrained: bool = True,
        backbone_config: Optional[Dict[str, Any]] = None,
        local_files_only: bool = True,
        backbone: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.feature_layers = tuple(int(index) for index in feature_layers)
        if len(self.feature_layers) != 3:
            raise ValueError("feature_layers must contain exactly three layer indices")
        self.backbone = backbone or _load_huggingface_backbone(
            model_name=model_name,
            pretrained=pretrained,
            backbone_config=backbone_config,
            local_files_only=local_files_only,
        )

        config = getattr(self.backbone, "config", None)
        if config is None or not hasattr(config, "hidden_size"):
            raise ValueError("DINOv3 backbone must expose config.hidden_size")
        self.hidden_size = int(config.hidden_size)
        patch_size = getattr(config, "patch_size", 16)
        if isinstance(patch_size, (list, tuple)):
            if len(patch_size) != 2 or patch_size[0] != patch_size[1]:
                raise ValueError("Only square DINOv3 patches are supported")
            patch_size = patch_size[0]
        self.patch_size = int(patch_size)
        depth = int(getattr(config, "num_hidden_layers", max(self.feature_layers)))
        if min(self.feature_layers) < 1 or max(self.feature_layers) > depth:
            raise ValueError(
                "feature_layers {} are invalid for a {}-layer backbone".format(
                    self.feature_layers, depth
                )
            )
        self.freeze()

    @property
    def config_dict(self) -> Dict[str, Any]:
        config = self.backbone.config
        if hasattr(config, "to_dict"):
            return dict(config.to_dict())
        return dict(vars(config))

    def freeze(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> "FrozenDINOv3Encoder":
        # The regression modules follow ``mode``; DINOv3 always stays frozen/eval.
        super().train(False)
        return self

    def _tokens_to_map(
        self,
        tokens: Tensor,
        grid_height: int,
        grid_width: int,
    ) -> Tensor:
        patch_count = grid_height * grid_width
        if tokens.ndim != 3 or tokens.shape[1] < patch_count:
            raise RuntimeError(
                "DINOv3 hidden state cannot be reshaped to a {}x{} patch grid".format(
                    grid_height, grid_width
                )
            )
        # Patch tokens are last; class/register tokens, when present, are prefixes.
        patches = tokens[:, -patch_count:, :]
        return patches.transpose(1, 2).reshape(
            tokens.shape[0], self.hidden_size, grid_height, grid_width
        )

    def forward(self, images: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        height, width = images.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                "Input height and width must be divisible by DINOv3 patch size {}".format(
                    self.patch_size
                )
            )
        with torch.no_grad():
            output = self.backbone(
                pixel_values=images,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = output.hidden_states
        if hidden_states is None or len(hidden_states) <= max(self.feature_layers):
            raise RuntimeError("DINOv3 did not return the requested hidden states")
        grid = (height // self.patch_size, width // self.patch_size)
        normalizer = getattr(self.backbone, "norm", None)
        return tuple(
            self._tokens_to_map(
                normalizer(hidden_states[index])
                if normalizer is not None
                else hidden_states[index],
                *grid,
            )
            for index in self.feature_layers
        )


class ScaleDescriptor(nn.Module):
    """Compact one feature scale using projection and dual pooling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, feature: Tensor) -> Tensor:
        feature = self.project(feature)
        average = F.adaptive_avg_pool2d(feature, 1).flatten(1)
        maximum = F.adaptive_max_pool2d(feature, 1).flatten(1)
        return torch.cat((average, maximum), dim=1)


class FrozenDINOv3BiomassNet(nn.Module):
    """Frozen DINOv3 multi-depth biomass-composition regressor.

    Output order is grass, clover, white_clover, red_clover and weeds. The
    four disjoint components sum to one; total clover is white plus red.
    """

    target_names = ("grass", "clover", "white_clover", "red_clover", "weeds")

    def __init__(
        self,
        backbone_name: str = DEFAULT_DINOV3_BACKBONE,
        feature_layers: Sequence[int] = (4, 8, 12),
        pretrained_backbone: bool = True,
        backbone_config: Optional[Dict[str, Any]] = None,
        local_files_only: bool = True,
        hidden_channels: int = 512,
        dropout: float = 0.3,
        backbone: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.encoder = FrozenDINOv3Encoder(
            model_name=backbone_name,
            feature_layers=feature_layers,
            pretrained=pretrained_backbone,
            backbone_config=backbone_config,
            local_files_only=local_files_only,
            backbone=backbone,
        )
        channels = self.encoder.hidden_size

        self.low_context = MultiScaleHybridCrossBlock(channels, channels)
        self.mid_context = MultiScaleHybridCrossBlock(channels, channels)
        self.low_attention = DualPoolingAttention(channels)
        self.mid_attention = DualPoolingAttention(channels)
        self.high_attention = DualPoolingAttention(channels)

        self.low_descriptor = ScaleDescriptor(channels, 64)
        self.mid_descriptor = ScaleDescriptor(channels, 128)
        self.high_descriptor = ScaleDescriptor(channels, 256)
        descriptor_channels = 2 * (64 + 128 + 256)
        self.regression_head = nn.Sequential(
            nn.Linear(descriptor_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 4),
        )

    @property
    def backbone_config(self) -> Dict[str, Any]:
        return self.encoder.config_dict

    def train(self, mode: bool = True) -> "FrozenDINOv3BiomassNet":
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        low, mid, high = self.encoder(images)
        low = self.low_attention(self.low_context(low))
        mid = self.mid_attention(self.mid_context(mid))
        high = self.high_attention(high)

        descriptor = torch.cat(
            (
                self.low_descriptor(low),
                self.mid_descriptor(mid),
                self.high_descriptor(high),
            ),
            dim=1,
        )
        components = self.regression_head(descriptor).softmax(dim=1)
        grass, white_clover, red_clover, weeds = components.unbind(dim=1)
        return torch.stack(
            (
                grass,
                white_clover + red_clover,
                white_clover,
                red_clover,
                weeds,
            ),
            dim=1,
        )
