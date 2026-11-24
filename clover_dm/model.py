"""DeepLabv3+ with a MobileNetV2 encoder and SE-enhanced ASPP.

The structure follows Figures 1--3 of Ji et al. (2023): the standard
MobileNetV2 encoder supplies low- and high-level features, ASPP uses rates
6/12/18 and global pooling, and SE is applied after multi-scale fusion.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ) -> None:
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        scale = self.fc(F.adaptive_avg_pool2d(x, 1))
        return x * scale


class ASPPSE(nn.Module):
    def __init__(
        self,
        in_channels: int = 320,
        out_channels: int = 256,
        rates: Tuple[int, int, int] = (6, 12, 18),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [ConvBNReLU(in_channels, out_channels, kernel_size=1)]
            + [
                ConvBNReLU(in_channels, out_channels, kernel_size=3, dilation=rate)
                for rate in rates
            ]
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            # BatchNorm is intentionally omitted: a 1x1 pooled tensor is valid
            # even when the training batch size is one.
            nn.ReLU(inplace=True),
        )
        fused_channels = out_channels * (len(rates) + 2)
        self.se = SqueezeExcitation(fused_channels)
        self.project = nn.Sequential(
            ConvBNReLU(fused_channels, out_channels, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        size = x.shape[-2:]
        features = [branch(x) for branch in self.branches]
        pooled = F.interpolate(
            self.pool(x), size=size, mode="bilinear", align_corners=False
        )
        features.append(pooled)
        return self.project(self.se(torch.cat(features, dim=1)))


class MobileNetV2Encoder(nn.Module):
    """Standard output-stride-32 MobileNetV2 from Table 1 of the paper."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights: Optional[MobileNet_V2_Weights]
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v2(weights=weights)
        # The final 1x1 layer produces 1280 channels; Table 1 and Figure 2 use
        # the preceding 320-channel feature map.
        self.features = backbone.features[:-1]

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        low = x
        for index, block in enumerate(self.features):
            x = block(x)
            if index == 3:  # 24 channels at output stride 4
                low = x
        return low, x  # high: 320 channels at output stride 32


class DeepLabV3PlusSE(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        pretrained_backbone: bool = True,
        aspp_channels: int = 256,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.encoder = MobileNetV2Encoder(pretrained=pretrained_backbone)
        self.aspp = ASPPSE(in_channels=320, out_channels=aspp_channels)
        self.low_projection = ConvBNReLU(24, 48, kernel_size=1)
        self.decoder = nn.Sequential(
            ConvBNReLU(aspp_channels + 48, 256, kernel_size=3),
            ConvBNReLU(256, 256, kernel_size=3),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        input_size = x.shape[-2:]
        low, high = self.encoder(x)
        high = self.aspp(high)
        high = F.interpolate(
            high, size=low.shape[-2:], mode="bilinear", align_corners=False
        )
        logits = self.decoder(torch.cat((high, self.low_projection(low)), dim=1))
        return F.interpolate(
            logits, size=input_size, mode="bilinear", align_corners=False
        )


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
