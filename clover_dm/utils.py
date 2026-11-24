"""Shared configuration, checkpoint, reproducibility and visualization helpers."""

import json
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn

from beasline import BASELINE_MODELS
from .constants import MASK_COLORS
from .model import DeepLabV3PlusSE
from model.DINOv3BiomassNet import DEFAULT_DINOV3_BACKBONE, FrozenDINOv3BiomassNet


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def get_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def save_json(data: Any, path: Union[str, Path]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)


def build_model_from_config(config: Dict[str, Any], pretrained: bool) -> DeepLabV3PlusSE:
    model_config = config["model"]
    return DeepLabV3PlusSE(
        num_classes=int(model_config.get("num_classes", 5)),
        pretrained_backbone=pretrained,
        aspp_channels=int(model_config.get("aspp_channels", 256)),
    )


def load_segmentation_checkpoint(
    checkpoint_path: Union[str, Path],
    device: torch.device,
    config: Optional[Dict[str, Any]] = None,
) -> DeepLabV3PlusSE:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    effective_config = config or checkpoint_config
    if effective_config is None:
        raise ValueError("Checkpoint has no configuration; pass config explicitly")
    model = build_model_from_config(effective_config, pretrained=False)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def colorize_mask(mask: np.ndarray) -> Image.Image:
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for index, rgb in enumerate(MASK_COLORS):
        color[mask == index] = rgb
    return Image.fromarray(color, mode="RGB")


def build_biomass_model(
    config: Dict[str, Any], pretrained: bool
) -> nn.Module:
    model_config = config["model"]
    architecture = str(
        model_config.get("architecture", "frozen_dinov3_vitl16_mscale_dpa")
    ).lower()
    if architecture in BASELINE_MODELS:
        model = BASELINE_MODELS[architecture](
            pretrained_backbone=pretrained,
            dropout=float(model_config.get("dropout", 0.3)),
            hidden_channels=int(model_config.get("hidden_channels", 512)),
        )
        if bool(model_config.get("freeze_backbone", False)):
            model.freeze_feature_extractor()
        return model

    dinov3_names = {"dinov3", "frozen_dinov3_vitl16_mscale_dpa"}
    if architecture not in dinov3_names:
        available = sorted((*BASELINE_MODELS, *dinov3_names))
        raise ValueError(
            "Unknown biomass architecture {!r}. Choose one of: {}".format(
                architecture, ", ".join(available)
            )
        )
    configured_backbone = str(
        model_config.get("backbone_name", DEFAULT_DINOV3_BACKBONE)
    )
    backbone_path = Path(configured_backbone).expanduser()
    if not backbone_path.is_absolute():
        project_relative = Path(__file__).resolve().parents[1] / backbone_path
        if project_relative.exists():
            backbone_path = project_relative
    backbone_name = (
        str(backbone_path.resolve()) if backbone_path.exists() else configured_backbone
    )
    model = FrozenDINOv3BiomassNet(
        backbone_name=backbone_name,
        feature_layers=tuple(model_config.get("feature_layers", (4, 8, 12))),
        pretrained_backbone=pretrained,
        backbone_config=model_config.get("backbone_config"),
        local_files_only=bool(model_config.get("local_files_only", True)),
        dropout=float(model_config.get("dropout", 0.3)),
        hidden_channels=int(model_config.get("hidden_channels", 512)),
    )
    # Makes checkpoints self-contained: inference can reconstruct DINOv3 before
    # loading the stored state dict without downloading the backbone again.
    model_config["backbone_config"] = model.backbone_config
    return model


def load_biomass_checkpoint(
    checkpoint_path: Union[str, Path],
    device: torch.device,
    config: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    effective_config = checkpoint_config or config
    if effective_config is None:
        raise ValueError("Checkpoint has no configuration; pass config explicitly")
    model = build_biomass_model(effective_config, pretrained=False)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
    return model.to(device).eval()
