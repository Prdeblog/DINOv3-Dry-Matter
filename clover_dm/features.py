"""Pixel-level features that connect segmentation to random-forest regression."""

from typing import Dict

import torch
from torch import Tensor

from .constants import CLASS_NAMES


def pixel_statistics(labels: Tensor) -> Dict[str, float]:
    """Return paper-style class pixel counts and normalized fractions."""
    flat = labels.reshape(-1).long().cpu()
    counts = torch.bincount(flat, minlength=len(CLASS_NAMES))[: len(CLASS_NAMES)]
    total = int(counts.sum().item())
    output: Dict[str, float] = {}
    for index, name in enumerate(CLASS_NAMES):
        count = int(counts[index].item())
        output["pixel_count_{}".format(name)] = count
        output["pixel_fraction_{}".format(name)] = count / max(total, 1)
    return output

