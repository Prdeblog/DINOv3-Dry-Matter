"""Dependency-free semantic segmentation metrics."""

from typing import Dict

import torch
from torch import Tensor


class ConfusionMatrix:
    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.matrix = torch.zeros((num_classes, num_classes), dtype=torch.float64)

    @torch.no_grad()
    def update(self, logits_or_labels: Tensor, target: Tensor) -> None:
        if logits_or_labels.ndim == target.ndim + 1:
            prediction = logits_or_labels.argmax(dim=1)
        else:
            prediction = logits_or_labels
        prediction, target = prediction.reshape(-1), target.reshape(-1)
        valid = (target != self.ignore_index) & (target >= 0) & (
            target < self.num_classes
        )
        indices = self.num_classes * target[valid].long() + prediction[valid].long()
        counts = torch.bincount(
            indices.cpu(), minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        self.matrix += counts

    def compute(self) -> Dict[str, Tensor]:
        cm = self.matrix
        tp = cm.diag()
        actual = cm.sum(dim=1)
        predicted = cm.sum(dim=0)
        union = actual + predicted - tp
        iou = torch.where(union > 0, tp / union, torch.nan)
        precision = torch.where(predicted > 0, tp / predicted, torch.nan)
        recall = torch.where(actual > 0, tp / actual, torch.nan)
        total = cm.sum()
        overall_accuracy = tp.sum() / total.clamp_min(1)
        # Equation (4) is a one-vs-rest binary accuracy for every class.
        tn = total - actual - predicted + tp
        paper_pa = (tp + tn) / total.clamp_min(1)
        return {
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "paper_pa": paper_pa,
            "miou": torch.nanmean(iou),
            "mean_precision": torch.nanmean(precision),
            "mean_recall": torch.nanmean(recall),
            "mean_paper_pa": torch.nanmean(paper_pa),
            "overall_accuracy": overall_accuracy,
        }

