"""Loss and metrics for partially labeled biomass composition data."""

from typing import Dict

import numpy as np
from torch import Tensor

from .constants import TARGET_NAMES


def masked_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(dtype=prediction.dtype)
    return ((prediction - target).square() * weights).sum() / weights.sum().clamp_min(1.0)


def biomass_metrics(
    prediction: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    maes = []
    rmses = []
    r2_scores = []
    mapes = []
    for index, name in enumerate(TARGET_NAMES):
        valid = mask[:, index].astype(bool)
        if not np.any(valid):
            continue
        valid_prediction = prediction[valid, index]
        valid_target = target[valid, index]
        error = (valid_prediction - valid_target) * 100.0
        mae = float(np.mean(np.abs(error)))
        rmse = float(np.sqrt(np.mean(np.square(error))))
        target_residual = valid_target - np.mean(valid_target)
        total_sum_squares = float(np.sum(np.square(target_residual)))
        target_metrics = {"mae_percent": mae, "rmse_percent": rmse}

        if valid_target.size >= 2 and total_sum_squares > np.finfo(np.float64).eps:
            residual_sum_squares = float(
                np.sum(np.square(valid_prediction - valid_target))
            )
            r2 = 1.0 - residual_sum_squares / total_sum_squares
            target_metrics["r2"] = r2
            r2_scores.append(r2)

        nonzero = np.abs(valid_target) > np.finfo(np.float32).eps
        if np.any(nonzero):
            mape = float(
                np.mean(
                    np.abs(
                        (valid_prediction[nonzero] - valid_target[nonzero])
                        / valid_target[nonzero]
                    )
                )
                * 100.0
            )
            target_metrics["mape_percent"] = mape
            mapes.append(mape)

        metrics[name] = target_metrics
        maes.append(mae)
        rmses.append(rmse)
    if maes:
        metrics["mean"] = {
            "mae_percent": float(np.mean(maes)),
            "rmse_percent": float(np.mean(rmses)),
        }
        if r2_scores:
            metrics["mean"]["r2"] = float(np.mean(r2_scores))
        if mapes:
            metrics["mean"]["mape_percent"] = float(np.mean(mapes))
    return metrics
