"""Evaluate labeled data or export predictions for unlabeled biomass images."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from clover_dm.constants import TARGET_NAMES
from clover_dm.data import BiomassDataset
from clover_dm.regression import biomass_metrics
from clover_dm.utils import get_device, load_biomass_checkpoint, load_config, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--source", choices=("phone", "camera"))
    parser.add_argument("--split", choices=("train", "val", "predict"), default="val")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/biomass_eval"))
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = get_device(args.device)
    data_config = config["data"]
    if args.manifest is not None:
        dataset = BiomassDataset(
            args.manifest,
            args.split,
            image_size=int(data_config["image_size"]),
            train=False,
        )
        source_label = str(args.manifest)
    else:
        data_root = args.data_root or Path(data_config.get("root", "data"))
        source = args.source or str(data_config.get("source", "camera"))
        dataset = BiomassDataset(
            data_root,
            args.split,
            image_size=int(data_config["image_size"]),
            train=False,
            source=source,
        )
        source_label = source
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    model = load_biomass_checkpoint(args.checkpoint, device, config)
    print("source={}, split={}, samples={}".format(source_label, args.split, len(dataset)))
    prediction_parts = []
    target_parts = []
    mask_parts = []
    names = []
    with torch.inference_mode():
        for images, targets, target_mask, batch_names in loader:
            predictions = model(images.to(device, non_blocking=True)).cpu().numpy()
            prediction_parts.append(predictions)
            target_parts.append(targets.numpy())
            mask_parts.append(target_mask.numpy())
            names.extend(batch_names)
    prediction = np.concatenate(prediction_parts)
    target = np.concatenate(target_parts)
    mask = np.concatenate(mask_parts)

    output = pd.DataFrame({"image": names})
    for index, name in enumerate(TARGET_NAMES):
        output["pred_{}_percent".format(name)] = prediction[:, index] * 100.0
        if np.any(mask[:, index]):
            truth = np.full(len(dataset), np.nan)
            truth[mask[:, index]] = target[mask[:, index], index] * 100.0
            output["true_{}_percent".format(name)] = truth
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_dir / "{}_predictions.csv".format(args.split), index=False)
    metrics = biomass_metrics(prediction, target, mask)
    if metrics:
        save_json(metrics, args.output_dir / "{}_metrics.json".format(args.split))
        print(
            "{} samples: mean RMSE={:.3f}%, mean MAE={:.3f}%, "
            "mean R2={:.4f}, mean MAPE={:.3f}%".format(
                len(dataset),
                metrics["mean"]["rmse_percent"],
                metrics["mean"]["mae_percent"],
                metrics["mean"].get("r2", float("nan")),
                metrics["mean"].get("mape_percent", float("nan")),
            )
        )
    else:
        print("Wrote predictions for {} unlabeled images".format(len(dataset)))


if __name__ == "__main__":
    main()
