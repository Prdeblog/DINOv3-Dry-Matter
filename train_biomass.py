"""Train direct image-to-biomass regression using only biomass_data."""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from clover_dm.constants import TARGET_NAMES
from clover_dm.data import BiomassDataset
from clover_dm.regression import biomass_metrics, masked_mse
from clover_dm.utils import build_biomass_model, get_device, load_config, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument(
        "--seed",
        type=int,
        help="Override experiment.seed from the YAML configuration",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--source", choices=("phone", "camera"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--architecture",
        help="Override model.architecture from the YAML configuration",
    )
    pretrained_group = parser.add_mutually_exclusive_group()
    pretrained_group.add_argument(
        "--pretrained-backbone",
        dest="pretrained_backbone",
        action="store_true",
        help="Use ImageNet or DINOv3 pretrained backbone weights",
    )
    pretrained_group.add_argument(
        "--no-pretrained-backbone",
        dest="pretrained_backbone",
        action="store_false",
        help="Initialize the backbone without pretrained weights",
    )
    parser.set_defaults(pretrained_backbone=None)
    parser.add_argument("--hidden-channels", type=int)
    parser.add_argument("--dropout", type=float)
    return parser.parse_args()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: AdamW = None,
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    valid_count = 0
    predictions: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    masks: List[np.ndarray] = []

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, target, target_mask, _ in loader:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            target_mask = target_mask.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(images)
            loss = masked_mse(prediction, target, target_mask)
            if training:
                loss.backward()
                optimizer.step()
            count = int(target_mask.sum().item())
            loss_sum += float(loss.item()) * count
            valid_count += count
            predictions.append(prediction.detach().cpu().numpy())
            targets.append(target.detach().cpu().numpy())
            masks.append(target_mask.detach().cpu().numpy())
    metrics = biomass_metrics(
        np.concatenate(predictions), np.concatenate(targets), np.concatenate(masks)
    )
    return loss_sum / max(valid_count, 1), metrics


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    epoch: int,
    best_mae: float,
    config: Dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_mae": best_mae,
            "config": config,
            "target_names": TARGET_NAMES,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed
    model_config = config.setdefault("model", {})
    if args.architecture is not None:
        model_config["architecture"] = args.architecture
    if args.pretrained_backbone is not None:
        model_config["pretrained_backbone"] = args.pretrained_backbone
    if args.hidden_channels is not None:
        model_config["hidden_channels"] = args.hidden_channels
    if args.dropout is not None:
        model_config["dropout"] = args.dropout
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    device = get_device(args.device)
    data_config = config["data"]
    train_config = config["training"]
    data_root = args.data_root or Path(data_config.get("root", "data"))
    source = args.source or str(data_config.get("source", "camera"))
    data_config["root"] = str(data_root)
    data_config["source"] = source
    image_size = int(data_config["image_size"])
    datasets = {
        "train": BiomassDataset(
            data_root,
            "train",
            image_size=image_size,
            train=True,
            source=source,
        ),
        "val": BiomassDataset(
            data_root,
            "val",
            image_size=image_size,
            train=False,
            source=source,
        ),
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=int(train_config["batch_size"]),
            shuffle=split == "train",
            num_workers=int(data_config.get("num_workers", 0)),
            pin_memory=device.type == "cuda",
        )
        for split, dataset in datasets.items()
    }

    model = build_biomass_model(
        config, pretrained=bool(config["model"].get("pretrained_backbone", True))
    ).to(device)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = AdamW(
        trainable_parameters,
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config.get("weight_decay", 0.0)),
    )
    epochs = int(train_config["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    output_dir = Path(
        str(config["experiment"]["output_dir"]).format(
            source=source,
            architecture=model_config.get("architecture", "dinov3"),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    best_mae = float("inf")
    history: List[Dict[str, float]] = []

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_mae = float(checkpoint.get("best_mae", best_mae))

    print(
        "device={}, architecture={}, source={}, seed={}, train={}, val={}".format(
            device,
            model_config.get("architecture", "dinov3"),
            source,
            seed,
            len(datasets["train"]),
            len(datasets["val"]),
        )
    )
    for epoch in range(start_epoch, epochs):
        train_loss, train_metrics = run_epoch(model, loaders["train"], device, optimizer)
        val_loss, val_metrics = run_epoch(model, loaders["val"], device)
        scheduler.step()
        val_mae = val_metrics["mean"]["mae_percent"]
        train_mean = train_metrics["mean"]
        val_mean = val_metrics["mean"]
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_mae_percent": train_mean["mae_percent"],
            "val_mae_percent": val_mae,
            "train_r2": train_mean.get("r2", float("nan")),
            "val_r2": val_mean.get("r2", float("nan")),
            "train_mape_percent": train_mean.get(
                "mape_percent", float("nan")
            ),
            "val_mape_percent": val_mean.get("mape_percent", float("nan")),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        save_checkpoint(
            output_dir / "last_biomass.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            min(best_mae, val_mae),
            config,
        )
        if val_mae < best_mae:
            best_mae = val_mae
            save_checkpoint(
                output_dir / "best_biomass.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_mae,
                config,
            )
            save_json(val_metrics, output_dir / "best_val_metrics.json")
        with (output_dir / "biomass_history.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(
            "Epoch {:03d}/{:03d} train_loss={:.5f} val_loss={:.5f} "
            "val_MAE={:.3f}% val_R2={:.4f} val_MAPE={:.3f}%".format(
                epoch + 1,
                epochs,
                train_loss,
                val_loss,
                val_mae,
                val_mean.get("r2", float("nan")),
                val_mean.get("mape_percent", float("nan")),
            )
        )


if __name__ == "__main__":
    main()
