"""Predict biomass composition directly from one image."""

import argparse
import json
from pathlib import Path

import torch

from clover_dm.constants import TARGET_NAMES
from clover_dm.data import load_image
from clover_dm.utils import get_device, load_biomass_checkpoint, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = get_device(args.device)
    model = load_biomass_checkpoint(args.checkpoint, device, config)
    image_size = int(config["data"]["image_size"])
    tensor, _ = load_image(args.image, image_size)
    with torch.inference_mode():
        prediction = model(tensor.unsqueeze(0).to(device))[0].cpu().tolist()
    result = {
        "image": str(args.image),
        "biomass_percent": {
            name: round(float(value) * 100.0, 4)
            for name, value in zip(TARGET_NAMES, prediction)
        },
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
