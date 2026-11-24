"""Datasets and transforms for segmentation and biomass regression."""

import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

from .constants import GRASSCLOVER15_TO_FIVE, IGNORE_INDEX, TARGET_NAMES

PathLike = Union[str, Path]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

BIOMASS_SOURCE_FILES = {
    "camera": {"train": "train.csv", "val": "val.csv"},
    "phone": {"train": "phone_gt_train.csv", "val": "phone_gt_val.csv"},
}


def resolve_path(path: str, manifest_path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (manifest_path.parent / candidate).resolve()


def read_manifest(path: PathLike, split: Optional[str] = None) -> List[Dict[str, str]]:
    manifest_path = Path(path).resolve()
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Manifest is empty: {}".format(manifest_path))
    if split is not None:
        rows = [row for row in rows if row.get("split", "") == split]
    for row in rows:
        for key in ("image", "mask"):
            if row.get(key):
                row[key] = str(resolve_path(row[key], manifest_path))
    return rows


def read_biomass_source(
    data_root: PathLike,
    source: str,
    split: str,
) -> List[Dict[str, str]]:
    """Read a native ``data/camera`` or ``data/phone`` composition CSV."""
    source = source.lower()
    if source not in BIOMASS_SOURCE_FILES:
        raise ValueError("source must be 'phone' or 'camera'")
    if split not in BIOMASS_SOURCE_FILES[source]:
        raise ValueError(
            "Native {} data only supports train and val splits, not '{}'".format(
                source, split
            )
        )

    source_dir = Path(data_root).resolve() / source
    csv_path = source_dir / BIOMASS_SOURCE_FILES[source][split]
    image_dir = source_dir / "images"
    if not csv_path.is_file():
        raise FileNotFoundError("Biomass label CSV not found: {}".format(csv_path))
    if not image_dir.is_dir():
        raise FileNotFoundError("Biomass image directory not found: {}".format(image_dir))

    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        native_rows = list(csv.DictReader(stream))
    if not native_rows:
        raise ValueError("Biomass label CSV is empty: {}".format(csv_path))

    rows: List[Dict[str, str]] = []
    for native in native_rows:
        image_name = (native.get("Image Name") or "").strip()
        if not image_name:
            raise ValueError("Missing 'Image Name' in {}".format(csv_path))
        image_path = image_dir / image_name
        if not image_path.is_file():
            raise FileNotFoundError("Biomass image not found: {}".format(image_path))
        rows.append(
            {
                "image": str(image_path),
                "split": split,
                "source": source,
                "grass": (native.get("Grass Dried") or "").strip(),
                "clover": (native.get("Clover Dried") or "").strip(),
                "white_clover": "",
                "red_clover": "",
                "weeds": (native.get("Weeds Dried") or "").strip(),
            }
        )
    return rows


def remap_mask(mask: np.ndarray, mask_format: str) -> np.ndarray:
    if mask_format == "five_class":
        output = mask.astype(np.int64, copy=True)
        valid = ((output >= 0) & (output < 5)) | (output == IGNORE_INDEX)
        if not np.all(valid):
            values = np.unique(output[~valid]).tolist()
            raise ValueError("Unexpected five-class mask values: {}".format(values))
        return output
    if mask_format != "grassclover15":
        raise ValueError("mask_format must be 'grassclover15' or 'five_class'")
    output = np.full(mask.shape, IGNORE_INDEX, dtype=np.int64)
    for source, target in GRASSCLOVER15_TO_FIVE.items():
        output[mask == source] = target
    unknown = set(np.unique(mask).tolist()) - set(GRASSCLOVER15_TO_FIVE)
    if unknown:
        raise ValueError("Unexpected GrassClover mask values: {}".format(sorted(unknown)))
    return output


class PairedTransform:
    def __init__(self, image_size: int, train: bool) -> None:
        self.image_size = image_size
        self.train = train

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[Tensor, Tensor]:
        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        mask = TF.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST,
        )
        if self.train:
            if random.random() < 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            if random.random() < 0.5:
                image, mask = TF.vflip(image), TF.vflip(mask)
            turns = random.randint(0, 3)
            if turns:
                angle = 90 * turns
                image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
            brightness = random.uniform(0.85, 1.15)
            contrast = random.uniform(0.85, 1.15)
            image = ImageEnhance.Brightness(image).enhance(brightness)
            image = ImageEnhance.Contrast(image).enhance(contrast)
        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, IMAGENET_MEAN, IMAGENET_STD)
        mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.uint8).copy()).long()
        return image_tensor, mask_tensor


class SegmentationDataset(Dataset):
    def __init__(
        self,
        manifest: PathLike,
        split: str,
        image_size: int = 512,
        mask_format: str = "grassclover15",
        train: bool = False,
    ) -> None:
        self.rows = read_manifest(manifest, split=split)
        if not self.rows:
            raise ValueError("No rows for split '{}' in {}".format(split, manifest))
        self.mask_format = mask_format
        self.transform = PairedTransform(image_size=image_size, train=train)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, str]:
        row = self.rows[index]
        image_path, mask_path = Path(row["image"]), Path(row["mask"])
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        with Image.open(mask_path) as source:
            raw_mask = np.asarray(source.convert("L"), dtype=np.uint8)
        mapped_mask = remap_mask(raw_mask, self.mask_format)
        mask = Image.fromarray(mapped_mask.astype(np.uint8), mode="L")
        image_tensor, mask_tensor = self.transform(image, mask)
        return image_tensor, mask_tensor, image_path.name


def load_image(path: PathLike, image_size: int) -> Tuple[Tensor, Image.Image]:
    with Image.open(path) as source:
        original = source.convert("RGB")
    resized = TF.resize(
        original,
        [image_size, image_size],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    tensor = TF.normalize(TF.to_tensor(resized), IMAGENET_MEAN, IMAGENET_STD)
    return tensor, original


class ImageManifestDataset(Dataset):
    def __init__(self, manifest: PathLike, image_size: int) -> None:
        self.rows = read_manifest(manifest)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        tensor, _ = load_image(self.rows[index]["image"], self.image_size)
        return tensor, index


class BiomassTransform:
    """RGB augmentation and DINOv3-compatible ImageNet normalization."""

    def __init__(self, image_size: int, train: bool) -> None:
        self.image_size = image_size
        self.train = train

    def __call__(self, image: Image.Image) -> Tensor:
        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        if self.train:
            if random.random() < 0.5:
                image = TF.hflip(image)
            if random.random() < 0.5:
                image = TF.vflip(image)
            turns = random.randint(0, 3)
            if turns:
                image = TF.rotate(
                    image,
                    90 * turns,
                    interpolation=InterpolationMode.BILINEAR,
                )
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
        return TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)


class BiomassDataset(Dataset):
    """GrassClover images with partially available biomass labels.

    Targets are returned as fractions in [0, 1]. The official CSV uses
    percentages, while custom manifests may use either percentages or
    fractions. A target mask lets basic annotations contribute grass,
    total-clover and weed labels without inventing red/white-clover labels.
    """

    def __init__(
        self,
        data_path: PathLike,
        split: Optional[str],
        image_size: int = 512,
        train: bool = False,
        source: Optional[str] = None,
    ) -> None:
        if source is None:
            self.rows = read_manifest(data_path, split=split)
        else:
            if split is None:
                raise ValueError("split is required when loading phone/camera data")
            self.rows = read_biomass_source(data_path, source=source, split=split)
        if not self.rows:
            raise ValueError("No rows for split '{}' in {}".format(split, data_path))
        self.transform = BiomassTransform(image_size=image_size, train=train)

        values = []
        for row in self.rows:
            for name in TARGET_NAMES:
                value = row.get(name, "").strip()
                if value:
                    values.append(abs(float(value)))
        self.target_scale = 100.0 if values and max(values) > 1.5 else 1.0

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor, str]:
        row = self.rows[index]
        image_path = Path(row["image"])
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        image_tensor = self.transform(image)

        targets = torch.zeros(len(TARGET_NAMES), dtype=torch.float32)
        target_mask = torch.zeros(len(TARGET_NAMES), dtype=torch.bool)
        for target_index, name in enumerate(TARGET_NAMES):
            raw = row.get(name, "").strip()
            if raw:
                targets[target_index] = float(raw) / self.target_scale
                target_mask[target_index] = True
        return image_tensor, targets, target_mask, image_path.name
