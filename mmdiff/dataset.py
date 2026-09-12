"""Dataset utilities for the Optical-SAR-Infrared text-to-image dataset."""

import json
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class OpticalSARInfraredDataset(Dataset):
    """Read one modality of Optical-SAR-Infrared from its JSONL annotations.

    Each JSONL row must contain ``file_name`` (relative to ``dataset_root``)
    and ``text``.  ``modality`` is deliberately restricted to the directory
    names used by this dataset: ``opt``, ``sar`` and ``ir``.  When ``category``
    is set, it is matched against the parent directory in ``file_name``.
    """

    VALID_MODALITIES = {"opt", "sar", "ir"}

    def __init__(
        self,
        dataset_root: str,
        modality: str,
        category: Optional[str] = None,
        tokenizer=None,
        resolution: int = 256,
        center_crop: bool = True,
        random_flip: bool = True,
    ):
        if modality not in self.VALID_MODALITIES:
            raise ValueError(
                f"modality must be one of {sorted(self.VALID_MODALITIES)}, got {modality!r}."
            )

        self.dataset_root = Path(dataset_root)
        self.modality = modality
        self.category = category
        self.tokenizer = tokenizer
        annotation_path = self.dataset_root / f"{modality}_train.jsonl"
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Training annotations not found: {annotation_path}")

        with annotation_path.open("r", encoding="utf-8") as annotation_file:
            self.samples = [json.loads(line) for line in annotation_file if line.strip()]
        if not self.samples:
            raise ValueError(f"No samples found in {annotation_path}")
        for sample in self.samples:
            if not {"file_name", "text"}.issubset(sample):
                raise ValueError("Each JSONL row must contain 'file_name' and 'text'.")
        if category is not None:
            self.samples = [
                sample for sample in self.samples if Path(sample["file_name"]).parent.name == category
            ]
            if not self.samples:
                raise ValueError(
                    f"No {modality!r} training samples found for category {category!r} in {annotation_path}."
                )

        self.image_transform: Callable = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
                transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda image: image),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]
        image_path = self.dataset_root / sample["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Image referenced by JSONL does not exist: {image_path}")

        with Image.open(image_path) as image:
            pixel_values = self.image_transform(image.convert("RGB"))
        text = sample["text"]
        item: Dict[str, object] = {
            "pixel_values": pixel_values,
            "text": text,
            "image_path": str(image_path),
        }
        if self.tokenizer is not None:
            item["input_ids"] = self.tokenizer(
                text,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids[0]
        return item


def collate_optical_sar_infrared(examples):
    """Batch processed image/text pairs while retaining their source texts."""
    batch = {
        "pixel_values": torch.stack([example["pixel_values"] for example in examples])
        .to(memory_format=torch.contiguous_format)
        .float(),
        "texts": [example["text"] for example in examples],
        "image_paths": [example["image_path"] for example in examples],
    }
    if "input_ids" in examples[0]:
        batch["input_ids"] = torch.stack([example["input_ids"] for example in examples])
    return batch
