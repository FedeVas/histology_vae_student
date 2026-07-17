from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.datasets.split import validate_metadata


class HistologyPatchDataset(Dataset):
    """
    Dataset для histology patches, описанных в metadata CSV.

    Metadata должна содержать:
        path
        patient_id
        slide_id
        split

    Опционально:
        patch_id
        label
        x_coordinate
        y_coordinate
        scanner
        stain_batch
    """

    def __init__(
        self,
        metadata: pd.DataFrame | str | Path,
        split: str,
        transform: Any = None,
        root_dir: str | Path = ".",
        paired_transform: Any = None,
    ) -> None:
        if isinstance(metadata, (str, Path)):
            metadata_frame = pd.read_csv(metadata)
        else:
            metadata_frame = metadata.copy()

        validate_metadata(
            metadata_frame,
            require_split=True,
        )

        supported_splits = {
            "train",
            "validation",
            "test",
        }

        if split not in supported_splits:
            raise ValueError(
                f"Split must be one of "
                f"{sorted(supported_splits)}, "
                f"received: {split!r}"
            )

        split_metadata = metadata_frame[
            metadata_frame["split"] == split
        ].copy()

        if split_metadata.empty:
            raise ValueError(
                f"No samples were found for split "
                f"{split!r}."
            )

        # Сначала сохраняем аргументы как атрибуты.
        self.transform = transform
        self.paired_transform = paired_transform

        # Теперь их можно безопасно проверять.
        if (
            self.transform is not None
            and self.paired_transform is not None
        ):
            raise ValueError(
                "Use either transform or "
                "paired_transform, not both."
            )

        if (
            self.transform is None
            and self.paired_transform is None
        ):
            raise ValueError(
                "Either transform or paired_transform "
                "must be provided."
            )

        self.metadata = (
            split_metadata
            .reset_index(drop=True)
        )

        self.split = split
        self.root_dir = Path(root_dir)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.metadata.iloc[index]

        image_path = self._resolve_image_path(row["path"])

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image was not found: {image_path.resolve()}"
            )

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                
                if self.paired_transform is not None:
                    input_image, target_image = self.paired_transform(image)
                    
                elif self.transform is not None:
                    input_image = self.transform(image)
                    target_image = None
                else:
                    raise ValueError(
                        "Dataset requires transform or "
                        "paired_transform to be specified."
                    )
        except OSError as error:
            raise OSError(
                f"Failed to read image: {image_path.resolve()}"
            ) from error

        # if self.transform is not None:
        #     image = self.transform(image)

        label = self._extract_label(row)

        sample: dict[str, Any] = {
            "image": input_image,
            "label": label,
            "path": str(image_path),
            "patient_id": str(row["patient_id"]),
            "slide_id": str(row["slide_id"]),
        }
        if target_image is not None:
            sample["target_image"] = target_image
        optional_string_fields = (
            "sample_id",
            "patch_id",
            "class_code",
            "class_name",
            "source",
            "group_id_source",
        )

        for field_name in optional_string_fields:
            if field_name not in row.index:
                continue

            field_value = row[field_name]

            if pd.isna(field_value):
                continue

            sample[field_name] = str(field_value)

        return sample

    def _resolve_image_path(self, stored_path: str) -> Path:
        image_path = Path(stored_path)

        if image_path.is_absolute():
            return image_path

        return self.root_dir / image_path

    @staticmethod
    def _extract_label(row: pd.Series) -> int:
        if "label" not in row.index:
            return -1

        label = row["label"]

        if pd.isna(label):
            return -1

        try:
            return int(label)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "The current Dataset implementation expects numeric labels. "
                f"Received label: {label!r}"
            ) from error