from pathlib import Path

import pandas as pd
import pytest
import torch

from src.datasets.patch_dataset import HistologyPatchDataset
from src.datasets.split import (
    assign_patient_splits,
    assert_no_patient_leakage,
    get_split_label_summary,
)
from src.datasets.synthetic import (
    generate_synthetic_histology_dataset,
)
from src.datasets.transforms import (
    build_evaluation_transforms,
)


def create_test_metadata(
    temporary_directory: Path,
) -> pd.DataFrame:
    image_directory = (
        temporary_directory
        / "synthetic_patches"
    )

    metadata_path = (
        temporary_directory
        / "metadata.csv"
    )

    metadata = generate_synthetic_histology_dataset(
        output_dir=image_directory,
        metadata_path=metadata_path,
        num_patients=12,
        slides_per_patient=1,
        patches_per_slide=2,
        image_size=32,
        seed=42,
    )

    metadata = assign_patient_splits(
        metadata=metadata,
        train_fraction=0.70,
        validation_fraction=0.15,
        test_fraction=0.15,
        seed=42,
    )

    return metadata


def test_patient_splits_have_no_leakage(
    tmp_path: Path,
) -> None:
    metadata = create_test_metadata(tmp_path)

    assert_no_patient_leakage(metadata)

    available_splits = set(
        metadata["split"].unique()
    )

    assert available_splits == {
        "train",
        "validation",
        "test",
    }


def test_dataset_returns_expected_image_shape(
    tmp_path: Path,
) -> None:
    metadata = create_test_metadata(tmp_path)

    dataset = HistologyPatchDataset(
        metadata=metadata,
        split="train",
        transform=build_evaluation_transforms(
            image_size=32
        ),
    )

    sample = dataset[0]

    assert isinstance(
        sample["image"],
        torch.Tensor,
    )

    assert sample["image"].shape == (
        3,
        32,
        32,
    )

    assert sample["image"].dtype == torch.float32

    assert sample["image"].min().item() >= 0.0
    assert sample["image"].max().item() <= 1.0

    assert isinstance(
        sample["patient_id"],
        str,
    )

    assert isinstance(
        sample["slide_id"],
        str,
    )


def test_dataset_contains_only_requested_split(
    tmp_path: Path,
) -> None:
    metadata = create_test_metadata(tmp_path)

    validation_dataset = HistologyPatchDataset(
        metadata=metadata,
        split="validation",
        transform=build_evaluation_transforms(
            image_size=32
        ),
    )

    assert set(
        validation_dataset.metadata["split"].unique()
    ) == {"validation"}
    

def test_stratified_patient_split_contains_both_labels() -> None:
    records: list[dict[str, object]] = []

    for patient_index in range(12):
        patient_id = f"patient_{patient_index:03d}"
        label = patient_index % 2

        for patch_index in range(2):
            records.append(
                {
                    "path": (
                        f"unused/{patient_id}_"
                        f"{patch_index}.png"
                    ),
                    "patient_id": patient_id,
                    "slide_id": (
                        f"{patient_id}_slide"
                    ),
                    "label": label,
                }
            )

    metadata = pd.DataFrame(records)

    split_metadata = assign_patient_splits(
        metadata=metadata,
        train_fraction=0.70,
        validation_fraction=0.15,
        test_fraction=0.15,
        seed=42,
        stratify_column="label",
    )

    assert_no_patient_leakage(
        split_metadata
    )

    patient_distribution = (
        split_metadata
        .groupby(
            [
                "split",
                "label",
            ]
        )["patient_id"]
        .nunique()
        .unstack(fill_value=0)
    )

    assert patient_distribution.loc[
        "train", 0
    ] == 4

    assert patient_distribution.loc[
        "train", 1
    ] == 4

    assert patient_distribution.loc[
        "validation", 0
    ] == 1

    assert patient_distribution.loc[
        "validation", 1
    ] == 1

    assert patient_distribution.loc[
        "test", 0
    ] == 1

    assert patient_distribution.loc[
        "test", 1
    ] == 1


def test_stratified_split_rejects_multiple_labels_per_patient() -> None:
    metadata = pd.DataFrame(
        [
            {
                "path": "unused/a.png",
                "patient_id": "patient_001",
                "slide_id": "slide_001",
                "label": 0,
            },
            {
                "path": "unused/b.png",
                "patient_id": "patient_001",
                "slide_id": "slide_001",
                "label": 1,
            },
            {
                "path": "unused/c.png",
                "patient_id": "patient_002",
                "slide_id": "slide_002",
                "label": 0,
            },
        ]
    )

    with pytest.raises(ValueError):
        assign_patient_splits(
            metadata=metadata,
            stratify_column="label",
        )