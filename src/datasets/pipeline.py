from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from torch.utils.data import DataLoader

from src.datasets.patch_dataset import HistologyPatchDataset
from src.datasets.split import (
    assign_patient_splits,
    validate_metadata,
)
from src.datasets.synthetic import generate_synthetic_histology_dataset
from src.datasets.transforms import (
    build_evaluation_transforms,
    build_train_transforms,
)
from src.utils.device import RuntimeDevice
from src.utils.reproducibility import (
    create_torch_generator,
    seed_data_loader_worker,
)


@dataclass
class DataBundle:
    """
    Все datasets и DataLoaders одного эксперимента.
    """

    metadata: pd.DataFrame

    train_dataset: HistologyPatchDataset
    validation_dataset: HistologyPatchDataset
    test_dataset: HistologyPatchDataset

    train_loader: DataLoader
    validation_loader: DataLoader
    test_loader: DataLoader


def prepare_metadata(config: dict[str, Any]) -> pd.DataFrame:
    """
    Загружает metadata или создаёт synthetic dataset.

    При отсутствии split назначает train/validation/test
    на уровне пациентов.
    """
    data_config = config["data"]
    project_config = config["project"]
    synthetic_config = data_config["synthetic"]

    metadata_path = Path(data_config["metadata_csv"])

    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)

    elif bool(synthetic_config["enabled"]):
        metadata = generate_synthetic_histology_dataset(
            output_dir=synthetic_config["output_dir"],
            metadata_path=metadata_path,
            num_patients=int(synthetic_config["num_patients"]),
            slides_per_patient=int(
                synthetic_config["slides_per_patient"]
            ),
            patches_per_slide=int(
                synthetic_config["patches_per_slide"]
            ),
            image_size=int(data_config["image_size"]),
            seed=int(project_config["seed"]),
        )

    else:
        raise FileNotFoundError(
            f"Metadata file does not exist: {metadata_path.resolve()}"
        )

    if "split" not in metadata.columns:
        split_config = data_config["split"]

        metadata = assign_patient_splits(
            metadata=metadata,
            train_fraction=float(split_config["train_fraction"]),
            validation_fraction=float(
                split_config["validation_fraction"]
            ),
            test_fraction=float(split_config["test_fraction"]),
            seed=int(project_config["seed"]),
        )

        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_csv(metadata_path, index=False)

    validate_metadata(metadata, require_split=True)

    return metadata


def create_data_loader(
    dataset: HistologyPatchDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """
    Создаёт воспроизводимый DataLoader.
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_data_loader_worker,
        generator=create_torch_generator(seed),
    )


def build_data_bundle(
    config: dict[str, Any],
    runtime: RuntimeDevice,
) -> DataBundle:
    """
    Создаёт datasets и DataLoaders для всех splits.
    """
    metadata = prepare_metadata(config)

    data_config = config["data"]
    augmentation_config = data_config["augmentation"]

    image_size = int(data_config["image_size"])
    root_dir = data_config["root_dir"]

    train_transform = build_train_transforms(
        image_size=image_size,
        horizontal_flip_probability=float(
            augmentation_config["horizontal_flip_probability"]
        ),
        vertical_flip_probability=float(
            augmentation_config["vertical_flip_probability"]
        ),
        use_random_quarter_turn=bool(
            augmentation_config["random_quarter_turn"]
        ),
    )

    evaluation_transform = build_evaluation_transforms(
        image_size=image_size
    )

    train_dataset = HistologyPatchDataset(
        metadata=metadata,
        split="train",
        transform=train_transform,
        root_dir=root_dir,
    )

    validation_dataset = HistologyPatchDataset(
        metadata=metadata,
        split="validation",
        transform=evaluation_transform,
        root_dir=root_dir,
    )

    test_dataset = HistologyPatchDataset(
        metadata=metadata,
        split="test",
        transform=evaluation_transform,
        root_dir=root_dir,
    )

    batch_size = int(data_config["batch_size"])
    num_workers = int(data_config["num_workers"])
    seed = int(config["project"]["seed"])

    train_loader = create_data_loader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=runtime.pin_memory,
        shuffle=True,
        seed=seed,
    )

    validation_loader = create_data_loader(
        dataset=validation_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=runtime.pin_memory,
        shuffle=False,
        seed=seed + 1,
    )

    test_loader = create_data_loader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=runtime.pin_memory,
        shuffle=False,
        seed=seed + 2,
    )

    return DataBundle(
        metadata=metadata,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
    )