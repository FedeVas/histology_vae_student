from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from src.datasets.patch_dataset import HistologyPatchDataset
from src.datasets.split import (
    assign_patient_splits,
    validate_metadata,
)
from src.datasets.synthetic import (
    generate_synthetic_histology_dataset,
)
from src.datasets.transforms import (
    build_evaluation_transforms,
    build_train_transforms,
)
from src.utils.reproducibility import (
    create_torch_generator,
    seed_data_loader_worker,
)


@dataclass(frozen=True)
class DatasetBundle:
    train: HistologyPatchDataset
    validation: HistologyPatchDataset
    test: HistologyPatchDataset


@dataclass(frozen=True)
class DataLoaderBundle:
    train: DataLoader
    validation: DataLoader
    test: DataLoader


def prepare_metadata(config: dict) -> pd.DataFrame:
    """
    Загружает metadata или создаёт synthetic dataset.

    Если split ещё отсутствует, назначает его на уровне пациентов.
    """
    data_config = config["data"]
    synthetic_config = data_config["synthetic"]

    metadata_path = Path(data_config["metadata_csv"])

    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
    else:
        if not bool(synthetic_config["enabled"]):
            raise FileNotFoundError(
                "Metadata file does not exist and synthetic generation "
                f"is disabled: {metadata_path.resolve()}"
            )

        metadata = generate_synthetic_histology_dataset(
            output_dir=synthetic_config["output_dir"],
            metadata_path=metadata_path,
            num_patients=int(
                synthetic_config["num_patients"]
            ),
            slides_per_patient=int(
                synthetic_config["slides_per_patient"]
            ),
            patches_per_slide=int(
                synthetic_config["patches_per_slide"]
            ),
            image_size=int(data_config["image_size"]),
            seed=int(config["project"]["seed"]),
        )

    if "split" not in metadata.columns:
        split_config = data_config["split"]

        metadata = assign_patient_splits(
            metadata=metadata,
            train_fraction=float(
                split_config["train_fraction"]
            ),
            validation_fraction=float(
                split_config["validation_fraction"]
            ),
            test_fraction=float(
                split_config["test_fraction"]
            ),
            seed=int(config["project"]["seed"]),
        )

        metadata_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata.to_csv(
            metadata_path,
            index=False,
        )

    validate_metadata(
        metadata,
        require_split=True,
    )

    return metadata


def build_datasets(
    config: dict,
    metadata: pd.DataFrame,
) -> DatasetBundle:
    """
    Создаёт train, validation и test datasets.
    """
    data_config = config["data"]
    augmentation_config = data_config["augmentation"]

    train_transform = build_train_transforms(
        image_size=int(data_config["image_size"]),
        horizontal_flip_probability=float(
            augmentation_config[
                "horizontal_flip_probability"
            ]
        ),
        vertical_flip_probability=float(
            augmentation_config[
                "vertical_flip_probability"
            ]
        ),
        use_random_quarter_turn=bool(
            augmentation_config["random_quarter_turn"]
        ),
    )

    evaluation_transform = build_evaluation_transforms(
        image_size=int(data_config["image_size"])
    )

    common_arguments = {
        "metadata": metadata,
        "root_dir": data_config["root_dir"],
    }

    return DatasetBundle(
        train=HistologyPatchDataset(
            split="train",
            transform=train_transform,
            **common_arguments,
        ),
        validation=HistologyPatchDataset(
            split="validation",
            transform=evaluation_transform,
            **common_arguments,
        ),
        test=HistologyPatchDataset(
            split="test",
            transform=evaluation_transform,
            **common_arguments,
        ),
    )


def build_data_loaders(
    config: dict,
    datasets: DatasetBundle,
    pin_memory: bool,
) -> DataLoaderBundle:
    """
    Создаёт DataLoaders для всех splits.
    """
    seed = int(config["project"]["seed"])

    return DataLoaderBundle(
        train=_create_data_loader(
            dataset=datasets.train,
            config=config,
            pin_memory=pin_memory,
            shuffle=True,
            seed=seed,
        ),
        validation=_create_data_loader(
            dataset=datasets.validation,
            config=config,
            pin_memory=pin_memory,
            shuffle=False,
            seed=seed + 1,
        ),
        test=_create_data_loader(
            dataset=datasets.test,
            config=config,
            pin_memory=pin_memory,
            shuffle=False,
            seed=seed + 2,
        ),
    )


def _create_data_loader(
    dataset: HistologyPatchDataset,
    config: dict,
    pin_memory: bool,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    data_config = config["data"]

    number_of_workers = int(
        data_config["num_workers"]
    )

    return DataLoader(
        dataset=dataset,
        batch_size=int(data_config["batch_size"]),
        shuffle=shuffle,
        num_workers=number_of_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=number_of_workers > 0,
        worker_init_fn=seed_data_loader_worker,
        generator=create_torch_generator(seed),
    )