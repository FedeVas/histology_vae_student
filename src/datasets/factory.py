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
    Загружает готовую metadata.

    Если metadata отсутствует и включён synthetic mode,
    создаёт synthetic dataset.

    Готовые публичные datasets должны предоставлять
    metadata CSV с уже назначенной колонкой split.
    """
    data_config = config["data"]

    synthetic_config = data_config.get(
        "synthetic",
        {},
    )

    metadata_path = Path(
        data_config["metadata_csv"]
    )

    if metadata_path.exists():
        metadata = pd.read_csv(
            metadata_path
        )

    else:
        synthetic_enabled = bool(
            synthetic_config.get(
                "enabled",
                False,
            )
        )

        if not synthetic_enabled:
            raise FileNotFoundError(
                "Metadata file does not exist and "
                "synthetic generation is disabled: "
                f"{metadata_path.resolve()}"
            )

        metadata = (
            generate_synthetic_histology_dataset(
                output_dir=synthetic_config[
                    "output_dir"
                ],
                metadata_path=metadata_path,
                num_patients=int(
                    synthetic_config[
                        "num_patients"
                    ]
                ),
                slides_per_patient=int(
                    synthetic_config[
                        "slides_per_patient"
                    ]
                ),
                patches_per_slide=int(
                    synthetic_config[
                        "patches_per_slide"
                    ]
                ),
                image_size=int(
                    data_config["image_size"]
                ),
                seed=int(
                    config["project"]["seed"]
                ),
            )
        )

    split_config = data_config.get(
        "split",
        {},
    )

    force_reassign = bool(
        split_config.get(
            "force_reassign",
            False,
        )
    )

    split_is_missing = (
        "split" not in metadata.columns
    )

    if split_is_missing or force_reassign:
        if not split_config:
            raise ValueError(
                "Metadata does not contain split and "
                "data.split configuration is missing."
            )

        stratify_column = split_config.get(
            "stratify_column"
        )

        if stratify_column is not None:
            stratify_column = str(
                stratify_column
            )

        metadata = assign_patient_splits(
            metadata=metadata,
            train_fraction=float(
                split_config["train_fraction"]
            ),
            validation_fraction=float(
                split_config[
                    "validation_fraction"
                ]
            ),
            test_fraction=float(
                split_config["test_fraction"]
            ),
            seed=int(
                config["project"]["seed"]
            ),
            stratify_column=stratify_column,
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
    color_mode = str(
        data_config.get(
            "color_mode",
            "rgb",
        )
    )
    augmentation_config = data_config["augmentation"]

    train_transform = build_train_transforms(
        image_size=int(data_config["image_size"]),
        color_mode=color_mode,
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
        image_size=int(data_config["image_size"]),
        color_mode=color_mode,
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


def build_evaluation_data_loader(
    config: dict,
    metadata: pd.DataFrame,
    split: str,
    pin_memory: bool,
) -> DataLoader:
    """
    Создаёт детерминированный DataLoader для извлечения
    embeddings из train, validation или test.

    В отличие от training loader:
    - не применяет случайные augmentation;
    - не перемешивает изображения;
    - не удаляет последний неполный batch.
    """
    allowed_splits = {
        "train",
        "validation",
        "test",
    }

    if split not in allowed_splits:
        raise ValueError(
            f"Unknown split: {split!r}. "
            f"Expected one of {sorted(allowed_splits)}."
        )

    data_config = config["data"]

    evaluation_transform = (
        build_evaluation_transforms(
            image_size=int(
                data_config["image_size"]
            ),
            color_mode=str(
                data_config.get(
                    "color_mode",
                    "rgb",
                )
            )
        )
    )

    dataset = HistologyPatchDataset(
        metadata=metadata,
        split=split,
        transform=evaluation_transform,
    )

    number_of_workers = int(
        data_config["num_workers"]
    )

    return DataLoader(
        dataset=dataset,
        batch_size=int(
            data_config["batch_size"]
        ),
        shuffle=False,
        num_workers=number_of_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(
            number_of_workers > 0
        ),
    )