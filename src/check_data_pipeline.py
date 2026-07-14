from __future__ import annotations

from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from src.datasets.patch_dataset import HistologyPatchDataset
from src.datasets.split import (
    assign_patient_splits,
    get_split_summary,
    validate_metadata,
)
from src.datasets.synthetic import (
    generate_synthetic_histology_dataset,
)
from src.datasets.transforms import (
    build_evaluation_transforms,
    build_train_transforms,
)
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.reproducibility import (
    create_torch_generator,
    seed_data_loader_worker,
    seed_everything,
)


CONFIG_PATH = Path("configs/vae_base.yaml")


def create_metadata_if_required(
    config: dict,
) -> pd.DataFrame:
    data_config = config["data"]
    synthetic_config = data_config["synthetic"]

    metadata_path = Path(data_config["metadata_csv"])

    if metadata_path.exists():
        print(f"Loading metadata: {metadata_path}")
        return pd.read_csv(metadata_path)

    if not synthetic_config["enabled"]:
        raise FileNotFoundError(
            f"Metadata file does not exist: {metadata_path.resolve()}"
        )

    print("Synthetic metadata was not found.")
    print("Generating synthetic histology-like patches...")

    return generate_synthetic_histology_dataset(
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


def add_patient_splits_if_required(
    metadata: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    metadata_path = Path(config["data"]["metadata_csv"])

    if "split" in metadata.columns:
        validate_metadata(
            metadata,
            require_split=True,
        )
        return metadata

    split_config = config["data"]["split"]

    print("Assigning patient-level splits...")

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

    return metadata


def create_data_loader(
    dataset: HistologyPatchDataset,
    config: dict,
    pin_memory: bool,
    shuffle: bool,
) -> DataLoader:
    data_config = config["data"]
    seed = int(config["project"]["seed"])
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


def save_batch_preview(
    images,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    number_of_images = min(
        len(images),
        16,
    )

    image_grid = make_grid(
        images[:number_of_images],
        nrow=4,
        padding=2,
    )

    save_image(
        image_grid,
        output_path,
    )


def main() -> None:
    config = load_config(CONFIG_PATH)

    seed_everything(
        seed=int(config["project"]["seed"]),
        deterministic=bool(
            config["device"]["deterministic"]
        ),
    )

    runtime = resolve_device(
        accelerator=str(
            config["device"]["accelerator"]
        ),
        mixed_precision=config["training"][
            "mixed_precision"
        ],
        pin_memory=config["data"]["pin_memory"],
    )

    metadata = create_metadata_if_required(config)
    metadata = add_patient_splits_if_required(
        metadata=metadata,
        config=config,
    )

    print()
    print("Dataset split summary")
    print(get_split_summary(metadata).to_string(index=False))
    print()

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
            augmentation_config[
                "random_quarter_turn"
            ]
        ),
    )

    evaluation_transform = build_evaluation_transforms(
        image_size=int(data_config["image_size"])
    )

    train_dataset = HistologyPatchDataset(
        metadata=metadata,
        split="train",
        transform=train_transform,
        root_dir=data_config["root_dir"],
    )

    validation_dataset = HistologyPatchDataset(
        metadata=metadata,
        split="validation",
        transform=evaluation_transform,
        root_dir=data_config["root_dir"],
    )

    test_dataset = HistologyPatchDataset(
        metadata=metadata,
        split="test",
        transform=evaluation_transform,
        root_dir=data_config["root_dir"],
    )

    train_loader = create_data_loader(
        dataset=train_dataset,
        config=config,
        pin_memory=runtime.pin_memory,
        shuffle=True,
    )

    train_batch = next(iter(train_loader))
    images = train_batch["image"]

    print("Train batch check")
    print(f"Image tensor shape: {tuple(images.shape)}")
    print(f"Image dtype:        {images.dtype}")
    print(f"Minimum value:      {images.min().item():.4f}")
    print(f"Maximum value:      {images.max().item():.4f}")
    print(
        "Patient examples:   "
        f"{train_batch['patient_id'][:3]}"
    )
    print(
        "Slide examples:     "
        f"{train_batch['slide_id'][:3]}"
    )
    print()

    expected_shape = (
        images.shape[0],
        int(data_config["channels"]),
        int(data_config["image_size"]),
        int(data_config["image_size"]),
    )

    if tuple(images.shape) != expected_shape:
        raise RuntimeError(
            "Unexpected batch shape. "
            f"Expected {expected_shape}, "
            f"received {tuple(images.shape)}."
        )

    if images.min().item() < 0.0:
        raise RuntimeError(
            "Image values must not be below 0."
        )

    if images.max().item() > 1.0:
        raise RuntimeError(
            "Image values must not exceed 1."
        )

    output_path = (
        Path(config["project"]["output_dir"])
        / "data_checks"
        / "train_batch.png"
    )

    save_batch_preview(
        images=images,
        output_path=output_path,
    )

    print(f"Train samples:      {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")
    print(f"Test samples:       {len(test_dataset)}")
    print(f"Batch preview:      {output_path.resolve()}")
    print()
    print("Data pipeline check completed successfully.")


if __name__ == "__main__":
    main()