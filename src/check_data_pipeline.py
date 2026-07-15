from pathlib import Path

from torchvision.utils import make_grid, save_image

from src.datasets.factory import (
    build_data_loaders,
    build_datasets,
    prepare_metadata,
)
from src.datasets.split import get_split_label_summary
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.reproducibility import seed_everything


CONFIG_PATH = Path("configs/vae_base.yaml")


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

    metadata = prepare_metadata(config)
    datasets = build_datasets(
        config=config,
        metadata=metadata,
    )

    data_loaders = build_data_loaders(
        config=config,
        datasets=datasets,
        pin_memory=runtime.pin_memory,
    )

    print()
    print("Dataset split summary")
    print(get_split_label_summary(metadata).to_string(index=False))
    print()

    batch = next(iter(data_loaders.train))
    images = batch["image"]

    expected_shape = (
        images.shape[0],
        int(config["data"]["channels"]),
        int(config["data"]["image_size"]),
        int(config["data"]["image_size"]),
    )

    if tuple(images.shape) != expected_shape:
        raise RuntimeError(
            "Unexpected batch shape. "
            f"Expected {expected_shape}, "
            f"received {tuple(images.shape)}."
        )

    if images.min().item() < 0.0:
        raise RuntimeError(
            "Image values must not be below zero."
        )

    if images.max().item() > 1.0:
        raise RuntimeError(
            "Image values must not exceed one."
        )

    output_path = (
        Path(config["project"]["output_dir"])
        / "data_checks"
        / "train_batch.png"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    grid = make_grid(
        images[:16],
        nrow=4,
        padding=2,
    )

    save_image(
        grid,
        output_path,
    )

    print("Train batch check")
    print(f"Image tensor shape: {tuple(images.shape)}")
    print(f"Image dtype:        {images.dtype}")
    print(f"Minimum value:      {images.min().item():.4f}")
    print(f"Maximum value:      {images.max().item():.4f}")
    print()

    print(f"Train samples:      {len(datasets.train)}")
    print(f"Validation samples: {len(datasets.validation)}")
    print(f"Test samples:       {len(datasets.test)}")
    print(f"Batch preview:      {output_path.resolve()}")
    print()
    print("Data pipeline check completed successfully.")


if __name__ == "__main__":
    main()