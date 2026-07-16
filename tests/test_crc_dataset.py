from pathlib import Path

from PIL import Image

from src.datasets.crc import (
    CRC_TISSUE_CLASSES,
    build_crc_metadata,
    scan_crc_directory,
)


def create_fake_crc_directory(
    root_directory: Path,
    images_per_class: int,
) -> None:
    for tissue_class in CRC_TISSUE_CLASSES:
        class_directory = (
            root_directory
            / tissue_class.code
        )

        class_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        for image_index in range(
            images_per_class
        ):
            image = Image.new(
                mode="RGB",
                size=(32, 32),
                color=(
                    100 + image_index,
                    80,
                    120,
                ),
            )

            image.save(
                class_directory
                / (
                    f"{tissue_class.code}-"
                    f"{image_index:04d}.tif"
                )
            )


def test_scan_crc_directory_detects_all_classes(
    tmp_path: Path,
) -> None:
    dataset_directory = (
        tmp_path / "crc"
    )

    create_fake_crc_directory(
        root_directory=dataset_directory,
        images_per_class=2,
    )

    metadata = scan_crc_directory(
        root_directory=dataset_directory,
        source_name="test_source",
    )

    assert len(metadata) == 18
    assert metadata["class_code"].nunique() == 9
    assert metadata["label"].nunique() == 9


def test_crc_metadata_contains_balanced_splits(
    tmp_path: Path,
) -> None:
    train_directory = (
        tmp_path / "train"
    )

    test_directory = (
        tmp_path / "external_test"
    )

    create_fake_crc_directory(
        root_directory=train_directory,
        images_per_class=10,
    )

    create_fake_crc_directory(
        root_directory=test_directory,
        images_per_class=5,
    )

    metadata = build_crc_metadata(
        train_root=train_directory,
        external_test_root=test_directory,
        validation_fraction=0.2,
        seed=42,
        train_pool_per_class=None,
        external_test_per_class=None,
    )

    counts = (
        metadata.groupby(
            [
                "split",
                "class_code",
            ]
        )
        .size()
        .unstack(fill_value=0)
    )

    assert (
        counts.loc["train"] == 8
    ).all()

    assert (
        counts.loc["validation"] == 2
    ).all()

    assert (
        counts.loc["test"] == 5
    ).all()


def test_crc_metadata_is_reproducible(
    tmp_path: Path,
) -> None:
    train_directory = (
        tmp_path / "train"
    )

    test_directory = (
        tmp_path / "test"
    )

    create_fake_crc_directory(
        root_directory=train_directory,
        images_per_class=10,
    )

    create_fake_crc_directory(
        root_directory=test_directory,
        images_per_class=5,
    )

    first_metadata = build_crc_metadata(
        train_root=train_directory,
        external_test_root=test_directory,
        validation_fraction=0.2,
        seed=42,
    )

    second_metadata = build_crc_metadata(
        train_root=train_directory,
        external_test_root=test_directory,
        validation_fraction=0.2,
        seed=42,
    )

    assert (
        first_metadata[
            [
                "sample_id",
                "split",
            ]
        ]
        .equals(
            second_metadata[
                [
                    "sample_id",
                    "split",
                ]
            ]
        )
    )