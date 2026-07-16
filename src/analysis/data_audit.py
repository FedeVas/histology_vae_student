from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def run_crc_data_audit(
    metadata: pd.DataFrame,
    output_directory: str | Path,
    seed: int,
    verify_images: bool = True,
    verify_max_per_class_and_source: int | None = 200,
    color_statistics_per_class_and_source: int = 200,
    montage_images_per_class: int = 8,
) -> None:
    output_directory = Path(output_directory)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    class_counts = (
        metadata.groupby(
            [
                "source",
                "split",
                "class_code",
                "class_name",
            ],
            dropna=False,
        )
        .size()
        .rename("number_of_images")
        .reset_index()
    )

    class_counts.to_csv(
        output_directory / "class_counts.csv",
        index=False,
    )

    corrupted_images: list[dict[str, str]] = []
    size_records: list[dict[str, object]] = []

    if verify_images:
        verification_metadata = _sample_grouped(
            metadata=metadata,
            maximum_per_group=(
                verify_max_per_class_and_source
            ),
            seed=seed,
        )

        for row in verification_metadata.itertuples(
            index=False
        ):
            try:
                with Image.open(row.path) as image:
                    image.load()

                    size_records.append(
                        {
                            "sample_id": row.sample_id,
                            "source": row.source,
                            "class_code": row.class_code,
                            "width": image.width,
                            "height": image.height,
                            "mode": image.mode,
                        }
                    )

            except (OSError, ValueError) as error:
                corrupted_images.append(
                    {
                        "sample_id": row.sample_id,
                        "path": row.path,
                        "error": str(error),
                    }
                )

    pd.DataFrame(size_records).to_csv(
        output_directory
        / "verified_image_properties.csv",
        index=False,
    )

    pd.DataFrame(
        corrupted_images,
        columns=[
            "sample_id",
            "path",
            "error",
        ],
    ).to_csv(
        output_directory
        / "corrupted_images.csv",
        index=False,
    )

    color_statistics = (
        calculate_color_statistics(
            metadata=metadata,
            maximum_per_group=(
                color_statistics_per_class_and_source
            ),
            seed=seed,
        )
    )

    color_statistics.to_csv(
        output_directory
        / "image_color_statistics.csv",
        index=False,
    )

    color_summary = (
        color_statistics
        .groupby(
            [
                "source",
                "class_code",
            ]
        )
        .agg(
            images=("sample_id", "count"),
            mean_red=("mean_red", "mean"),
            mean_green=("mean_green", "mean"),
            mean_blue=("mean_blue", "mean"),
            mean_brightness=(
                "mean_brightness",
                "mean",
            ),
            mean_contrast=(
                "contrast",
                "mean",
            ),
        )
        .reset_index()
    )

    color_summary.to_csv(
        output_directory
        / "class_color_summary.csv",
        index=False,
    )

    create_class_montage(
        metadata=metadata[
            metadata["split"] == "train"
        ],
        output_path=(
            output_directory
            / "train_class_montage.png"
        ),
        images_per_class=montage_images_per_class,
        seed=seed,
    )

    create_class_montage(
        metadata=metadata[
            metadata["split"] == "test"
        ],
        output_path=(
            output_directory
            / "external_test_class_montage.png"
        ),
        images_per_class=montage_images_per_class,
        seed=seed + 1,
    )

    summary = {
        "total_images": int(len(metadata)),
        "train_images": int(
            (metadata["split"] == "train").sum()
        ),
        "validation_images": int(
            (
                metadata["split"]
                == "validation"
            ).sum()
        ),
        "external_test_images": int(
            (metadata["split"] == "test").sum()
        ),
        "number_of_classes": int(
            metadata["class_code"].nunique()
        ),
        "verified_images": int(
            len(size_records)
        ),
        "corrupted_images": int(
            len(corrupted_images)
        ),
        "patient_mapping_available": False,
        "internal_validation_level": "patch",
        "external_test_patient_disjoint": True,
    }

    with (
        output_directory / "audit_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )


def calculate_color_statistics(
    metadata: pd.DataFrame,
    maximum_per_group: int,
    seed: int,
) -> pd.DataFrame:
    sampled_metadata = _sample_grouped(
        metadata=metadata,
        maximum_per_group=maximum_per_group,
        seed=seed,
    )

    records: list[dict[str, object]] = []

    for row in sampled_metadata.itertuples(
        index=False
    ):
        with Image.open(row.path) as image:
            image_array = np.asarray(
                image.convert("RGB"),
                dtype=np.float32,
            ) / 255.0

        flattened = image_array.reshape(-1, 3)

        brightness = image_array.mean(axis=2)

        records.append(
            {
                "sample_id": row.sample_id,
                "source": row.source,
                "split": row.split,
                "class_code": row.class_code,
                "label": row.label,
                "mean_red": float(
                    flattened[:, 0].mean()
                ),
                "mean_green": float(
                    flattened[:, 1].mean()
                ),
                "mean_blue": float(
                    flattened[:, 2].mean()
                ),
                "mean_brightness": float(
                    brightness.mean()
                ),
                "contrast": float(
                    brightness.std()
                ),
            }
        )

    return pd.DataFrame.from_records(records)


def create_class_montage(
    metadata: pd.DataFrame,
    output_path: str | Path,
    images_per_class: int,
    seed: int,
) -> None:
    class_codes = sorted(
        metadata["class_code"].unique()
    )

    figure, axes = plt.subplots(
        nrows=len(class_codes),
        ncols=images_per_class,
        figsize=(
            images_per_class * 2,
            len(class_codes) * 2,
        ),
        squeeze=False,
    )

    for class_index, class_code in enumerate(
        class_codes
    ):
        class_frame = metadata[
            metadata["class_code"] == class_code
        ]

        number_to_sample = min(
            images_per_class,
            len(class_frame),
        )

        sampled_frame = class_frame.sample(
            n=number_to_sample,
            random_state=seed + class_index,
        )

        for column_index in range(
            images_per_class
        ):
            axis = axes[
                class_index,
                column_index,
            ]

            axis.axis("off")

            if column_index >= number_to_sample:
                continue

            row = sampled_frame.iloc[
                column_index
            ]

            with Image.open(row["path"]) as image:
                axis.imshow(
                    image.convert("RGB")
                )

            if column_index == 0:
                axis.set_title(
                    class_code,
                    loc="left",
                )

    figure.tight_layout()

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)


def _sample_grouped(
    metadata: pd.DataFrame,
    maximum_per_group: int | None,
    seed: int,
) -> pd.DataFrame:
    if maximum_per_group is None:
        return metadata.copy()

    sampled_frames: list[pd.DataFrame] = []

    group_columns = [
        "source",
        "class_code",
    ]

    for group_index, (_, group_frame) in enumerate(
        metadata.groupby(
            group_columns,
            sort=True,
        )
    ):
        sampled_frames.append(
            group_frame.sample(
                n=min(
                    maximum_per_group,
                    len(group_frame),
                ),
                random_state=seed + group_index,
            )
        )

    return pd.concat(
        sampled_frames,
        ignore_index=True,
    )