from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.analysis.nuclei_segmentation import (
    NucleiSegmentationConfig,
    extract_nuclei_feature_frame,
    get_nuclei_feature_columns,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract classical nuclei-segmentation "
            "features (OpenCV + scikit-image) from "
            "histology images."
        )
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--min-nucleus-area-px",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--watershed-min-distance",
        type=int,
        default=4,
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    metadata = pd.read_csv(
        arguments.metadata
    )

    config = NucleiSegmentationConfig(
        sample_size=arguments.sample_size,
        min_nucleus_area_px=(
            arguments.min_nucleus_area_px
        ),
        watershed_min_distance=(
            arguments.watershed_min_distance
        ),
    )

    feature_frame = extract_nuclei_feature_frame(
        metadata=metadata,
        config=config,
    )

    arguments.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_features_path = (
        arguments.output_dir
        / "all_segmentation_features.csv"
    )

    feature_frame.to_csv(
        all_features_path,
        index=False,
    )

    split_counts: dict[str, int] = {}

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        split_frame = feature_frame[
            feature_frame["split"]
            == split_name
        ].copy()

        if split_frame.empty:
            raise ValueError(
                f"Split {split_name!r} is empty."
            )

        split_path = (
            arguments.output_dir
            / f"{split_name}_features.csv"
        )

        split_frame.to_csv(
            split_path,
            index=False,
        )

        split_counts[split_name] = int(
            len(split_frame)
        )

    feature_columns = (
        get_nuclei_feature_columns(
            feature_frame
        )
    )

    summary = {
        "metadata_path": str(
            arguments.metadata.resolve()
        ),
        "number_of_images": int(
            len(feature_frame)
        ),
        "number_of_features": int(
            len(feature_columns)
        ),
        "feature_prefix": "segmentation_",
        "sample_size": int(
            arguments.sample_size
        ),
        "min_nucleus_area_px": int(
            arguments.min_nucleus_area_px
        ),
        "watershed_min_distance": int(
            arguments.watershed_min_distance
        ),
        "split_counts": split_counts,
        "method": (
            "classical: hematoxylin color "
            "deconvolution + Otsu threshold + "
            "watershed splitting"
        ),
    }

    with (
        arguments.output_dir
        / "feature_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print("=" * 68)
    print("CRC NUCLEI SEGMENTATION FEATURE EXTRACTION")
    print("=" * 68)

    print(
        f"Images:              "
        f"{len(feature_frame)}"
    )

    print(
        f"Segmentation features: "
        f"{len(feature_columns)}"
    )

    print()
    print(
        feature_frame.groupby("split")[
            "segmentation_nuclei_count"
        ]
        .mean()
        .rename("mean_nuclei_count")
        .to_string()
    )

    print()
    print(
        f"Results: "
        f"{arguments.output_dir.resolve()}"
    )

    print("=" * 68)


if __name__ == "__main__":
    main()
