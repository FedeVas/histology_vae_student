from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.analysis.color_features import (
    ColorFeatureConfig,
    extract_color_feature_frame,
    get_color_feature_columns,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract spatially invariant RGB/HSV "
            "features from histology images."
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
        default=64,
    )

    parser.add_argument(
        "--rgb-bins",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--hue-bins",
        type=int,
        default=18,
    )

    parser.add_argument(
        "--sv-bins",
        type=int,
        default=16,
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    metadata = pd.read_csv(
        arguments.metadata
    )

    config = ColorFeatureConfig(
        rgb_histogram_bins=(
            arguments.rgb_bins
        ),
        hue_histogram_bins=(
            arguments.hue_bins
        ),
        saturation_value_histogram_bins=(
            arguments.sv_bins
        ),
    )

    feature_frame = extract_color_feature_frame(
        metadata=metadata,
        sample_size=arguments.sample_size,
        config=config,
    )

    arguments.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_features_path = (
        arguments.output_dir
        / "all_color_features.csv"
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
        get_color_feature_columns(
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
        "feature_prefix": "color_",
        "sample_size": int(
            arguments.sample_size
        ),
        "rgb_histogram_bins": int(
            arguments.rgb_bins
        ),
        "hue_histogram_bins": int(
            arguments.hue_bins
        ),
        "saturation_value_histogram_bins": int(
            arguments.sv_bins
        ),
        "split_counts": split_counts,
        "spatial_information_used": False,
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
    print("CRC COLOR FEATURE EXTRACTION")
    print("=" * 68)

    print(
        f"Images:              "
        f"{len(feature_frame)}"
    )

    print(
        f"Color features:      "
        f"{len(feature_columns)}"
    )

    print(
        f"Sampling size:       "
        f"{arguments.sample_size} x "
        f"{arguments.sample_size}"
    )

    print()
    print(
        feature_frame.groupby(
            [
                "split",
                "class_code",
            ]
        )
        .size()
        .rename("images")
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