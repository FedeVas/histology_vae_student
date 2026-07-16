from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check feature split integrity "
            "before probe evaluation."
        )
    )

    parser.add_argument(
        "--train",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--validation",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--test",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--feature-prefix",
        type=str,
        required=True,
    )

    return parser.parse_args()


def get_feature_columns(
    frame: pd.DataFrame,
    prefix: str,
) -> list[str]:
    return [
        column
        for column in frame.columns
        if column.startswith(prefix)
    ]


def check_overlap(
    first: pd.DataFrame,
    second: pd.DataFrame,
    first_name: str,
    second_name: str,
    column: str,
) -> int:
    if (
        column not in first.columns
        or column not in second.columns
    ):
        return 0

    first_values = set(
        first[column]
        .dropna()
        .astype(str)
    )

    second_values = set(
        second[column]
        .dropna()
        .astype(str)
    )

    overlap = first_values.intersection(
        second_values
    )

    print(
        f"{column} overlap "
        f"{first_name}/{second_name}: "
        f"{len(overlap)}"
    )

    return len(overlap)


def main() -> None:
    arguments = parse_arguments()

    frames = {
        "train": pd.read_csv(
            arguments.train
        ),
        "validation": pd.read_csv(
            arguments.validation
        ),
        "test": pd.read_csv(
            arguments.test
        ),
    }

    reference_columns = get_feature_columns(
        frames["train"],
        arguments.feature_prefix,
    )

    if not reference_columns:
        raise ValueError(
            "No feature columns found for prefix "
            f"{arguments.feature_prefix!r}."
        )

    print("=" * 72)
    print("PROBE FEATURE INTEGRITY CHECK")
    print("=" * 72)

    print(
        f"Feature prefix:     "
        f"{arguments.feature_prefix}"
    )

    print(
        f"Feature dimensions: "
        f"{len(reference_columns)}"
    )

    print()

    for split_name, frame in frames.items():
        current_columns = (
            get_feature_columns(
                frame,
                arguments.feature_prefix,
            )
        )

        if current_columns != reference_columns:
            raise ValueError(
                "Feature columns differ for split "
                f"{split_name!r}."
            )

        feature_values = frame[
            reference_columns
        ].to_numpy(
            dtype=np.float64
        )

        if not np.isfinite(
            feature_values
        ).all():
            raise ValueError(
                "Non-finite feature values in split "
                f"{split_name!r}."
            )

        print(
            f"{split_name:12s} "
            f"rows={len(frame):5d} "
            f"classes={frame['label'].nunique():2d}"
        )

        if "source" in frame.columns:
            print(
                f"  sources: "
                f"{sorted(frame['source'].unique())}"
            )

        class_counts = (
            frame.groupby("label")
            .size()
            .to_dict()
        )

        print(
            f"  class counts: "
            f"{class_counts}"
        )

    print()
    overlap_total = 0

    split_pairs = [
        (
            "train",
            "validation",
        ),
        (
            "train",
            "test",
        ),
        (
            "validation",
            "test",
        ),
    ]

    for first_name, second_name in split_pairs:
        for column in (
            "sample_id",
            "path",
        ):
            overlap_total += check_overlap(
                first=frames[first_name],
                second=frames[second_name],
                first_name=first_name,
                second_name=second_name,
                column=column,
            )

    print()

    if overlap_total > 0:
        raise RuntimeError(
            "Metadata overlap was detected "
            "between feature splits."
        )

    print("No sample ID or path overlap detected.")
    print("=" * 72)


if __name__ == "__main__":
    main()