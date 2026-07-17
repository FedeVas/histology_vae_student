from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare probe predictions using "
            "paired stratified bootstrap."
        )
    )

    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help=(
            "Prediction result in LABEL=PATH format. "
            "May be provided multiple times."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def parse_prediction_specification(
    specification: str,
) -> tuple[str, Path]:
    if "=" not in specification:
        raise ValueError(
            "Prediction specification must use "
            "LABEL=PATH format."
        )

    label, path_value = specification.split(
        "=",
        maxsplit=1,
    )

    label = label.strip()
    path = Path(path_value.strip())

    if not label:
        raise ValueError(
            "Representation label must not be empty."
        )

    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file was not found: "
            f"{path.resolve()}"
        )

    return label, path


def normalize_sample_key(
    value: object,
) -> str:
    return (
        str(value)
        .strip()
        .replace("\\", "/")
        .lower()
    )


def load_prediction_frame(
    label: str,
    path: Path,
) -> pd.DataFrame:
    frame = pd.read_csv(path)

    required_columns = {
        "label",
        "predicted_label",
        "path",
    }

    missing_columns = (
        required_columns.difference(
            frame.columns
        )
    )

    if missing_columns:
        raise ValueError(
            f"{label!r} predictions are missing "
            f"columns: {sorted(missing_columns)}"
        )

    frame = frame.copy()

    frame["sample_key"] = frame[
        "path"
    ].map(normalize_sample_key)

    if frame["sample_key"].duplicated().any():
        duplicate_count = int(
            frame["sample_key"]
            .duplicated()
            .sum()
        )

        raise ValueError(
            f"{label!r} contains "
            f"{duplicate_count} duplicate paths."
        )

    frame["label"] = (
        frame["label"].astype(int)
    )

    frame["predicted_label"] = (
        frame["predicted_label"].astype(int)
    )

    return (
        frame.sort_values("sample_key")
        .reset_index(drop=True)
    )


def validate_prediction_alignment(
    prediction_frames: dict[str, pd.DataFrame],
) -> tuple[np.ndarray, list[int], dict[int, str]]:
    first_label = next(
        iter(prediction_frames)
    )

    reference = prediction_frames[
        first_label
    ]

    reference_keys = reference[
        "sample_key"
    ].tolist()

    reference_targets = reference[
        "label"
    ].to_numpy(dtype=np.int64)

    for label, frame in prediction_frames.items():
        if frame["sample_key"].tolist() != (
            reference_keys
        ):
            raise ValueError(
                "Prediction samples differ between "
                f"{first_label!r} and {label!r}."
            )

        current_targets = frame[
            "label"
        ].to_numpy(dtype=np.int64)

        if not np.array_equal(
            current_targets,
            reference_targets,
        ):
            raise ValueError(
                "True labels differ between "
                f"{first_label!r} and {label!r}."
            )

    labels = sorted(
        np.unique(reference_targets).tolist()
    )

    class_code_mapping: dict[int, str] = {}

    if "class_code" in reference.columns:
        class_table = (
            reference[
                [
                    "label",
                    "class_code",
                ]
            ]
            .drop_duplicates()
        )

        class_code_mapping = {
            int(row["label"]): str(
                row["class_code"]
            )
            for _, row in class_table.iterrows()
        }

    for class_label in labels:
        class_code_mapping.setdefault(
            class_label,
            str(class_label),
        )

    return (
        reference_targets,
        labels,
        class_code_mapping,
    )


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
    }


def calculate_per_class_metrics(
    representation: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int],
    class_code_mapping: dict[int, str],
) -> pd.DataFrame:
    (
        precision,
        recall,
        f1,
        support,
    ) = precision_recall_fscore_support(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        zero_division=0,
    )

    records: list[dict[str, object]] = []

    for index, label in enumerate(labels):
        records.append(
            {
                "representation": (
                    representation
                ),
                "label": int(label),
                "class_code": (
                    class_code_mapping[label]
                ),
                "precision": float(
                    precision[index]
                ),
                "recall": float(
                    recall[index]
                ),
                "f1": float(
                    f1[index]
                ),
                "support": int(
                    support[index]
                ),
            }
        )

    return pd.DataFrame.from_records(
        records
    )


def create_stratified_bootstrap_indices(
    y_true: np.ndarray,
    labels: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    sampled_indices: list[np.ndarray] = []

    for label in labels:
        class_indices = np.flatnonzero(
            y_true == label
        )

        sampled_class_indices = rng.choice(
            class_indices,
            size=len(class_indices),
            replace=True,
        )

        sampled_indices.append(
            sampled_class_indices
        )

    return np.concatenate(
        sampled_indices
    )


def calculate_paired_bootstrap(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    labels: list[int],
    number_of_iterations: int,
    seed: int,
) -> pd.DataFrame:
    if number_of_iterations <= 0:
        raise ValueError(
            "number_of_iterations must be positive."
        )

    rng = np.random.default_rng(seed)

    metric_names = (
        "balanced_accuracy",
        "macro_f1",
    )

    bootstrap_values = {
        representation: {
            metric_name: np.empty(
                number_of_iterations,
                dtype=np.float64,
            )
            for metric_name in metric_names
        }
        for representation in predictions
    }

    for iteration in range(
        number_of_iterations
    ):
        sampled_indices = (
            create_stratified_bootstrap_indices(
                y_true=y_true,
                labels=labels,
                rng=rng,
            )
        )

        sampled_targets = y_true[
            sampled_indices
        ]

        for (
            representation,
            representation_predictions,
        ) in predictions.items():
            sampled_predictions = (
                representation_predictions[
                    sampled_indices
                ]
            )

            metrics = calculate_metrics(
                y_true=sampled_targets,
                y_pred=sampled_predictions,
            )

            for metric_name in metric_names:
                bootstrap_values[
                    representation
                ][metric_name][
                    iteration
                ] = metrics[metric_name]

    records: list[dict[str, object]] = []

    for first, second in combinations(
        predictions.keys(),
        2,
    ):
        for metric_name in metric_names:
            differences = (
                bootstrap_values[first][
                    metric_name
                ]
                - bootstrap_values[second][
                    metric_name
                ]
            )

            records.append(
                {
                    "first_representation": first,
                    "second_representation": second,
                    "metric": metric_name,
                    "mean_difference": float(
                        differences.mean()
                    ),
                    "ci_2_5": float(
                        np.quantile(
                            differences,
                            0.025,
                        )
                    ),
                    "ci_97_5": float(
                        np.quantile(
                            differences,
                            0.975,
                        )
                    ),
                    "probability_first_better": (
                        float(
                            np.mean(
                                differences > 0
                            )
                        )
                    ),
                    "bootstrap_iterations": int(
                        number_of_iterations
                    ),
                }
            )

    return pd.DataFrame.from_records(
        records
    )


def save_per_class_f1_plot(
    per_class_metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    class_codes = (
        per_class_metrics[
            [
                "label",
                "class_code",
            ]
        ]
        .drop_duplicates()
        .sort_values("label")[
            "class_code"
        ]
        .tolist()
    )

    representations = (
        per_class_metrics[
            "representation"
        ]
        .drop_duplicates()
        .tolist()
    )

    x_positions = np.arange(
        len(class_codes),
        dtype=np.float64,
    )

    bar_width = (
        0.8 / len(representations)
    )

    figure, axis = plt.subplots(
        figsize=(12, 6)
    )

    for representation_index, (
        representation
    ) in enumerate(representations):
        representation_frame = (
            per_class_metrics[
                per_class_metrics[
                    "representation"
                ]
                == representation
            ]
            .sort_values("label")
        )

        offset = (
            representation_index
            - (len(representations) - 1) / 2
        ) * bar_width

        axis.bar(
            x_positions + offset,
            representation_frame["f1"],
            width=bar_width,
            label=representation,
        )

    axis.set_xticks(
        x_positions
    )

    axis.set_xticklabels(
        class_codes
    )

    axis.set_ylim(
        0.0,
        1.0,
    )

    axis.set_xlabel(
        "Tissue class"
    )

    axis.set_ylabel(
        "F1 score"
    )

    axis.set_title(
        "External test F1 by tissue class"
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    arguments = parse_arguments()

    prediction_frames: dict[
        str,
        pd.DataFrame,
    ] = {}

    for specification in (
        arguments.prediction
    ):
        label, path = (
            parse_prediction_specification(
                specification
            )
        )

        if label in prediction_frames:
            raise ValueError(
                "Duplicate representation label: "
                f"{label!r}."
            )

        prediction_frames[label] = (
            load_prediction_frame(
                label=label,
                path=path,
            )
        )

    if len(prediction_frames) < 2:
        raise ValueError(
            "At least two prediction files "
            "are required."
        )

    (
        y_true,
        labels,
        class_code_mapping,
    ) = validate_prediction_alignment(
        prediction_frames
    )

    predictions = {
        representation: frame[
            "predicted_label"
        ].to_numpy(dtype=np.int64)
        for representation, frame
        in prediction_frames.items()
    }

    metric_records: list[
        dict[str, object]
    ] = []

    per_class_frames: list[
        pd.DataFrame
    ] = []

    for (
        representation,
        representation_predictions,
    ) in predictions.items():
        metric_records.append(
            {
                "representation": (
                    representation
                ),
                **calculate_metrics(
                    y_true=y_true,
                    y_pred=(
                        representation_predictions
                    ),
                ),
            }
        )

        per_class_frames.append(
            calculate_per_class_metrics(
                representation=representation,
                y_true=y_true,
                y_pred=(
                    representation_predictions
                ),
                labels=labels,
                class_code_mapping=(
                    class_code_mapping
                ),
            )
        )

    representation_metrics = (
        pd.DataFrame.from_records(
            metric_records
        )
    )

    per_class_metrics = pd.concat(
        per_class_frames,
        ignore_index=True,
    )

    bootstrap_results = (
        calculate_paired_bootstrap(
            y_true=y_true,
            predictions=predictions,
            labels=labels,
            number_of_iterations=(
                arguments.bootstrap_iterations
            ),
            seed=arguments.seed,
        )
    )

    best_model_by_class = (
        per_class_metrics
        .sort_values(
            [
                "label",
                "f1",
                "recall",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
        .groupby(
            "label",
            as_index=False,
        )
        .first()
    )

    arguments.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    representation_metrics.to_csv(
        arguments.output_dir
        / "representation_metrics.csv",
        index=False,
    )

    per_class_metrics.to_csv(
        arguments.output_dir
        / "per_class_metrics.csv",
        index=False,
    )

    bootstrap_results.to_csv(
        arguments.output_dir
        / "paired_bootstrap.csv",
        index=False,
    )

    best_model_by_class.to_csv(
        arguments.output_dir
        / "best_model_by_class.csv",
        index=False,
    )

    save_per_class_f1_plot(
        per_class_metrics=per_class_metrics,
        output_path=(
            arguments.output_dir
            / "per_class_f1.png"
        ),
    )

    print("=" * 88)
    print("PAIRED PROBE PREDICTION COMPARISON")
    print("=" * 88)

    print()
    print("Overall external-test metrics")
    print(
        representation_metrics.to_string(
            index=False
        )
    )

    print()
    print("Best representation by class")
    print(
        best_model_by_class[
            [
                "class_code",
                "representation",
                "precision",
                "recall",
                "f1",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("Paired bootstrap differences")
    print(
        bootstrap_results.to_string(
            index=False
        )
    )

    print()
    print(
        f"Results: "
        f"{arguments.output_dir.resolve()}"
    )

    print("=" * 88)


if __name__ == "__main__":
    main()