from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare linear probe results for "
            "different feature representations."
        )
    )

    parser.add_argument(
        "--result",
        action="append",
        required=True,
        help=(
            "Result in LABEL=PATH format. "
            "May be specified multiple times."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--number-of-classes",
        type=int,
        default=9,
    )

    return parser.parse_args()


def load_result(
    specification: str,
) -> dict[str, object]:
    if "=" not in specification:
        raise ValueError(
            "Each result must have LABEL=PATH format."
        )

    label, path_value = specification.split(
        "=",
        maxsplit=1,
    )

    label = label.strip()
    metrics_path = Path(
        path_value.strip()
    )

    if not label:
        raise ValueError(
            "Representation label must not be empty."
        )

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: "
            f"{metrics_path.resolve()}"
        )

    with metrics_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metrics = json.load(file)

    feature_dimensions = metrics.get(
        "feature_dimensions"
    )

    if feature_dimensions is None:
        feature_dimensions = metrics.get(
            "latent_dimensions"
        )

    return {
        "representation": label,
        "feature_prefix": metrics.get(
            "feature_prefix",
            "latent_",
        ),
        "feature_dimensions": (
            feature_dimensions
        ),
        "best_c": metrics["best_c"],
        "validation_balanced_accuracy": (
            metrics["validation"][
                "balanced_accuracy"
            ]
        ),
        "external_balanced_accuracy": (
            metrics["external_test"][
                "balanced_accuracy"
            ]
        ),
        "external_macro_f1": (
            metrics["external_test"][
                "macro_f1"
            ]
        ),
        "external_accuracy": (
            metrics["external_test"][
                "accuracy"
            ]
        ),
        "balanced_accuracy_gap": (
            metrics[
                "balanced_accuracy_gap"
            ]
        ),
    }


def save_metric_plot(
    comparison: pd.DataFrame,
    metric_column: str,
    y_label: str,
    title: str,
    random_baseline: float,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    axis.bar(
        comparison["representation"],
        comparison[metric_column],
    )

    axis.axhline(
        random_baseline,
        linestyle="--",
        label=(
            f"Random class baseline "
            f"({random_baseline:.3f})"
        ),
    )

    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.set_ylim(
        0.0,
        1.0,
    )

    axis.tick_params(
        axis="x",
        rotation=15,
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=170,
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    arguments = parse_arguments()

    if arguments.number_of_classes <= 1:
        raise ValueError(
            "number_of_classes must exceed one."
        )

    records = [
        load_result(specification)
        for specification in arguments.result
    ]

    comparison = pd.DataFrame.from_records(
        records
    )

    arguments.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        arguments.output_dir
        / "probe_comparison.csv",
        index=False,
    )

    random_baseline = (
        1.0 / arguments.number_of_classes
    )

    save_metric_plot(
        comparison=comparison,
        metric_column=(
            "external_balanced_accuracy"
        ),
        y_label="Balanced accuracy",
        title=(
            "External test balanced accuracy"
        ),
        random_baseline=random_baseline,
        output_path=(
            arguments.output_dir
            / "external_balanced_accuracy.png"
        ),
    )

    save_metric_plot(
        comparison=comparison,
        metric_column="external_macro_f1",
        y_label="Macro-F1",
        title="External test macro-F1",
        random_baseline=random_baseline,
        output_path=(
            arguments.output_dir
            / "external_macro_f1.png"
        ),
    )

    print("=" * 72)
    print("REPRESENTATION PROBE COMPARISON")
    print("=" * 72)

    print(
        comparison[
            [
                "representation",
                "feature_dimensions",
                "validation_balanced_accuracy",
                "external_balanced_accuracy",
                "external_macro_f1",
                "balanced_accuracy_gap",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Random class baseline: "
        f"{random_baseline:.4f}"
    )

    print(
        f"Results: "
        f"{arguments.output_dir.resolve()}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()