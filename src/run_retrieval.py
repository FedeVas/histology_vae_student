from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.analysis.retrieval import (
    evaluate_nearest_neighbor_retrieval,
)
from src.analysis.retrieval_plots import (
    save_retrieval_montages,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve nearest train histology patches "
            "for external query embeddings."
        )
    )

    parser.add_argument(
        "--train-embeddings",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--query-embeddings",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[
            1,
            3,
            5,
        ],
    )

    parser.add_argument(
        "--metric",
        type=str,
        choices=[
            "euclidean",
            "cosine",
        ],
        default="euclidean",
    )

    parser.add_argument(
        "--queries-per-class",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--montage-neighbors",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    train_embeddings = pd.read_csv(
        arguments.train_embeddings
    )

    query_embeddings = pd.read_csv(
        arguments.query_embeddings
    )

    result = (
        evaluate_nearest_neighbor_retrieval(
            train_embeddings=train_embeddings,
            query_embeddings=query_embeddings,
            k_values=arguments.k_values,
            metric=arguments.metric,
        )
    )

    arguments.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.neighbors.to_csv(
        arguments.output_dir
        / "neighbors.csv",
        index=False,
    )

    result.query_summary.to_csv(
        arguments.output_dir
        / "query_summary.csv",
        index=False,
    )

    result.per_class_metrics.to_csv(
        arguments.output_dir
        / "per_class_metrics.csv",
        index=False,
    )

    with (
        arguments.output_dir
        / "metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result.metrics,
            file,
            indent=2,
        )

    montage_directory = (
        arguments.output_dir
        / "montages"
    )

    save_retrieval_montages(
        neighbors=result.neighbors,
        output_directory=(
            montage_directory
        ),
        neighbors_per_query=(
            arguments.montage_neighbors
        ),
        queries_per_class=(
            arguments.queries_per_class
        ),
        seed=arguments.seed,
    )

    print("=" * 72)
    print("CRC LATENT NEAREST-NEIGHBOR RETRIEVAL")
    print("=" * 72)

    print(
        f"Metric:               "
        f"{result.metrics['metric']}"
    )

    print(
        f"Train images:          "
        f"{result.metrics['number_of_train_images']}"
    )

    print(
        f"External queries:      "
        f"{result.metrics['number_of_queries']}"
    )

    print(
        f"Latent dimensions:     "
        f"{result.metrics['latent_dimensions']}"
    )

    print()
    print(
        f"Top-1 class accuracy:  "
        f"{result.metrics['top_1_accuracy']:.4f}"
    )

    print(
        f"Mean reciprocal rank:  "
        f"{result.metrics['mean_reciprocal_rank']:.4f}"
    )

    for k_value in sorted(
        set(arguments.k_values)
    ):
        print(
            f"Precision@{k_value:<2d}:           "
            f"{result.metrics[f'precision_at_{k_value}']:.4f}"
        )

        print(
            f"Hit rate@{k_value:<2d}:            "
            f"{result.metrics[f'hit_rate_at_{k_value}']:.4f}"
        )

    print()
    print("Per-class retrieval")

    display_columns = [
        "class_code",
        "queries",
        "top_1_accuracy",
        "mean_reciprocal_rank",
    ]

    for k_value in sorted(
        set(arguments.k_values)
    ):
        display_columns.append(
            f"precision_at_{k_value}"
        )

    print(
        result.per_class_metrics[
            display_columns
        ].to_string(
            index=False,
        )
    )

    print()
    print(
        f"Results: "
        f"{arguments.output_dir.resolve()}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()