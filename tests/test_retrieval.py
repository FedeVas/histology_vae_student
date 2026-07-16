import numpy as np
import pandas as pd
import pytest

from src.analysis.retrieval import (
    evaluate_nearest_neighbor_retrieval,
)


def create_retrieval_embeddings(
    samples_per_class: int,
    seed: int,
    prefix: str,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    records: list[dict[str, object]] = []

    classes = [
        "ADI",
        "LYM",
        "TUM",
    ]

    for label, class_code in enumerate(
        classes
    ):
        center = np.zeros(
            5,
            dtype=np.float64,
        )

        center[label] = 8.0

        features = rng.normal(
            loc=center,
            scale=0.25,
            size=(
                samples_per_class,
                5,
            ),
        )

        for sample_index, feature in enumerate(
            features
        ):
            records.append(
                {
                    "sample_id": (
                        f"{prefix}_{label}_"
                        f"{sample_index}"
                    ),
                    "path": (
                        f"unused/{prefix}_{label}_"
                        f"{sample_index}.png"
                    ),
                    "source": prefix,
                    "class_code": class_code,
                    "class_name": class_code,
                    "label": label,
                    "latent_000": feature[0],
                    "latent_001": feature[1],
                    "latent_002": feature[2],
                    "latent_003": feature[3],
                    "latent_004": feature[4],
                }
            )

    return pd.DataFrame(records)


def test_retrieval_finds_same_class_neighbors() -> None:
    train = create_retrieval_embeddings(
        samples_per_class=30,
        seed=1,
        prefix="train",
    )

    query = create_retrieval_embeddings(
        samples_per_class=10,
        seed=2,
        prefix="query",
    )

    result = (
        evaluate_nearest_neighbor_retrieval(
            train_embeddings=train,
            query_embeddings=query,
            k_values=[
                1,
                3,
                5,
            ],
        )
    )

    assert (
        result.metrics[
            "top_1_accuracy"
        ]
        > 0.95
    )

    assert (
        result.metrics[
            "precision_at_5"
        ]
        > 0.95
    )

    assert len(result.query_summary) == len(
        query
    )

    assert len(result.neighbors) == (
        len(query) * 5
    )


def test_retrieval_rejects_different_latent_columns() -> None:
    train = create_retrieval_embeddings(
        samples_per_class=5,
        seed=1,
        prefix="train",
    )

    query = create_retrieval_embeddings(
        samples_per_class=5,
        seed=2,
        prefix="query",
    )

    query = query.drop(
        columns="latent_004"
    )

    with pytest.raises(ValueError):
        evaluate_nearest_neighbor_retrieval(
            train_embeddings=train,
            query_embeddings=query,
            k_values=[1],
        )


def test_retrieval_rejects_excessive_k() -> None:
    train = create_retrieval_embeddings(
        samples_per_class=2,
        seed=1,
        prefix="train",
    )

    query = create_retrieval_embeddings(
        samples_per_class=2,
        seed=2,
        prefix="query",
    )

    with pytest.raises(ValueError):
        evaluate_nearest_neighbor_retrieval(
            train_embeddings=train,
            query_embeddings=query,
            k_values=[
                100,
            ],
        )