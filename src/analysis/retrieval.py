from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class RetrievalEvaluation:
    metrics: dict[str, Any]
    per_class_metrics: pd.DataFrame
    query_summary: pd.DataFrame
    neighbors: pd.DataFrame
    feature_columns: list[str]


def evaluate_nearest_neighbor_retrieval(
    train_embeddings: pd.DataFrame,
    query_embeddings: pd.DataFrame,
    k_values: list[int],
    metric: str = "euclidean",
) -> RetrievalEvaluation:
    """
    Ищет ближайшие train-патчи для каждого query-патча.

    Предполагаемый сценарий:
        train_embeddings -> NCT-CRC-HE-100K train
        query_embeddings -> CRC-VAL-HE-7K external test

    Метки не участвуют в поиске. Они используются только
    после retrieval для оценки совпадений классов.
    """
    normalized_k_values = _validate_k_values(
        k_values=k_values,
        number_of_train_samples=len(
            train_embeddings
        ),
    )

    feature_columns = _validate_embeddings(
        train_embeddings=train_embeddings,
        query_embeddings=query_embeddings,
    )

    x_train = (
        train_embeddings[feature_columns]
        .to_numpy(dtype=np.float64)
    )

    x_query = (
        query_embeddings[feature_columns]
        .to_numpy(dtype=np.float64)
    )

    # Масштабирование обучается только на train.
    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(
        x_train
    )

    x_query_scaled = scaler.transform(
        x_query
    )

    maximum_k = max(normalized_k_values)

    algorithm = (
        "brute"
        if metric == "cosine"
        else "auto"
    )

    neighbor_model = NearestNeighbors(
        n_neighbors=maximum_k,
        metric=metric,
        algorithm=algorithm,
    )

    neighbor_model.fit(x_train_scaled)

    distances, neighbor_indices = (
        neighbor_model.kneighbors(
            x_query_scaled,
            return_distance=True,
        )
    )

    neighbors = _build_neighbors_frame(
        train_embeddings=train_embeddings,
        query_embeddings=query_embeddings,
        distances=distances,
        neighbor_indices=neighbor_indices,
    )

    query_summary = _build_query_summary(
        neighbors=neighbors,
        k_values=normalized_k_values,
    )

    per_class_metrics = (
        _build_per_class_metrics(
            query_summary=query_summary,
            k_values=normalized_k_values,
        )
    )

    metrics: dict[str, Any] = {
        "metric": metric,
        "number_of_train_images": int(
            len(train_embeddings)
        ),
        "number_of_queries": int(
            len(query_embeddings)
        ),
        "latent_dimensions": int(
            len(feature_columns)
        ),
        "maximum_k": int(maximum_k),
        "top_1_accuracy": float(
            query_summary[
                "top_1_correct"
            ].mean()
        ),
        "mean_reciprocal_rank": float(
            query_summary[
                "reciprocal_rank"
            ].mean()
        ),
        "mean_top_1_distance": float(
            query_summary[
                "top_1_distance"
            ].mean()
        ),
    }

    for k_value in normalized_k_values:
        metrics[
            f"precision_at_{k_value}"
        ] = float(
            query_summary[
                f"precision_at_{k_value}"
            ].mean()
        )

        metrics[
            f"hit_rate_at_{k_value}"
        ] = float(
            query_summary[
                f"hit_at_{k_value}"
            ].mean()
        )

    return RetrievalEvaluation(
        metrics=metrics,
        per_class_metrics=per_class_metrics,
        query_summary=query_summary,
        neighbors=neighbors,
        feature_columns=feature_columns,
    )


def _build_neighbors_frame(
    train_embeddings: pd.DataFrame,
    query_embeddings: pd.DataFrame,
    distances: np.ndarray,
    neighbor_indices: np.ndarray,
) -> pd.DataFrame:
    train_embeddings = (
        train_embeddings
        .reset_index(drop=True)
    )

    query_embeddings = (
        query_embeddings
        .reset_index(drop=True)
    )

    records: list[dict[str, Any]] = []

    number_of_queries = (
        neighbor_indices.shape[0]
    )

    number_of_neighbors = (
        neighbor_indices.shape[1]
    )

    for query_index in range(
        number_of_queries
    ):
        query_row = query_embeddings.iloc[
            query_index
        ]

        for neighbor_offset in range(
            number_of_neighbors
        ):
            train_index = int(
                neighbor_indices[
                    query_index,
                    neighbor_offset,
                ]
            )

            train_row = train_embeddings.iloc[
                train_index
            ]

            query_label = int(
                query_row["label"]
            )

            neighbor_label = int(
                train_row["label"]
            )

            records.append(
                {
                    "query_index": query_index,
                    "query_sample_id": (
                        _get_string_value(
                            query_row,
                            "sample_id",
                        )
                    ),
                    "query_path": str(
                        query_row["path"]
                    ),
                    "query_source": (
                        _get_string_value(
                            query_row,
                            "source",
                        )
                    ),
                    "query_label": query_label,
                    "query_class_code": (
                        _get_class_code(
                            query_row
                        )
                    ),
                    "query_class_name": (
                        _get_string_value(
                            query_row,
                            "class_name",
                        )
                    ),
                    "neighbor_rank": (
                        neighbor_offset + 1
                    ),
                    "neighbor_train_index": (
                        train_index
                    ),
                    "neighbor_sample_id": (
                        _get_string_value(
                            train_row,
                            "sample_id",
                        )
                    ),
                    "neighbor_path": str(
                        train_row["path"]
                    ),
                    "neighbor_source": (
                        _get_string_value(
                            train_row,
                            "source",
                        )
                    ),
                    "neighbor_label": (
                        neighbor_label
                    ),
                    "neighbor_class_code": (
                        _get_class_code(
                            train_row
                        )
                    ),
                    "neighbor_class_name": (
                        _get_string_value(
                            train_row,
                            "class_name",
                        )
                    ),
                    "distance": float(
                        distances[
                            query_index,
                            neighbor_offset,
                        ]
                    ),
                    "same_class": (
                        query_label
                        == neighbor_label
                    ),
                }
            )

    return pd.DataFrame.from_records(
        records
    )


def _build_query_summary(
    neighbors: pd.DataFrame,
    k_values: list[int],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    grouped_queries = neighbors.groupby(
        "query_index",
        sort=True,
    )

    for query_index, query_neighbors in (
        grouped_queries
    ):
        query_neighbors = (
            query_neighbors
            .sort_values("neighbor_rank")
            .reset_index(drop=True)
        )

        first_row = query_neighbors.iloc[0]

        matching_neighbors = query_neighbors[
            query_neighbors["same_class"]
        ]

        if matching_neighbors.empty:
            first_matching_rank = None
            reciprocal_rank = 0.0

        else:
            first_matching_rank = int(
                matching_neighbors.iloc[0][
                    "neighbor_rank"
                ]
            )

            reciprocal_rank = (
                1.0 / first_matching_rank
            )

        record: dict[str, Any] = {
            "query_index": int(query_index),
            "query_sample_id": (
                first_row["query_sample_id"]
            ),
            "query_path": (
                first_row["query_path"]
            ),
            "query_source": (
                first_row["query_source"]
            ),
            "query_label": int(
                first_row["query_label"]
            ),
            "query_class_code": (
                first_row["query_class_code"]
            ),
            "query_class_name": (
                first_row["query_class_name"]
            ),
            "top_1_neighbor_label": int(
                first_row["neighbor_label"]
            ),
            "top_1_neighbor_class_code": (
                first_row[
                    "neighbor_class_code"
                ]
            ),
            "top_1_distance": float(
                first_row["distance"]
            ),
            "top_1_correct": bool(
                first_row["same_class"]
            ),
            "first_matching_rank": (
                first_matching_rank
            ),
            "reciprocal_rank": float(
                reciprocal_rank
            ),
        }

        for k_value in k_values:
            top_k = query_neighbors.iloc[
                :k_value
            ]

            matching_count = int(
                top_k["same_class"].sum()
            )

            record[
                f"precision_at_{k_value}"
            ] = (
                matching_count / k_value
            )

            record[
                f"hit_at_{k_value}"
            ] = bool(
                matching_count > 0
            )

        records.append(record)

    return pd.DataFrame.from_records(
        records
    )


def _build_per_class_metrics(
    query_summary: pd.DataFrame,
    k_values: list[int],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    grouped_classes = query_summary.groupby(
        [
            "query_label",
            "query_class_code",
            "query_class_name",
        ],
        dropna=False,
        sort=True,
    )

    for (
        class_key,
        class_frame,
    ) in grouped_classes:
        (
            label,
            class_code,
            class_name,
        ) = class_key

        record: dict[str, Any] = {
            "label": int(label),
            "class_code": str(
                class_code
            ),
            "class_name": str(
                class_name
            ),
            "queries": int(
                len(class_frame)
            ),
            "top_1_accuracy": float(
                class_frame[
                    "top_1_correct"
                ].mean()
            ),
            "mean_reciprocal_rank": float(
                class_frame[
                    "reciprocal_rank"
                ].mean()
            ),
            "mean_top_1_distance": float(
                class_frame[
                    "top_1_distance"
                ].mean()
            ),
        }

        for k_value in k_values:
            record[
                f"precision_at_{k_value}"
            ] = float(
                class_frame[
                    f"precision_at_{k_value}"
                ].mean()
            )

            record[
                f"hit_rate_at_{k_value}"
            ] = float(
                class_frame[
                    f"hit_at_{k_value}"
                ].mean()
            )

        records.append(record)

    return (
        pd.DataFrame.from_records(records)
        .sort_values("label")
        .reset_index(drop=True)
    )


def _validate_embeddings(
    train_embeddings: pd.DataFrame,
    query_embeddings: pd.DataFrame,
) -> list[str]:
    frames = {
        "train": train_embeddings,
        "query": query_embeddings,
    }

    train_feature_columns = [
        column
        for column in train_embeddings.columns
        if column.startswith("latent_")
    ]

    if not train_feature_columns:
        raise ValueError(
            "No latent feature columns were found."
        )

    for frame_name, frame in frames.items():
        if frame.empty:
            raise ValueError(
                f"{frame_name} embeddings are empty."
            )

        required_columns = {
            "path",
            "label",
        }

        missing_columns = (
            required_columns
            .difference(frame.columns)
        )

        if missing_columns:
            raise ValueError(
                f"{frame_name} embeddings are "
                "missing columns: "
                f"{sorted(missing_columns)}"
            )

        feature_columns = [
            column
            for column in frame.columns
            if column.startswith("latent_")
        ]

        if (
            feature_columns
            != train_feature_columns
        ):
            raise ValueError(
                "Train and query latent columns differ."
            )

        if frame[
            feature_columns
        ].isna().any().any():
            raise ValueError(
                f"{frame_name} embeddings "
                "contain NaN values."
            )

    train_labels = set(
        train_embeddings[
            "label"
        ].astype(int)
    )

    query_labels = set(
        query_embeddings[
            "label"
        ].astype(int)
    )

    unknown_query_labels = (
        query_labels - train_labels
    )

    if unknown_query_labels:
        raise ValueError(
            "Query contains labels absent from train: "
            f"{sorted(unknown_query_labels)}"
        )

    return train_feature_columns


def _validate_k_values(
    k_values: list[int],
    number_of_train_samples: int,
) -> list[int]:
    if not k_values:
        raise ValueError(
            "k_values must not be empty."
        )

    normalized_values = sorted(
        set(int(value) for value in k_values)
    )

    if normalized_values[0] <= 0:
        raise ValueError(
            "Every k value must be positive."
        )

    if normalized_values[-1] > (
        number_of_train_samples
    ):
        raise ValueError(
            "Maximum k exceeds the number "
            "of train samples."
        )

    return normalized_values


def _get_class_code(
    row: pd.Series,
) -> str:
    if (
        "class_code" in row.index
        and not pd.isna(
            row["class_code"]
        )
    ):
        return str(row["class_code"])

    return str(int(row["label"]))


def _get_string_value(
    row: pd.Series,
    column: str,
) -> str:
    if column not in row.index:
        return ""

    value = row[column]

    if pd.isna(value):
        return ""

    return str(value)