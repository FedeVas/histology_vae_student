from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.analysis.embedding_projection import (
    get_feature_columns,
)


"""
Unsupervised-кластеризация замороженных признаков (VAE latent_,
color_, dinov2_..._, segmentation_), с тем же train-only
дисциплиной, что и src/analysis/linear_probe.py: scaler и
clustering модель фиттятся только на train split, а метрики
качества считаются на отдельном (validation/external test) split.

Это отвечает на другой вопрос, чем linear probe: не "насколько
представление линейно разделимо для известных классов", а
"группируются ли объекты в естественные кластеры, совпадающие
с классами, без использования меток при обучении кластеризации".
Метки используются только после кластеризации, для оценки.
"""


@dataclass(frozen=True)
class ClusteringConfig:
    number_of_clusters: int
    algorithm: Literal[
        "kmeans", "gaussian_mixture"
    ] = "kmeans"
    seed: int = 42

    def validate(self) -> None:
        if self.number_of_clusters < 2:
            raise ValueError(
                "number_of_clusters must be at least 2."
            )

        if self.algorithm not in (
            "kmeans",
            "gaussian_mixture",
        ):
            raise ValueError(
                "algorithm must be 'kmeans' or "
                "'gaussian_mixture', received "
                f"{self.algorithm!r}."
            )


@dataclass(frozen=True)
class FittedClusteringModel:
    scaler: StandardScaler
    model: KMeans | GaussianMixture
    feature_prefix: str
    config: ClusteringConfig


@dataclass(frozen=True)
class ClusteringEvaluation:
    number_of_samples: int
    number_of_clusters: int

    # Доля объектов, попавших в кластер, где они образуют
    # большинство (0..1, выше -> лучше).
    cluster_purity: float

    # sklearn.metrics.adjusted_rand_score: скорректированное на
    # случайное совпадение согласие кластеров с true classes.
    adjusted_rand_index: float

    # sklearn.metrics.normalized_mutual_info_score.
    normalized_mutual_info: float

    cluster_to_class_table: pd.DataFrame

    def summary(self) -> dict[str, float | int]:
        return {
            "number_of_samples": (
                self.number_of_samples
            ),
            "number_of_clusters": (
                self.number_of_clusters
            ),
            "cluster_purity": (
                self.cluster_purity
            ),
            "adjusted_rand_index": (
                self.adjusted_rand_index
            ),
            "normalized_mutual_info": (
                self.normalized_mutual_info
            ),
        }


def fit_clustering_model(
    train_embeddings: pd.DataFrame,
    feature_prefix: str,
    config: ClusteringConfig,
) -> FittedClusteringModel:
    """
    Фиттит StandardScaler и модель кластеризации только на
    train_embeddings.
    """
    config.validate()

    if train_embeddings.empty:
        raise ValueError(
            "train_embeddings must not be empty."
        )

    feature_columns = get_feature_columns(
        embeddings=train_embeddings,
        feature_prefix=feature_prefix,
    )

    feature_values = train_embeddings[
        feature_columns
    ].to_numpy(dtype=np.float64)

    scaler = StandardScaler().fit(
        feature_values
    )

    standardized_features = scaler.transform(
        feature_values
    )

    if len(train_embeddings) <= config.number_of_clusters:
        raise ValueError(
            "number_of_clusters must be smaller than "
            "the number of training samples "
            f"({len(train_embeddings)})."
        )

    if config.algorithm == "kmeans":
        model: KMeans | GaussianMixture = KMeans(
            n_clusters=(
                config.number_of_clusters
            ),
            random_state=config.seed,
            n_init=10,
        ).fit(standardized_features)

    else:
        model = GaussianMixture(
            n_components=(
                config.number_of_clusters
            ),
            random_state=config.seed,
        ).fit(standardized_features)

    return FittedClusteringModel(
        scaler=scaler,
        model=model,
        feature_prefix=feature_prefix,
        config=config,
    )


def predict_clusters(
    fitted_model: FittedClusteringModel,
    embeddings: pd.DataFrame,
) -> np.ndarray:
    """
    Присваивает каждому объекту embeddings ближайший кластер,
    используя scaler и модель, обученные на train.
    """
    feature_columns = get_feature_columns(
        embeddings=embeddings,
        feature_prefix=(
            fitted_model.feature_prefix
        ),
    )

    feature_values = embeddings[
        feature_columns
    ].to_numpy(dtype=np.float64)

    standardized_features = (
        fitted_model.scaler.transform(
            feature_values
        )
    )

    return fitted_model.model.predict(
        standardized_features
    )


def evaluate_clustering(
    fitted_model: FittedClusteringModel,
    embeddings: pd.DataFrame,
    class_column: str = "class_code",
) -> ClusteringEvaluation:
    """
    Предсказывает кластеры для embeddings (обычно validation или
    external test split, НЕ train) и сравнивает их с истинными
    классами. Метки используются только здесь, не при фиттинге.
    """
    if class_column not in embeddings.columns:
        raise ValueError(
            f"Column {class_column!r} was not found "
            "in embeddings."
        )

    if embeddings.empty:
        raise ValueError(
            "embeddings must not be empty."
        )

    cluster_assignments = predict_clusters(
        fitted_model=fitted_model,
        embeddings=embeddings,
    )

    true_classes = (
        embeddings[class_column]
        .astype(str)
        .to_numpy()
    )

    crosstab = pd.crosstab(
        pd.Series(
            cluster_assignments,
            name="cluster",
        ),
        pd.Series(
            true_classes,
            name="true_class",
        ),
    )

    cluster_purity = float(
        crosstab.max(axis=1).sum()
        / crosstab.to_numpy().sum()
    )

    adjusted_rand_index = float(
        adjusted_rand_score(
            true_classes,
            cluster_assignments,
        )
    )

    normalized_mutual_info = float(
        normalized_mutual_info_score(
            true_classes,
            cluster_assignments,
        )
    )

    return ClusteringEvaluation(
        number_of_samples=len(embeddings),
        number_of_clusters=(
            fitted_model.config.number_of_clusters
        ),
        cluster_purity=cluster_purity,
        adjusted_rand_index=(
            adjusted_rand_index
        ),
        normalized_mutual_info=(
            normalized_mutual_info
        ),
        cluster_to_class_table=crosstab,
    )
