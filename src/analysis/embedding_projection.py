from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


"""
Обобщённая 2D-проекция признаков (PCA или UMAP) для сравнения
разных представлений (VAE latent_, color_, dinov2_..._,
segmentation_) на одних и тех же графиках.

В отличие от src/analysis/plots.py::create_pca_embedding_plot
(который жёстко привязан к префиксу "latent_" и сразу сохраняет
PNG на диск), этот модуль:

    - принимает произвольный feature_prefix, как и остальной
      analysis-код проекта (linear_probe.py, nuclei_segmentation.py);
    - поддерживает PCA и UMAP;
    - возвращает координаты и Axes, а не только файл на диске,
      чтобы им было удобно пользоваться из notebooks (несколько
      панелей сравнения моделей на одной figure).
"""


_METADATA_COLUMNS = (
    "sample_id",
    "path",
    "patient_id",
    "slide_id",
    "patch_id",
    "source",
    "class_code",
    "class_name",
    "label",
    "split",
)


@dataclass(frozen=True)
class ProjectionResult:
    method: Literal["pca", "umap"]
    feature_prefix: str
    number_of_input_dimensions: int

    coordinates: pd.DataFrame

    # Только для PCA. None для UMAP (UMAP не даёт
    # интерпретируемой explained variance по компонентам).
    explained_variance_ratio: (
        tuple[float, float] | None
    )


def get_feature_columns(
    embeddings: pd.DataFrame,
    feature_prefix: str,
) -> list[str]:
    columns = [
        column
        for column in embeddings.columns
        if column.startswith(feature_prefix)
    ]

    if len(columns) < 2:
        raise ValueError(
            "At least two feature columns with prefix "
            f"{feature_prefix!r} are required, found "
            f"{len(columns)}."
        )

    return columns


def compute_2d_projection(
    embeddings: pd.DataFrame,
    feature_prefix: str,
    method: Literal["pca", "umap"] = "pca",
    seed: int = 42,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.03,
) -> ProjectionResult:
    """
    Стандартизирует признаки с заданным префиксом и проецирует
    их в 2D через PCA или UMAP.

    Стандартизация (StandardScaler) обязательна: без неё
    признаки с большим масштабом (например color-статистики)
    доминировали бы над остальными.
    """
    if method not in ("pca", "umap"):
        raise ValueError(
            "method must be 'pca' or 'umap', received "
            f"{method!r}."
        )

    if len(embeddings) < 2:
        raise ValueError(
            "At least two samples are required for a "
            "2D projection."
        )

    feature_columns = get_feature_columns(
        embeddings=embeddings,
        feature_prefix=feature_prefix,
    )

    feature_values = embeddings[
        feature_columns
    ].to_numpy(dtype=np.float64)

    standardized_features = (
        StandardScaler().fit_transform(
            feature_values
        )
    )

    explained_variance_ratio: (
        tuple[float, float] | None
    ) = None

    if method == "pca":
        reducer = PCA(
            n_components=2,
            random_state=seed,
        )

        coordinates = reducer.fit_transform(
            standardized_features
        )

        explained_variance_ratio = (
            float(
                reducer
                .explained_variance_ratio_[0]
            ),
            float(
                reducer
                .explained_variance_ratio_[1]
            ),
        )

    else:
        try:
            import umap
        except ImportError as error:
            raise ImportError(
                "The 'umap-learn' package is required "
                "for method='umap'. It is already listed "
                "in requirements.txt; install with: "
                "pip install umap-learn"
            ) from error

        # UMAP требует n_neighbors < число samples.
        safe_n_neighbors = min(
            umap_n_neighbors,
            len(embeddings) - 1,
        )

        reducer = umap.UMAP(
            n_components=2,
            random_state=seed,
            n_neighbors=safe_n_neighbors,
            min_dist=umap_min_dist,
        )

        coordinates = reducer.fit_transform(
            standardized_features
        )

    metadata_columns = [
        column
        for column in _METADATA_COLUMNS
        if column in embeddings.columns
    ]

    result_frame = embeddings[
        metadata_columns
    ].reset_index(drop=True).copy()

    result_frame["proj_1"] = coordinates[:, 0]
    result_frame["proj_2"] = coordinates[:, 1]

    return ProjectionResult(
        method=method,
        feature_prefix=feature_prefix,
        number_of_input_dimensions=len(
            feature_columns
        ),
        coordinates=result_frame,
        explained_variance_ratio=(
            explained_variance_ratio
        ),
    )


def plot_projection(
    result: ProjectionResult,
    ax: plt.Axes | None = None,
    title: str | None = None,
    legend: bool = True,
) -> plt.Axes:
    """
    Рисует scatter plot проекции, раскрашенный по классу ткани.

    Если ax не передан, создаёт новую figure/axes (для
    самостоятельного использования). Если передан — рисует поверх
    него (для панельных сравнений нескольких моделей на одной
    figure в notebooks).
    """
    coordinates = result.coordinates

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    if (
        "class_code" in coordinates.columns
        and coordinates["class_code"].ne("").any()
    ):
        group_column = "class_code"

        groups = sorted(
            coordinates.loc[
                coordinates["class_code"].ne(""),
                "class_code",
            ].unique()
        )

    elif "label" in coordinates.columns:
        group_column = "label"

        groups = sorted(
            coordinates["label"].unique()
        )

    else:
        group_column = None
        groups = []

    if group_column is not None and len(groups) > 1:
        for group_value in groups:
            group_mask = (
                coordinates[group_column]
                == group_value
            )

            ax.scatter(
                coordinates.loc[
                    group_mask, "proj_1"
                ],
                coordinates.loc[
                    group_mask, "proj_2"
                ],
                label=str(group_value),
                alpha=0.70,
                s=18,
            )

        if legend:
            ax.legend(
                title="Tissue class",
                fontsize=8,
                title_fontsize=9,
                markerscale=1.2,
            )

    else:
        ax.scatter(
            coordinates["proj_1"],
            coordinates["proj_2"],
            alpha=0.70,
            s=18,
        )

    if title is None:
        if (
            result.method == "pca"
            and result.explained_variance_ratio
            is not None
        ):
            variance_1, variance_2 = (
                result.explained_variance_ratio
            )

            title = (
                f"{result.feature_prefix} (PCA)\n"
                f"PC1: {variance_1:.1%}, "
                f"PC2: {variance_2:.1%}"
            )
        else:
            title = (
                f"{result.feature_prefix} "
                f"({result.method.upper()})"
            )

    ax.set_title(title, fontsize=10)
    ax.set_xlabel(f"{result.method}_1")
    ax.set_ylabel(f"{result.method}_2")
    ax.grid(alpha=0.2)

    return ax


def save_projection_plot(
    result: ProjectionResult,
    output_path: str | Path,
    title: str | None = None,
) -> None:
    figure, ax = plt.subplots(figsize=(7, 6))

    plot_projection(
        result=result,
        ax=ax,
        title=title,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)
