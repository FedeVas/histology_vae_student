from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def create_pca_embedding_plot(
    embeddings: pd.DataFrame,
    output_path: str | Path,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Стандартизирует latent features и строит PCA projection.
    """
    latent_columns = [
        column
        for column in embeddings.columns
        if column.startswith("latent_")
    ]

    if len(latent_columns) < 2:
        raise ValueError(
            "At least two latent dimensions are required."
        )

    if len(embeddings) < 2:
        raise ValueError(
            "At least two samples are required for PCA."
        )

    latent_values = embeddings[
        latent_columns
    ].to_numpy(dtype=np.float64)

    standardized_latent_values = (
        StandardScaler()
        .fit_transform(latent_values)
    )

    pca = PCA(
        n_components=2,
        random_state=seed,
    )

    coordinates = pca.fit_transform(
        standardized_latent_values
    )

    possible_metadata_columns = (
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

    metadata_columns = [
        column
        for column in possible_metadata_columns
        if column in embeddings.columns
    ]

    result = embeddings[
        metadata_columns
    ].copy()

    result["pca_1"] = coordinates[:, 0]
    result["pca_2"] = coordinates[:, 1]

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(8, 6)
    )

    if (
        "class_code" in result.columns
        and result["class_code"].ne("").any()
    ):
        group_column = "class_code"

        groups = sorted(
            result.loc[
                result["class_code"].ne(""),
                "class_code",
            ].unique()
        )

    else:
        group_column = "label"

        groups = sorted(
            result.loc[
                result["label"] >= 0,
                "label",
            ].unique()
        )

    if len(groups) > 1:
        for group_value in groups:
            group_mask = (
                result[group_column]
                == group_value
            )

            axis.scatter(
                result.loc[
                    group_mask,
                    "pca_1",
                ],
                result.loc[
                    group_mask,
                    "pca_2",
                ],
                label=str(group_value),
                alpha=0.70,
                s=20,
            )

        axis.legend(
            title="Tissue class",
            bbox_to_anchor=(
                1.02,
                1.0,
            ),
            loc="upper left",
        )

    else:
        axis.scatter(
            result["pca_1"],
            result["pca_2"],
            alpha=0.70,
            s=20,
        )

    explained_variance = (
        pca.explained_variance_ratio_
    )

    axis.set_title(
        "PCA of VAE latent means\n"
        f"PC1: {explained_variance[0]:.1%}, "
        f"PC2: {explained_variance[1]:.1%}"
    )

    axis.set_xlabel("Principal component 1")
    axis.set_ylabel("Principal component 2")
    axis.grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)

    return result, explained_variance