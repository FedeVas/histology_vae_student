import numpy as np
import pandas as pd

from src.analysis.linear_probe import (
    fit_linear_probe,
)


def create_separable_embeddings(
    samples_per_class: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    records: list[dict[str, object]] = []

    class_codes = [
        "ADI",
        "LYM",
        "TUM",
    ]

    for label, class_code in enumerate(
        class_codes
    ):
        center = np.zeros(
            4,
            dtype=np.float64,
        )

        center[label] = 6.0

        features = rng.normal(
            loc=center,
            scale=0.35,
            size=(
                samples_per_class,
                4,
            ),
        )

        for sample_index, feature in enumerate(
            features
        ):
            records.append(
                {
                    "sample_id": (
                        f"{seed}_{label}_"
                        f"{sample_index}"
                    ),
                    "path": (
                        f"unused/{seed}_{label}_"
                        f"{sample_index}.png"
                    ),
                    "class_code": class_code,
                    "class_name": class_code,
                    "label": label,
                    "latent_000": feature[0],
                    "latent_001": feature[1],
                    "latent_002": feature[2],
                    "latent_003": feature[3],
                }
            )

    return pd.DataFrame(records)


def test_linear_probe_classifies_separable_embeddings() -> None:
    train = create_separable_embeddings(
        samples_per_class=30,
        seed=1,
    )

    validation = create_separable_embeddings(
        samples_per_class=10,
        seed=2,
    )

    test = create_separable_embeddings(
        samples_per_class=10,
        seed=3,
    )

    result = fit_linear_probe(
        train_embeddings=train,
        validation_embeddings=validation,
        test_embeddings=test,
        c_values=[
            0.01,
            0.1,
            1.0,
        ],
        seed=42,
    )

    assert (
        result.validation.metrics[
            "balanced_accuracy"
        ]
        > 0.95
    )

    assert (
        result.test.metrics[
            "balanced_accuracy"
        ]
        > 0.95
    )

    assert len(
        result.test.predictions
    ) == len(test)

    assert result.best_c in {
        0.01,
        0.1,
        1.0,
    }
    

def test_linear_probe_supports_custom_feature_prefix() -> None:
    train = create_separable_embeddings(
        samples_per_class=30,
        seed=11,
    ).rename(
        columns=lambda column: (
            column.replace(
                "latent_",
                "color_",
            )
            if column.startswith("latent_")
            else column
        )
    )

    validation = create_separable_embeddings(
        samples_per_class=10,
        seed=12,
    ).rename(
        columns=lambda column: (
            column.replace(
                "latent_",
                "color_",
            )
            if column.startswith("latent_")
            else column
        )
    )

    test = create_separable_embeddings(
        samples_per_class=10,
        seed=13,
    ).rename(
        columns=lambda column: (
            column.replace(
                "latent_",
                "color_",
            )
            if column.startswith("latent_")
            else column
        )
    )

    result = fit_linear_probe(
        train_embeddings=train,
        validation_embeddings=validation,
        test_embeddings=test,
        c_values=[
            0.01,
            0.1,
            1.0,
        ],
        seed=42,
        feature_prefix="color_",
    )

    assert result.feature_prefix == "color_"

    assert all(
        column.startswith("color_")
        for column in result.feature_columns
    )

    assert (
        result.test.metrics[
            "balanced_accuracy"
        ]
        > 0.95
    )
  
  
def test_linear_probe_supports_pca_reduction() -> None:
    train = create_separable_embeddings(
        samples_per_class=30,
        seed=21,
    )

    validation = create_separable_embeddings(
        samples_per_class=10,
        seed=22,
    )

    test = create_separable_embeddings(
        samples_per_class=10,
        seed=23,
    )

    result = fit_linear_probe(
        train_embeddings=train,
        validation_embeddings=validation,
        test_embeddings=test,
        c_values=[
            0.01,
            0.1,
            1.0,
        ],
        seed=42,
        feature_prefix="latent_",
        pca_components=3,
    )

    assert result.pca_components == 3

    assert "pca" in (
        result.final_model.named_steps
    )

    assert (
        result.test.metrics[
            "balanced_accuracy"
        ]
        > 0.95
    )
