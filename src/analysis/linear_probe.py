from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ProbeEvaluation:
    metrics: dict[str, float]
    predictions: pd.DataFrame
    classification_report: pd.DataFrame
    confusion_matrix: np.ndarray
    normalized_confusion_matrix: np.ndarray


@dataclass
class LinearProbeResult:
    best_c: float
    selection_results: pd.DataFrame

    validation: ProbeEvaluation
    test_before_refit: ProbeEvaluation
    test: ProbeEvaluation

    final_model: Pipeline
    feature_columns: list[str]
    feature_prefix: strd
    pca_components: int | None
    class_table: pd.DataFrame


def fit_linear_probe(
    train_embeddings: pd.DataFrame,
    validation_embeddings: pd.DataFrame,
    test_embeddings: pd.DataFrame,
    c_values: list[float],
    seed: int = 42,
    feature_prefix: str = "latent_",
    pca_components: int | None = None,
) -> LinearProbeResult:
    """
    Выбирает regularization C на validation и оценивает
    итоговую модель на external test.

    VAE и его embeddings при этом не изменяются.
    """
    if not feature_prefix:
        raise ValueError(
            "feature_prefix must be a non-empty string."
        )
    if not c_values:
        raise ValueError(
            "c_values must contain at least one value."
        )

    if any(value <= 0 for value in c_values):
        raise ValueError(
            "Every C value must be positive."
        )

    feature_columns = _validate_embeddings(
        train_embeddings=train_embeddings,
        validation_embeddings=(
            validation_embeddings
        ),
        test_embeddings=test_embeddings,
        feature_prefix=feature_prefix
    )
    
    if pca_components is not None:
        if pca_components <= 0:
            raise ValueError(
                "pca_components must be a positive integer or None."
            )
        
        maximum_components = min(
            len(train_embeddings),
            len(feature_columns)
        )
        
        if pca_components > maximum_components:
            raise ValueError(
                "pca_components exceeds the maximum "
                "supported dimensionality. "
                f"Requested {pca_components}, "
                f"maximum {maximum_components}."
            )

    class_table = _build_class_table(
        [
            train_embeddings,
            validation_embeddings,
            test_embeddings,
        ]
    )

    labels = (
        class_table["label"]
        .astype(int)
        .tolist()
    )

    x_train = train_embeddings[
        feature_columns
    ].to_numpy(dtype=np.float64)

    y_train = (
        train_embeddings["label"]
        .to_numpy(dtype=np.int64)
    )

    x_validation = validation_embeddings[
        feature_columns
    ].to_numpy(dtype=np.float64)

    y_validation = (
        validation_embeddings["label"]
        .to_numpy(dtype=np.int64)
    )

    x_test = test_embeddings[
        feature_columns
    ].to_numpy(dtype=np.float64)

    y_test = (
        test_embeddings["label"]
        .to_numpy(dtype=np.int64)
    )

    selection_rows: list[dict[str, float]] = []

    for c_value in c_values:
        candidate_model = _build_probe_model(
            c_value=float(c_value),
            seed=seed,
            pca_components=pca_components
        )

        candidate_model.fit(
            x_train,
            y_train,
        )

        validation_predictions = (
            candidate_model.predict(
                x_validation
            )
        )

        validation_metrics = (
            _calculate_basic_metrics(
                y_true=y_validation,
                y_pred=validation_predictions,
            )
        )

        selection_rows.append(
            {
                "c": float(c_value),
                "validation_accuracy": (
                    validation_metrics[
                        "accuracy"
                    ]
                ),
                "validation_balanced_accuracy": (
                    validation_metrics[
                        "balanced_accuracy"
                    ]
                ),
                "validation_macro_f1": (
                    validation_metrics[
                        "macro_f1"
                    ]
                ),
                "validation_weighted_f1": (
                    validation_metrics[
                        "weighted_f1"
                    ]
                ),
            }
        )

    selection_results = pd.DataFrame(
        selection_rows
    )

    selection_results = (
        selection_results
        .sort_values(
            by=[
                "validation_balanced_accuracy",
                "validation_macro_f1",
                "c",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    best_c = float(
        selection_results.iloc[0]["c"]
    )

    # Эта модель обучена только на train и используется
    # для честной validation-оценки.
    validation_model = _build_probe_model(
        c_value=best_c,
        seed=seed,
        pca_components=pca_components
    )

    validation_model.fit(
        x_train,
        y_train,
    )

    validation_evaluation = _evaluate_model(
        model=validation_model,
        embeddings=validation_embeddings,
        x=x_validation,
        y_true=y_validation,
        labels=labels,
        class_table=class_table,
        feature_columns=feature_columns
    )
    
    test_before_refit_evaluation = _evaluate_model(
        model=validation_model,
        embeddings=test_embeddings,
        x=x_test,
        y_true=y_test,
        labels=labels,
        class_table=class_table,
        feature_columns=feature_columns
    )

    # После выбора C можно использовать всю внутреннюю выборку.
    x_train_final = np.concatenate(
        [
            x_train,
            x_validation,
        ],
        axis=0,
    )

    y_train_final = np.concatenate(
        [
            y_train,
            y_validation,
        ],
        axis=0,
    )

    final_model = _build_probe_model(
        c_value=best_c,
        seed=seed,
        pca_components=pca_components
    )

    final_model.fit(
        x_train_final,
        y_train_final,
    )

    test_evaluation = _evaluate_model(
        model=final_model,
        embeddings=test_embeddings,
        x=x_test,
        y_true=y_test,
        labels=labels,
        class_table=class_table,
        feature_columns=feature_columns,
    )

    return LinearProbeResult(
        best_c=best_c,
        selection_results=selection_results,
        validation=validation_evaluation,
        test_before_refit=test_before_refit_evaluation,
        test=test_evaluation,
        final_model=final_model,
        feature_columns=feature_columns,
        feature_prefix=feature_prefix,
        pca_components=pca_components,
        class_table=class_table,
    )


def _build_probe_model(
    c_value: float,
    seed: int,
    pca_components: int | None = None,
) -> Pipeline:
    steps: list[tuple[str, object]] = [
        (
            "scaler",
            StandardScaler(),
        ),
    ]

    if pca_components is not None:
        steps.append(
            (
                "pca",
                PCA(
                    n_components=pca_components,
                    random_state=seed,
                ),
            )
        )

    steps.append(
        (
            "classifier",
            LogisticRegression(
                C=c_value,
                solver="lbfgs",
                max_iter=5000,
                random_state=seed,
            ),
        )
    )

    return Pipeline(steps=steps)


def _evaluate_model(
    model: Pipeline,
    embeddings: pd.DataFrame,
    x: np.ndarray,
    y_true: np.ndarray,
    labels: list[int],
    class_table: pd.DataFrame,
    feature_columns: list[str],
) -> ProbeEvaluation:
    y_pred = model.predict(x)
    probabilities = model.predict_proba(x)

    classifier = model.named_steps[
        "classifier"
    ]

    probability_labels = [
        int(value)
        for value in classifier.classes_
    ]

    if probability_labels != labels:
        raise RuntimeError(
            "Classifier class order does not match "
            "the expected label order."
        )

    metrics = _calculate_basic_metrics(
        y_true=y_true,
        y_pred=y_pred,
    )

    metrics["log_loss"] = float(
        log_loss(
            y_true,
            probabilities,
            labels=labels,
        )
    )

    class_codes = (
        class_table["class_code"]
        .astype(str)
        .tolist()
    )

    report = classification_report(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        target_names=class_codes,
        output_dict=True,
        zero_division=0,
    )

    report_frame = (
        pd.DataFrame(report)
        .transpose()
        .reset_index()
        .rename(
            columns={
                "index": "class_or_average"
            }
        )
    )

    raw_confusion_matrix = confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
    )

    normalized_confusion_matrix = (
        confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels,
            normalize="true",
        )
    )

    predictions = _build_prediction_frame(
        embeddings=embeddings,
        y_pred=y_pred,
        probabilities=probabilities,
        labels=labels,
        class_table=class_table,
        feature_columns=feature_columns
    )

    return ProbeEvaluation(
        metrics=metrics,
        predictions=predictions,
        classification_report=report_frame,
        confusion_matrix=raw_confusion_matrix,
        normalized_confusion_matrix=(
            normalized_confusion_matrix
        ),
    )


def _calculate_basic_metrics(
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
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def _build_prediction_frame(
    embeddings: pd.DataFrame,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    labels: list[int],
    class_table: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    feature_column_set = set(feature_columns)
    metadata_columns = [
        column
        for column in embeddings.columns
        if column not in feature_column_set
    ]

    predictions = (
        embeddings[metadata_columns]
        .copy()
        .reset_index(drop=True)
    )

    predictions["predicted_label"] = y_pred

    label_to_code = dict(
        zip(
            class_table["label"],
            class_table["class_code"],
        )
    )

    label_to_name = dict(
        zip(
            class_table["label"],
            class_table["class_name"],
        )
    )

    predictions["predicted_class_code"] = [
        label_to_code[int(label)]
        for label in y_pred
    ]

    predictions["predicted_class_name"] = [
        label_to_name[int(label)]
        for label in y_pred
    ]

    predictions["correct"] = (
        predictions["label"].astype(int)
        == predictions["predicted_label"]
    )

    for probability_index, label in enumerate(
        labels
    ):
        class_code = str(
            label_to_code[label]
        )

        predictions[
            f"probability_{class_code}"
        ] = probabilities[
            :,
            probability_index,
        ]

    return predictions


def _validate_embeddings(
    train_embeddings: pd.DataFrame,
    validation_embeddings: pd.DataFrame,
    test_embeddings: pd.DataFrame,
    feature_prefix: str
) -> list[str]:
    frames = {
        "train": train_embeddings,
        "validation": validation_embeddings,
        "test": test_embeddings,
    }

    train_feature_columns = [
        column
        for column in train_embeddings.columns
        if column.startswith(feature_prefix)
    ]

    if not train_feature_columns:
        raise ValueError(
            f"No feature columns found for prefix {feature_prefix!r}."
        )

    for split_name, frame in frames.items():
        if frame.empty:
            raise ValueError(
                f"{split_name} embeddings are empty."
            )

        if "label" not in frame.columns:
            raise ValueError(
                f"{split_name} embeddings do not "
                "contain label."
            )

        current_feature_columns = [
            column
            for column in frame.columns
            if column.startswith(feature_prefix)
        ]

        if (
            current_feature_columns
            != train_feature_columns
        ):
            raise ValueError(
                "Feature columns differ between splits "
                f"for prefix {feature_prefix!r}. "
                f"Problematic split: {split_name!r}."
            )

        if frame[
            train_feature_columns
        ].isna().any().any():
            raise ValueError(
                f"{split_name} embeddings contain NaN."
            )

    train_labels = set(
        train_embeddings[
            "label"
        ].astype(int)
    )

    for split_name, frame in frames.items():
        current_labels = set(
            frame["label"].astype(int)
        )

        if current_labels != train_labels:
            raise ValueError(
                "Every split must contain the same "
                f"classes. Problematic split: {split_name}."
            )

    return train_feature_columns


def _build_class_table(
    frames: list[pd.DataFrame],
) -> pd.DataFrame:
    class_records: list[pd.DataFrame] = []

    for frame in frames:
        current = pd.DataFrame(
            {
                "label": frame[
                    "label"
                ].astype(int),
            }
        )

        if "class_code" in frame.columns:
            current["class_code"] = (
                frame["class_code"].astype(str)
            )
        else:
            current["class_code"] = (
                current["label"].astype(str)
            )

        if "class_name" in frame.columns:
            current["class_name"] = (
                frame["class_name"].astype(str)
            )
        else:
            current["class_name"] = (
                current["class_code"]
            )

        class_records.append(current)

    class_table = (
        pd.concat(
            class_records,
            ignore_index=True,
        )
        .drop_duplicates()
        .sort_values("label")
        .reset_index(drop=True)
    )

    mapping_counts = (
        class_table.groupby("label")
        .agg(
            class_codes=(
                "class_code",
                "nunique",
            ),
            class_names=(
                "class_name",
                "nunique",
            ),
        )
    )

    if (
        mapping_counts["class_codes"] > 1
    ).any():
        raise ValueError(
            "One label maps to multiple class codes."
        )

    if (
        mapping_counts["class_names"] > 1
    ).any():
        raise ValueError(
            "One label maps to multiple class names."
        )

    return (
        class_table
        .drop_duplicates(subset=["label"])
        .reset_index(drop=True)
    )