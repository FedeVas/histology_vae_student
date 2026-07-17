from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score


"""
Patient-level aggregation and bootstrap confidence intervals.

Мотивация (см. model_card.md -> "Known limitations" и
README.md -> "Roadmap"): внутренняя валидация и bootstrap-анализ в
docs/RESULTS.md выполняются на уровне patch, потому что публичный
train-архив (NCT-CRC-HE-100K) не даёт надёжного patch-to-patient
mapping. Это ограничение данных, а не кода: сам пайплайн
patient-aware split (src/datasets/split.py) уже существует и уже
используется для synthetic-датасета.

Этот модуль — недостающее звено между patch-level предсказаниями и
patient-level метриками: он существует и протестирован уже сейчас,
чтобы его можно было немедленно применить, как только появится
датасет с реальным patient_id (см. notebooks/05_patient_level_analysis.ipynb).
"""


def check_patient_leakage(
    metadata: pd.DataFrame,
    patient_column: str = "patient_id",
    split_column: str = "split",
) -> pd.DataFrame:
    """
    Проверяет, что ни один patient_id не встречается в
    нескольких split одновременно (например train и test).

    Такая утечка была бы серьёзной методологической ошибкой:
    внешние метрики выглядели бы лучше, чем реальная
    generalization на новых пациентах.

    Returns
    -------
    pd.DataFrame
        Одна строка на "протекающего" пациента с колонкой
        splits (список split, в которых он встречается).
        Пустой DataFrame означает отсутствие утечки.
    """
    required_columns = {
        patient_column,
        split_column,
    }

    missing_columns = required_columns.difference(
        metadata.columns
    )

    if missing_columns:
        raise ValueError(
            "metadata is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if metadata.empty:
        raise ValueError(
            "metadata must not be empty."
        )

    splits_per_patient = (
        metadata.groupby(patient_column)[
            split_column
        ].agg(lambda values: sorted(set(values)))
    )

    leaking_patients = splits_per_patient[
        splits_per_patient.map(len) > 1
    ]

    return (
        leaking_patients.rename("splits")
        .reset_index()
    )


def aggregate_predictions_by_patient(
    frame: pd.DataFrame,
    true_label_column: str,
    predicted_label_column: str,
    patient_column: str = "patient_id",
) -> pd.DataFrame:
    """
    Схлопывает patch-level предсказания до одной строки на пациента
    через majority vote.

    Каждый пациент должен иметь единственное истинное значение
    true_label_column: разные значения для одного пациента обычно
    означают ошибку в разметке или в patient_id, а не легитимную
    ситуацию для majority voting.
    """
    required_columns = {
        patient_column,
        true_label_column,
        predicted_label_column,
    }

    missing_columns = required_columns.difference(
        frame.columns
    )

    if missing_columns:
        raise ValueError(
            "frame is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if frame.empty:
        raise ValueError("frame must not be empty.")

    true_label_counts_per_patient = (
        frame.groupby(patient_column)[
            true_label_column
        ].nunique()
    )

    conflicting_patients = (
        true_label_counts_per_patient[
            true_label_counts_per_patient > 1
        ]
    )

    if not conflicting_patients.empty:
        raise ValueError(
            "Some patients have multiple true labels, "
            "which makes patient-level majority voting "
            "ill-defined: "
            f"{conflicting_patients.index.tolist()}"
        )

    records: list[dict[str, object]] = []

    for patient_id, patient_frame in frame.groupby(
        patient_column
    ):
        true_label = patient_frame[
            true_label_column
        ].iloc[0]

        vote_counts = patient_frame[
            predicted_label_column
        ].value_counts()

        majority_prediction = vote_counts.index[0]
        majority_votes = int(vote_counts.iloc[0])

        records.append(
            {
                patient_column: patient_id,
                true_label_column: true_label,
                predicted_label_column: (
                    majority_prediction
                ),
                "number_of_patches": len(
                    patient_frame
                ),
                "majority_vote_fraction": (
                    majority_votes
                    / len(patient_frame)
                ),
            }
        )

    return pd.DataFrame.from_records(records)


def calculate_patient_level_metrics(
    patient_frame: pd.DataFrame,
    true_label_column: str,
    predicted_label_column: str,
) -> dict[str, float]:
    """
    Balanced accuracy и macro-F1 на уже агрегированном
    (один ряд на пациента) DataFrame.
    """
    if patient_frame.empty:
        raise ValueError(
            "patient_frame must not be empty."
        )

    return {
        "balanced_accuracy": float(
            balanced_accuracy_score(
                patient_frame[true_label_column],
                patient_frame[
                    predicted_label_column
                ],
            )
        ),
        "macro_f1": float(
            f1_score(
                patient_frame[true_label_column],
                patient_frame[
                    predicted_label_column
                ],
                average="macro",
                zero_division=0,
            )
        ),
    }


def create_patient_bootstrap_sample(
    unique_patient_ids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Ресемплирует пациентов (не строки/patches) с возвращением.

    Это ключевое отличие от patch-level bootstrap в
    src/compare_probe_predictions.py::create_stratified_bootstrap_indices,
    который ресемплирует строки: там единицей ресемплинга является
    patch, здесь — пациент.
    """
    if len(unique_patient_ids) == 0:
        raise ValueError(
            "unique_patient_ids must not be empty."
        )

    return rng.choice(
        unique_patient_ids,
        size=len(unique_patient_ids),
        replace=True,
    )


def create_patient_cluster_bootstrap_indices(
    patch_patient_ids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Cluster bootstrap для patch-level метрик: ресемплирует
    уникальных пациентов с возвращением, затем включает ВСЕ patches
    каждого выбранного пациента (patches одного пациента, попавшего
    в выборку дважды, встречаются дважды).

    В отличие от create_patient_bootstrap_sample (которая
    ресемплирует уже агрегированный, один-ряд-на-пациента
    DataFrame), эта функция возвращает индексы строк в исходном,
    patch-level массиве — то есть метрика остаётся patch-level
    (без majority vote), но единицей ресемплинга всё равно является
    пациент.

    Это стандартный cluster bootstrap для коррелированных данных:
    корректная замена src.compare_probe_predictions
    .create_stratified_bootstrap_indices (которая ресемплирует
    отдельные patches и поэтому недооценивает неопределённость,
    когда patches одного пациента коррелированы).
    """
    if len(patch_patient_ids) == 0:
        raise ValueError(
            "patch_patient_ids must not be empty."
        )

    unique_patient_ids = np.unique(
        patch_patient_ids
    )

    sampled_patient_ids = rng.choice(
        unique_patient_ids,
        size=len(unique_patient_ids),
        replace=True,
    )

    indices_by_patient: dict[
        object, list[int]
    ] = {}

    for row_index, patient_id in enumerate(
        patch_patient_ids
    ):
        indices_by_patient.setdefault(
            patient_id, []
        ).append(row_index)

    sampled_indices: list[int] = []

    for patient_id in sampled_patient_ids:
        sampled_indices.extend(
            indices_by_patient[patient_id]
        )

    return np.array(
        sampled_indices, dtype=int
    )


def bootstrap_patch_level_metric_with_patient_clusters(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    patient_ids: np.ndarray,
    metric_fn=balanced_accuracy_score,
    number_of_iterations: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """
    Patch-level метрика (без patient-aggregation) с cluster
    bootstrap по пациентам.

    Полезна, когда хочется сохранить ту же метрику, что и в
    существующем patch-level анализе (docs/RESULTS.md), но
    получить честный (а не оптимистично узкий) доверительный
    интервал, учитывающий, что patches одного пациента не являются
    независимыми наблюдениями.
    """
    if not (
        len(true_labels)
        == len(predicted_labels)
        == len(patient_ids)
    ):
        raise ValueError(
            "true_labels, predicted_labels and "
            "patient_ids must have equal length."
        )

    if len(true_labels) == 0:
        raise ValueError(
            "Inputs must not be empty."
        )

    if number_of_iterations <= 0:
        raise ValueError(
            "number_of_iterations must be positive."
        )

    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(
        predicted_labels
    )
    patient_ids = np.asarray(patient_ids)

    rng = np.random.default_rng(seed)

    point_estimate = float(
        metric_fn(
            true_labels, predicted_labels
        )
    )

    bootstrap_samples = np.empty(
        number_of_iterations,
        dtype=np.float64,
    )

    for iteration in range(
        number_of_iterations
    ):
        sampled_indices = (
            create_patient_cluster_bootstrap_indices(
                patch_patient_ids=(
                    patient_ids
                ),
                rng=rng,
            )
        )

        bootstrap_samples[iteration] = (
            metric_fn(
                true_labels[sampled_indices],
                predicted_labels[
                    sampled_indices
                ],
            )
        )

    return {
        "point_estimate": point_estimate,
        "bootstrap_mean": float(
            bootstrap_samples.mean()
        ),
        "ci_2_5": float(
            np.quantile(
                bootstrap_samples, 0.025
            )
        ),
        "ci_97_5": float(
            np.quantile(
                bootstrap_samples, 0.975
            )
        ),
        "number_of_patients": int(
            len(np.unique(patient_ids))
        ),
        "number_of_patches": int(
            len(true_labels)
        ),
        "bootstrap_iterations": int(
            number_of_iterations
        ),
    }


def calculate_patient_level_bootstrap_ci(
    patient_frame: pd.DataFrame,
    true_label_column: str,
    predicted_label_column: str,
    patient_column: str = "patient_id",
    number_of_iterations: int = 2000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """
    Bootstrap confidence intervals для balanced accuracy и
    macro-F1, где единица ресемплинга — пациент.

    Ожидает patient_frame, где каждая строка — один пациент,
    например результат aggregate_predictions_by_patient.
    """
    required_columns = {
        patient_column,
        true_label_column,
        predicted_label_column,
    }

    missing_columns = required_columns.difference(
        patient_frame.columns
    )

    if missing_columns:
        raise ValueError(
            "patient_frame is missing required "
            f"columns: {sorted(missing_columns)}"
        )

    if patient_frame.empty:
        raise ValueError(
            "patient_frame must not be empty."
        )

    if number_of_iterations <= 0:
        raise ValueError(
            "number_of_iterations must be positive."
        )

    if patient_frame[patient_column].duplicated().any():
        raise ValueError(
            "patient_frame must contain exactly one "
            "row per patient. Aggregate patch-level "
            "predictions first, e.g. with "
            "aggregate_predictions_by_patient."
        )

    indexed_frame = patient_frame.set_index(
        patient_column
    )

    unique_patient_ids = patient_frame[
        patient_column
    ].to_numpy()

    rng = np.random.default_rng(seed)

    metric_names = (
        "balanced_accuracy",
        "macro_f1",
    )

    bootstrap_samples: dict[str, np.ndarray] = {
        metric_name: np.empty(
            number_of_iterations,
            dtype=np.float64,
        )
        for metric_name in metric_names
    }

    for iteration in range(number_of_iterations):
        sampled_patient_ids = (
            create_patient_bootstrap_sample(
                unique_patient_ids=(
                    unique_patient_ids
                ),
                rng=rng,
            )
        )

        sampled_frame = indexed_frame.loc[
            sampled_patient_ids
        ]

        sampled_metrics = (
            calculate_patient_level_metrics(
                patient_frame=sampled_frame,
                true_label_column=(
                    true_label_column
                ),
                predicted_label_column=(
                    predicted_label_column
                ),
            )
        )

        for metric_name in metric_names:
            bootstrap_samples[metric_name][
                iteration
            ] = sampled_metrics[metric_name]

    point_estimate = (
        calculate_patient_level_metrics(
            patient_frame=patient_frame,
            true_label_column=true_label_column,
            predicted_label_column=(
                predicted_label_column
            ),
        )
    )

    results: dict[str, dict[str, float]] = {}

    for metric_name in metric_names:
        samples = bootstrap_samples[metric_name]

        results[metric_name] = {
            "point_estimate": (
                point_estimate[metric_name]
            ),
            "bootstrap_mean": float(
                samples.mean()
            ),
            "ci_2_5": float(
                np.quantile(samples, 0.025)
            ),
            "ci_97_5": float(
                np.quantile(samples, 0.975)
            ),
            "number_of_patients": int(
                len(unique_patient_ids)
            ),
            "bootstrap_iterations": int(
                number_of_iterations
            ),
        }

    return results
