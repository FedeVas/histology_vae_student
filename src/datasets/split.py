from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


REQUIRED_METADATA_COLUMNS = {
    "path",
    "patient_id",
    "slide_id",
}


def validate_metadata(
    metadata: pd.DataFrame,
    require_split: bool = False,
) -> None:
    """
    Проверяет структуру metadata DataFrame.

    Обязательные колонки:
        path
        patient_id
        slide_id

    При require_split=True также требуется колонка split.
    """
    required_columns = set(REQUIRED_METADATA_COLUMNS)

    if require_split:
        required_columns.add("split")

    missing_columns = required_columns.difference(metadata.columns)

    if missing_columns:
        raise ValueError(
            "Metadata is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if metadata.empty:
        raise ValueError("Metadata must contain at least one row.")

    if metadata["path"].isna().any():
        raise ValueError("Metadata contains empty image paths.")

    if metadata["patient_id"].isna().any():
        raise ValueError("Metadata contains empty patient IDs.")

    if metadata["slide_id"].isna().any():
        raise ValueError("Metadata contains empty slide IDs.")

    if require_split:
        allowed_splits = {"train", "validation", "test"}
        actual_splits = set(metadata["split"].astype(str).unique())

        unknown_splits = actual_splits.difference(allowed_splits)

        if unknown_splits:
            raise ValueError(
                f"Unknown split names: {sorted(unknown_splits)}"
            )

        assert_no_patient_leakage(metadata)


def _validate_fractions(
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> np.ndarray:
    fractions = np.asarray(
        [
            train_fraction,
            validation_fraction,
            test_fraction,
        ],
        dtype=np.float64,
    )

    if np.any(fractions <= 0):
        raise ValueError("All split fractions must be greater than zero.")

    if not np.isclose(fractions.sum(), 1.0):
        raise ValueError(
            "Train, validation and test fractions must sum to 1.0. "
            f"Received: {fractions.sum():.6f}"
        )

    return fractions


def _calculate_split_counts(
    number_of_patients: int,
    fractions: np.ndarray,
) -> np.ndarray:
    """
    Вычисляет число пациентов в каждом split.

    Используется метод largest remainder:
    сначала берётся целая часть, затем остатки распределяются
    по наибольшим дробным частям.
    """
    if number_of_patients < 3:
        raise ValueError(
            "At least three patients are required to create "
            "train, validation and test splits."
        )

    raw_counts = fractions * number_of_patients
    split_counts = np.floor(raw_counts).astype(int)

    remaining_patients = number_of_patients - int(split_counts.sum())

    fractional_parts = raw_counts - split_counts
    priority_indices = np.argsort(fractional_parts)[::-1]

    for index in priority_indices[:remaining_patients]:
        split_counts[index] += 1

    # Для достаточно большой выборки каждый split должен получить пациента.
    for empty_index in np.flatnonzero(split_counts == 0):
        donor_index = int(np.argmax(split_counts))

        if split_counts[donor_index] <= 1:
            raise ValueError(
                "Not enough patients to create three non-empty splits."
            )

        split_counts[donor_index] -= 1
        split_counts[empty_index] += 1

    if split_counts.sum() != number_of_patients:
        raise RuntimeError("Split count calculation failed.")

    return split_counts


def assign_patient_splits(
    metadata: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Назначает train/validation/test split на уровне пациентов.

    Все патчи одного пациента попадают только в один split.
    """
    validate_metadata(metadata, require_split=False)

    fractions = _validate_fractions(
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )

    patient_ids = np.asarray(
        sorted(metadata["patient_id"].astype(str).unique())
    )

    rng = np.random.default_rng(seed)
    rng.shuffle(patient_ids)

    split_counts = _calculate_split_counts(
        number_of_patients=len(patient_ids),
        fractions=fractions,
    )

    number_of_train_patients = int(split_counts[0])
    number_of_validation_patients = int(split_counts[1])

    train_end = number_of_train_patients
    validation_end = train_end + number_of_validation_patients

    train_patients = patient_ids[:train_end]
    validation_patients = patient_ids[train_end:validation_end]
    test_patients = patient_ids[validation_end:]

    split_map: dict[str, str] = {}

    split_map.update(
        _create_split_map(train_patients, split_name="train")
    )
    split_map.update(
        _create_split_map(
            validation_patients,
            split_name="validation",
        )
    )
    split_map.update(
        _create_split_map(test_patients, split_name="test")
    )

    result = metadata.copy()

    result["patient_id"] = result["patient_id"].astype(str)
    result["slide_id"] = result["slide_id"].astype(str)

    result["split"] = result["patient_id"].map(split_map)

    if result["split"].isna().any():
        raise RuntimeError("Some patients were not assigned to a split.")

    validate_metadata(result, require_split=True)

    return result


def _create_split_map(
    patient_ids: Iterable[str],
    split_name: str,
) -> dict[str, str]:
    return {
        str(patient_id): split_name
        for patient_id in patient_ids
    }


def assert_no_patient_leakage(metadata: pd.DataFrame) -> None:
    """
    Проверяет, что пациент не встречается сразу в нескольких splits.
    """
    if "split" not in metadata.columns:
        raise ValueError(
            "Metadata must contain a split column before leakage check."
        )

    split_counts_per_patient = (
        metadata.groupby("patient_id")["split"]
        .nunique()
    )

    leaking_patients = split_counts_per_patient[
        split_counts_per_patient > 1
    ]

    if not leaking_patients.empty:
        raise ValueError(
            "Patient leakage detected. The following patients appear "
            f"in multiple splits: {leaking_patients.index.tolist()}"
        )


def get_split_summary(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Возвращает статистику по split:
    число пациентов, слайдов и патчей.
    """
    validate_metadata(metadata, require_split=True)

    summary = (
        metadata.groupby("split")
        .agg(
            patients=("patient_id", "nunique"),
            slides=("slide_id", "nunique"),
            patches=("path", "count"),
        )
        .reindex(["train", "validation", "test"])
        .reset_index()
    )

    return summary