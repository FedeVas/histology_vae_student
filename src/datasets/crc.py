from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CRCTissueClass:
    code: str
    label: int
    name: str


CRC_TISSUE_CLASSES: tuple[CRCTissueClass, ...] = (
    CRCTissueClass("ADI", 0, "adipose tissue"),
    CRCTissueClass("BACK", 1, "background"),
    CRCTissueClass("DEB", 2, "debris"),
    CRCTissueClass("LYM", 3, "lymphocytes"),
    CRCTissueClass("MUC", 4, "mucus"),
    CRCTissueClass("MUS", 5, "smooth muscle"),
    CRCTissueClass("NORM", 6, "normal colon mucosa"),
    CRCTissueClass("STR", 7, "cancer-associated stroma"),
    CRCTissueClass(
        "TUM",
        8,
        "colorectal adenocarcinoma epithelium",
    ),
)

CRC_CLASS_BY_CODE = {
    tissue_class.code: tissue_class
    for tissue_class in CRC_TISSUE_CLASSES
}

SUPPORTED_IMAGE_EXTENSIONS = {
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
}


def scan_crc_directory(
    root_directory: str | Path,
    source_name: str,
) -> pd.DataFrame:
    """
    Индексирует изображения внутри NCT-CRC directory.

    Класс определяется по имени одной из родительских папок:
    ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR или TUM.
    """
    root_directory = Path(root_directory)

    if not root_directory.exists():
        raise FileNotFoundError(
            f"Dataset directory was not found: "
            f"{root_directory.resolve()}"
        )

    records: list[dict[str, object]] = []

    image_paths = sorted(
        path
        for path in root_directory.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_IMAGE_EXTENSIONS
        )
    )

    if not image_paths:
        raise FileNotFoundError(
            "No supported images were found inside: "
            f"{root_directory.resolve()}"
        )

    for image_path in image_paths:
        class_code = _find_class_code(
            image_path=image_path,
            root_directory=root_directory,
        )

        if class_code is None:
            raise ValueError(
                "Could not determine tissue class for image: "
                f"{image_path.resolve()}"
            )

        class_information = CRC_CLASS_BY_CODE[
            class_code
        ]

        sample_id = (
            f"{source_name}:{image_path.stem}"
        )

        # Архив не предоставляет публичное отображение
        # patch -> patient/slide. Эти значения являются
        # техническими fallback ID, а не patient IDs.
        fallback_group_id = (
            f"patch_fallback:{sample_id}"
        )

        records.append(
            {
                "sample_id": sample_id,
                "path": image_path.resolve().as_posix(),
                "source": source_name,
                "class_code": class_information.code,
                "class_name": class_information.name,
                "label": class_information.label,
                "patient_id": fallback_group_id,
                "slide_id": fallback_group_id,
                "group_id_source": "patch_id_fallback",
            }
        )

    metadata = pd.DataFrame.from_records(records)

    _validate_all_classes_are_present(metadata)

    return metadata


def build_crc_metadata(
    train_root: str | Path,
    external_test_root: str | Path,
    validation_fraction: float,
    seed: int,
    train_pool_per_class: int | None = None,
    external_test_per_class: int | None = None,
) -> pd.DataFrame:
    """
    Создаёт train, internal validation и external test metadata.

    Internal validation является patch-level split.
    CRC-VAL-HE-7K используется как отдельный external test.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "validation_fraction must be between 0 and 1."
        )

    train_pool = scan_crc_directory(
        root_directory=train_root,
        source_name="NCT-CRC-HE-100K",
    )

    external_test = scan_crc_directory(
        root_directory=external_test_root,
        source_name="CRC-VAL-HE-7K",
    )

    train_pool = _sample_per_class(
        metadata=train_pool,
        maximum_per_class=train_pool_per_class,
        seed=seed,
    )

    external_test = _sample_per_class(
        metadata=external_test,
        maximum_per_class=external_test_per_class,
        seed=seed + 1,
    )

    train_and_validation = (
        _assign_stratified_patch_split(
            metadata=train_pool,
            validation_fraction=validation_fraction,
            seed=seed,
        )
    )

    external_test = external_test.copy()
    external_test["split"] = "test"

    complete_metadata = pd.concat(
        [
            train_and_validation,
            external_test,
        ],
        ignore_index=True,
    )

    return (
        complete_metadata
        .sort_values(
            [
                "split",
                "label",
                "sample_id",
            ]
        )
        .reset_index(drop=True)
    )


def _find_class_code(
    image_path: Path,
    root_directory: Path,
) -> str | None:
    relative_path = image_path.relative_to(
        root_directory
    )

    for path_part in reversed(
        relative_path.parts[:-1]
    ):
        normalized_part = path_part.upper()

        if normalized_part in CRC_CLASS_BY_CODE:
            return normalized_part

    filename_prefix = (
        image_path.stem.split("-", maxsplit=1)[0]
        .upper()
    )

    if filename_prefix in CRC_CLASS_BY_CODE:
        return filename_prefix

    return None


def _sample_per_class(
    metadata: pd.DataFrame,
    maximum_per_class: int | None,
    seed: int,
) -> pd.DataFrame:
    if maximum_per_class is None:
        return metadata.copy()

    if maximum_per_class <= 0:
        raise ValueError(
            "maximum_per_class must be positive or null."
        )

    sampled_frames: list[pd.DataFrame] = []

    for class_index, (_, class_frame) in enumerate(
        metadata.groupby("class_code", sort=True)
    ):
        number_to_sample = min(
            maximum_per_class,
            len(class_frame),
        )

        sampled_frames.append(
            class_frame.sample(
                n=number_to_sample,
                random_state=seed + class_index,
                replace=False,
            )
        )

    return pd.concat(
        sampled_frames,
        ignore_index=True,
    )


def _assign_stratified_patch_split(
    metadata: pd.DataFrame,
    validation_fraction: float,
    seed: int,
) -> pd.DataFrame:
    result_frames: list[pd.DataFrame] = []

    rng = np.random.default_rng(seed)

    for _, class_frame in metadata.groupby(
        "class_code",
        sort=True,
    ):
        class_frame = (
            class_frame
            .sort_values("sample_id")
            .reset_index(drop=True)
            .copy()
        )

        indices = np.arange(len(class_frame))
        rng.shuffle(indices)

        validation_count = max(
            1,
            int(round(
                len(class_frame)
                * validation_fraction
            )),
        )

        validation_indices = set(
            indices[:validation_count].tolist()
        )

        class_frame["split"] = [
            (
                "validation"
                if index in validation_indices
                else "train"
            )
            for index in range(len(class_frame))
        ]

        result_frames.append(class_frame)

    return pd.concat(
        result_frames,
        ignore_index=True,
    )


def _validate_all_classes_are_present(
    metadata: pd.DataFrame,
) -> None:
    expected_classes = set(
        CRC_CLASS_BY_CODE
    )

    actual_classes = set(
        metadata["class_code"].unique()
    )

    missing_classes = (
        expected_classes - actual_classes
    )

    if missing_classes:
        raise ValueError(
            "Dataset is missing CRC tissue classes: "
            f"{sorted(missing_classes)}"
        )