from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2hed
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.segmentation import watershed


@dataclass(frozen=True)
class NucleiSegmentationConfig:
    """
    Настройки классической (non-deep-learning) сегментации ядер
    на H&E patches.

    Пайплайн:
        1. color deconvolution (Ruifrok & Johnston) -> hematoxylin channel;
        2. Otsu thresholding (OpenCV);
        3. morphological opening (OpenCV) для удаления шума;
        4. distance transform (OpenCV) + watershed (scikit-image)
           для разделения соприкасающихся ядер;
        5. region-based признаки формы и плотности (scikit-image).

    Это classical computer vision baseline, а не dedicated
    nuclei-сегментатор (например StarDist или Cellpose). Он не
    требует обучения и весов, но не так точен, как модели,
    обученные на разметке ядер.

    sample_size уменьшает изображение перед сегментацией для
    скорости. None использует исходное разрешение.
    """

    sample_size: int | None = 256
    min_nucleus_area_px: int = 5
    watershed_min_distance: int = 4
    morphology_kernel_size: int = 3

    def validate(self) -> None:
        if (
            self.sample_size is not None
            and self.sample_size <= 0
        ):
            raise ValueError(
                "sample_size must be positive or null."
            )

        if self.min_nucleus_area_px <= 0:
            raise ValueError(
                "min_nucleus_area_px must be positive."
            )

        if self.watershed_min_distance <= 0:
            raise ValueError(
                "watershed_min_distance must be positive."
            )

        if self.morphology_kernel_size <= 0:
            raise ValueError(
                "morphology_kernel_size must be positive."
            )


def segment_nuclei_from_array(
    image_rgb: np.ndarray,
    config: NucleiSegmentationConfig | None = None,
) -> np.ndarray:
    """
    Возвращает instance-segmentation mask ядер.

    0 обозначает фон/цитоплазму, положительные целые числа —
    идентификаторы отдельных ядер после watershed-разделения.
    """
    if config is None:
        config = NucleiSegmentationConfig()

    config.validate()

    normalized_rgb = _normalize_rgb_array(image_rgb)

    # Color deconvolution: канал 0 соответствует hematoxylin,
    # который в H&E окрашивает ядра.
    hed = rgb2hed(normalized_rgb)
    hematoxylin = np.clip(hed[..., 0], 0.0, None)

    maximum_intensity = float(hematoxylin.max())
    signal_range = float(
        hematoxylin.max() - hematoxylin.min()
    )

    # Однородные (например чисто фоновые) patches дают
    # ненулевой, но практически константный hematoxylin-сигнал.
    # Otsu-порог на константном изображении вырожден и может
    # пометить весь patch как ткань, поэтому проверяем разброс
    # сигнала, а не только его абсолютную величину.
    if (
        maximum_intensity <= 1e-8
        or signal_range <= 1e-4
    ):
        return np.zeros(
            hematoxylin.shape,
            dtype=np.int32,
        )

    normalized_hematoxylin = (
        hematoxylin / maximum_intensity
    )

    hematoxylin_uint8 = (
        normalized_hematoxylin * 255.0
    ).astype(np.uint8)

    _, binary_mask = cv2.threshold(
        hematoxylin_uint8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            config.morphology_kernel_size,
            config.morphology_kernel_size,
        ),
    )

    opened_mask = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    tissue_mask = opened_mask.astype(bool)

    if not tissue_mask.any():
        return np.zeros(
            tissue_mask.shape,
            dtype=np.int32,
        )

    distance = cv2.distanceTransform(
        opened_mask,
        cv2.DIST_L2,
        5,
    )

    peak_coordinates = peak_local_max(
        distance,
        min_distance=config.watershed_min_distance,
        labels=tissue_mask,
    )

    peak_mask = np.zeros(
        distance.shape,
        dtype=bool,
    )

    if len(peak_coordinates) > 0:
        peak_mask[
            tuple(peak_coordinates.T)
        ] = True

    markers = label(peak_mask)

    if markers.max() == 0:
        # Нет чётких локальных максимумов: рассматриваем
        # каждый connected component как одно ядро.
        return label(tissue_mask).astype(np.int32)

    instance_mask = watershed(
        -distance,
        markers,
        mask=tissue_mask,
    )

    return instance_mask.astype(np.int32)


def extract_nuclei_features_from_array(
    image_rgb: np.ndarray,
    config: NucleiSegmentationConfig | None = None,
) -> dict[str, float]:
    """
    Извлекает признаки количества и формы ядер.

    Признаки формы (eccentricity, solidity) являются классическими
    дескрипторами nuclear pleomorphism, используемыми в ручном
    гистологическом грейдинге (например, компонент Nottingham grading).
    Это упрощённая, обучающая аппроксимация, а не валидированный
    клинический признак.
    """
    if config is None:
        config = NucleiSegmentationConfig()

    config.validate()

    normalized_rgb = _normalize_rgb_array(image_rgb)

    instance_mask = segment_nuclei_from_array(
        image_rgb=normalized_rgb,
        config=config,
    )

    region_properties = [
        region
        for region in regionprops(instance_mask)
        if region.area >= config.min_nucleus_area_px
    ]

    total_pixels = float(
        instance_mask.shape[0]
        * instance_mask.shape[1]
    )

    nuclei_count = len(region_properties)

    features: dict[str, float] = {
        "segmentation_nuclei_count": (
            float(nuclei_count)
        ),
        "segmentation_nuclei_density": (
            float(
                sum(
                    region.area
                    for region in region_properties
                )
            )
            / total_pixels
        ),
    }

    shape_descriptors = (
        "area",
        "eccentricity",
        "solidity",
        "equivalent_diameter_area",
    )

    for descriptor in shape_descriptors:
        if nuclei_count == 0:
            features[
                f"segmentation_mean_nuclei_{descriptor}"
            ] = 0.0
            features[
                f"segmentation_std_nuclei_{descriptor}"
            ] = 0.0
            continue

        values = np.asarray(
            [
                getattr(region, descriptor)
                for region in region_properties
            ],
            dtype=np.float64,
        )

        features[
            f"segmentation_mean_nuclei_{descriptor}"
        ] = float(values.mean())

        features[
            f"segmentation_std_nuclei_{descriptor}"
        ] = float(values.std())

    if not all(
        np.isfinite(value)
        for value in features.values()
    ):
        raise ValueError(
            "Extracted segmentation features contain "
            "non-finite values."
        )

    return features


def extract_nuclei_feature_frame(
    metadata: pd.DataFrame,
    config: NucleiSegmentationConfig | None = None,
    progress_every: int | None = 200,
) -> pd.DataFrame:
    """
    Извлекает признаки сегментации ядер для всех строк metadata.

    Зеркалирует интерфейс extract_color_feature_frame, чтобы
    признаки можно было использовать в том же linear-probe пайплайне
    (см. src/run_linear_probe.py, --feature-prefix segmentation_).
    """
    required_columns = {
        "path",
        "split",
        "label",
    }

    missing_columns = (
        required_columns.difference(
            metadata.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Metadata is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if metadata.empty:
        raise ValueError(
            "Metadata must not be empty."
        )

    if config is None:
        config = NucleiSegmentationConfig()

    config.validate()

    records: list[dict[str, object]] = []

    reset_metadata = metadata.reset_index(
        drop=True
    )

    for row_index, row in reset_metadata.iterrows():
        image_path = Path(str(row["path"]))

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image was not found: "
                f"{image_path.resolve()}"
            )

        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")

            if config.sample_size is not None:
                rgb_image = rgb_image.resize(
                    (
                        config.sample_size,
                        config.sample_size,
                    ),
                    resample=(
                        Image.Resampling.BILINEAR
                    ),
                )

            image_array = np.asarray(
                rgb_image,
                dtype=np.uint8,
            )

        segmentation_features = (
            extract_nuclei_features_from_array(
                image_rgb=image_array,
                config=config,
            )
        )

        metadata_record = row.to_dict()

        duplicate_columns = set(
            metadata_record
        ).intersection(segmentation_features)

        if duplicate_columns:
            raise ValueError(
                "Metadata already contains generated "
                "segmentation feature columns: "
                f"{sorted(duplicate_columns)}"
            )

        records.append(
            {
                **metadata_record,
                **segmentation_features,
            }
        )

        processed_images = row_index + 1

        if (
            progress_every is not None
            and progress_every > 0
            and processed_images % progress_every == 0
        ):
            print(
                f"Processed segmentation features: "
                f"{processed_images}/"
                f"{len(reset_metadata)}"
            )

    feature_frame = pd.DataFrame.from_records(
        records
    )

    feature_columns = get_nuclei_feature_columns(
        feature_frame
    )

    if feature_frame[
        feature_columns
    ].isna().any().any():
        raise RuntimeError(
            "Generated segmentation feature frame "
            "contains missing values."
        )

    return feature_frame


def get_nuclei_feature_columns(
    feature_frame: pd.DataFrame,
) -> list[str]:
    columns = [
        column
        for column in feature_frame.columns
        if column.startswith("segmentation_")
    ]

    if not columns:
        raise ValueError(
            "No segmentation feature columns were found."
        )

    return columns


def _normalize_rgb_array(
    image_rgb: np.ndarray,
) -> np.ndarray:
    image_rgb = np.asarray(image_rgb)

    if (
        image_rgb.ndim != 3
        or image_rgb.shape[-1] != 3
    ):
        raise ValueError(
            "RGB image must have shape "
            "height x width x 3."
        )

    if np.issubdtype(
        image_rgb.dtype,
        np.integer,
    ):
        maximum_value = float(
            np.iinfo(image_rgb.dtype).max
        )

        normalized_rgb = (
            image_rgb.astype(np.float32)
            / maximum_value
        )

    else:
        normalized_rgb = image_rgb.astype(
            np.float32
        )

        normalized_rgb = np.clip(
            normalized_rgb,
            0.0,
            1.0,
        )

    return normalized_rgb
