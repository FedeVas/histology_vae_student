from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2hsv


@dataclass(frozen=True)
class ColorFeatureConfig:
    """
    Настройки пространственно-независимых цветовых признаков.

    Ни один признак не учитывает положение пикселя.
    Поэтому модель видит распределение цветов, но не архитектуру ткани.
    """

    rgb_histogram_bins: int = 16
    hue_histogram_bins: int = 18
    saturation_value_histogram_bins: int = 16

    def validate(self) -> None:
        values = {
            "rgb_histogram_bins": self.rgb_histogram_bins,
            "hue_histogram_bins": self.hue_histogram_bins,
            "saturation_value_histogram_bins": (
                self.saturation_value_histogram_bins
            ),
        }

        for name, value in values.items():
            if value <= 0:
                raise ValueError(
                    f"{name} must be positive, got {value}."
                )


def extract_color_features_from_array(
    image_rgb: np.ndarray,
    config: ColorFeatureConfig | None = None,
) -> dict[str, float]:
    """
    Извлекает пространственно-независимые RGB/HSV-признаки.

    Используются:
    - mean/std и квантили RGB;
    - RGB-гистограммы;
    - статистики saturation и value;
    - HSV-гистограммы;
    - circular statistics для hue.

    Перестановка пикселей не должна менять эти признаки.
    """
    if config is None:
        config = ColorFeatureConfig()

    config.validate()

    normalized_rgb = _normalize_rgb_array(
        image_rgb
    )

    hsv = rgb2hsv(normalized_rgb)

    rgb_pixels = normalized_rgb.reshape(-1, 3)
    hsv_pixels = hsv.reshape(-1, 3)

    features: dict[str, float] = {}

    rgb_channel_names = (
        "red",
        "green",
        "blue",
    )

    for channel_index, channel_name in enumerate(
        rgb_channel_names
    ):
        values = rgb_pixels[:, channel_index]

        _add_distribution_moments(
            features=features,
            prefix=f"color_rgb_{channel_name}",
            values=values,
        )

        _add_normalized_histogram(
            features=features,
            prefix=f"color_rgb_{channel_name}_hist",
            values=values,
            number_of_bins=(
                config.rgb_histogram_bins
            ),
            value_range=(0.0, 1.0),
        )

    hue = hsv_pixels[:, 0]
    saturation = hsv_pixels[:, 1]
    value = hsv_pixels[:, 2]

    # Hue является циклической величиной:
    # 0 и 1 соответствуют соседним оттенкам.
    hue_angles = 2.0 * np.pi * hue

    hue_sin_mean = float(
        np.sin(hue_angles).mean()
    )

    hue_cos_mean = float(
        np.cos(hue_angles).mean()
    )

    features[
        "color_hsv_hue_sin_mean"
    ] = hue_sin_mean

    features[
        "color_hsv_hue_cos_mean"
    ] = hue_cos_mean

    features[
        "color_hsv_hue_concentration"
    ] = float(
        np.sqrt(
            hue_sin_mean**2
            + hue_cos_mean**2
        )
    )

    _add_normalized_histogram(
        features=features,
        prefix="color_hsv_hue_hist",
        values=hue,
        number_of_bins=(
            config.hue_histogram_bins
        ),
        value_range=(0.0, 1.0),
    )

    for channel_name, channel_values in (
        ("saturation", saturation),
        ("value", value),
    ):
        _add_distribution_moments(
            features=features,
            prefix=(
                f"color_hsv_{channel_name}"
            ),
            values=channel_values,
        )

        _add_normalized_histogram(
            features=features,
            prefix=(
                f"color_hsv_{channel_name}_hist"
            ),
            values=channel_values,
            number_of_bins=(
                config
                .saturation_value_histogram_bins
            ),
            value_range=(0.0, 1.0),
        )

    if not all(
        np.isfinite(value)
        for value in features.values()
    ):
        raise ValueError(
            "Extracted color features contain "
            "non-finite values."
        )

    return features


def extract_color_feature_frame(
    metadata: pd.DataFrame,
    sample_size: int | None = 64,
    config: ColorFeatureConfig | None = None,
    progress_every: int | None = 500,
) -> pd.DataFrame:
    """
    Извлекает цветовые признаки для всех строк metadata.

    sample_size уменьшает изображение перед вычислением
    глобальных статистик, но пространственная информация
    всё равно не используется.
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

    if sample_size is not None and sample_size <= 0:
        raise ValueError(
            "sample_size must be positive or null."
        )

    if config is None:
        config = ColorFeatureConfig()

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

            if sample_size is not None:
                rgb_image = rgb_image.resize(
                    (
                        sample_size,
                        sample_size,
                    ),
                    resample=(
                        Image.Resampling.BILINEAR
                    ),
                )

            image_array = np.asarray(
                rgb_image,
                dtype=np.uint8,
            )

        color_features = (
            extract_color_features_from_array(
                image_rgb=image_array,
                config=config,
            )
        )

        metadata_record = row.to_dict()

        duplicate_columns = set(
            metadata_record
        ).intersection(color_features)

        if duplicate_columns:
            raise ValueError(
                "Metadata already contains generated "
                "color feature columns: "
                f"{sorted(duplicate_columns)}"
            )

        records.append(
            {
                **metadata_record,
                **color_features,
            }
        )

        processed_images = row_index + 1

        if (
            progress_every is not None
            and progress_every > 0
            and processed_images % progress_every == 0
        ):
            print(
                f"Processed color features: "
                f"{processed_images}/"
                f"{len(reset_metadata)}"
            )

    feature_frame = pd.DataFrame.from_records(
        records
    )

    feature_columns = get_color_feature_columns(
        feature_frame
    )

    if feature_frame[
        feature_columns
    ].isna().any().any():
        raise RuntimeError(
            "Generated color feature frame "
            "contains missing values."
        )

    return feature_frame


def get_color_feature_columns(
    feature_frame: pd.DataFrame,
) -> list[str]:
    columns = [
        column
        for column in feature_frame.columns
        if column.startswith("color_")
    ]

    if not columns:
        raise ValueError(
            "No color feature columns were found."
        )

    return columns


def _add_distribution_moments(
    features: dict[str, float],
    prefix: str,
    values: np.ndarray,
) -> None:
    quantile_values = np.quantile(
        values,
        [
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
        ],
    )

    features[
        f"{prefix}_mean"
    ] = float(values.mean())

    features[
        f"{prefix}_std"
    ] = float(values.std())

    for quantile_name, quantile_value in zip(
        (
            "q10",
            "q25",
            "q50",
            "q75",
            "q90",
        ),
        quantile_values,
    ):
        features[
            f"{prefix}_{quantile_name}"
        ] = float(quantile_value)


def _add_normalized_histogram(
    features: dict[str, float],
    prefix: str,
    values: np.ndarray,
    number_of_bins: int,
    value_range: tuple[float, float],
) -> None:
    histogram, _ = np.histogram(
        values,
        bins=number_of_bins,
        range=value_range,
    )

    histogram = histogram.astype(
        np.float64
    )

    histogram_sum = float(
        histogram.sum()
    )

    if histogram_sum <= 0:
        raise ValueError(
            "Histogram contains no observations."
        )

    histogram /= histogram_sum

    for bin_index, bin_value in enumerate(
        histogram
    ):
        features[
            f"{prefix}_{bin_index:03d}"
        ] = float(bin_value)


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

        minimum = float(
            normalized_rgb.min()
        )

        maximum = float(
            normalized_rgb.max()
        )

        if minimum < 0.0 or maximum > 1.0:
            raise ValueError(
                "Floating-point RGB values must "
                "be inside [0, 1]."
            )

    return np.clip(
        normalized_rgb,
        0.0,
        1.0,
    )