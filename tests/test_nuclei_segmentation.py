from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageDraw

from src.analysis.nuclei_segmentation import (
    NucleiSegmentationConfig,
    extract_nuclei_feature_frame,
    extract_nuclei_features_from_array,
    get_nuclei_feature_columns,
    segment_nuclei_from_array,
)


def _draw_synthetic_nuclei_image(
    image_size: int,
    number_of_nuclei: int,
    seed: int,
) -> np.ndarray:
    """
    Рисует упрощённый H&E-подобный patch: розовый фон и тёмные
    эллипсы-"ядра". Не претендует на биологическую точность —
    нужен только для проверки того, что сегментация реагирует на
    количество тёмных объектов ожидаемым образом.
    """
    rng = np.random.default_rng(seed)

    image = Image.new(
        mode="RGB",
        size=(image_size, image_size),
        color=(235, 205, 219),
    )

    draw = ImageDraw.Draw(image)

    for _ in range(number_of_nuclei):
        x_center = int(
            rng.integers(10, image_size - 10)
        )
        y_center = int(
            rng.integers(10, image_size - 10)
        )
        radius = int(rng.integers(3, 6))

        draw.ellipse(
            (
                x_center - radius,
                y_center - radius,
                x_center + radius,
                y_center + radius,
            ),
            fill=(80, 50, 120),
        )

    return np.asarray(image, dtype=np.uint8)


def test_segmentation_mask_has_expected_shape() -> None:
    image = _draw_synthetic_nuclei_image(
        image_size=96,
        number_of_nuclei=30,
        seed=0,
    )

    mask = segment_nuclei_from_array(image)

    assert mask.shape == (96, 96)
    assert mask.dtype == np.int32


def test_more_nuclei_yield_higher_detected_count() -> None:
    sparse_image = _draw_synthetic_nuclei_image(
        image_size=128,
        number_of_nuclei=15,
        seed=1,
    )

    dense_image = _draw_synthetic_nuclei_image(
        image_size=128,
        number_of_nuclei=90,
        seed=2,
    )

    config = NucleiSegmentationConfig(
        sample_size=None
    )

    sparse_features = (
        extract_nuclei_features_from_array(
            sparse_image,
            config=config,
        )
    )

    dense_features = (
        extract_nuclei_features_from_array(
            dense_image,
            config=config,
        )
    )

    assert (
        dense_features[
            "segmentation_nuclei_count"
        ]
        > sparse_features[
            "segmentation_nuclei_count"
        ]
    )


def test_blank_image_produces_zero_nuclei() -> None:
    blank_image = np.full(
        (64, 64, 3),
        fill_value=235,
        dtype=np.uint8,
    )

    features = (
        extract_nuclei_features_from_array(
            blank_image
        )
    )

    assert features[
        "segmentation_nuclei_count"
    ] == 0.0
    assert features[
        "segmentation_nuclei_density"
    ] == 0.0


def test_features_are_finite() -> None:
    image = _draw_synthetic_nuclei_image(
        image_size=96,
        number_of_nuclei=40,
        seed=3,
    )

    features = (
        extract_nuclei_features_from_array(
            image
        )
    )

    assert all(
        name.startswith("segmentation_")
        for name in features
    )

    assert all(
        np.isfinite(value)
        for value in features.values()
    )


def test_invalid_config_raises_error() -> None:
    with pytest.raises(ValueError):
        NucleiSegmentationConfig(
            min_nucleus_area_px=0
        ).validate()


def test_extract_nuclei_feature_frame_end_to_end(
    tmp_path: Path,
) -> None:
    records = []

    for index, (split, number_of_nuclei) in enumerate(
        [
            ("train", 20),
            ("train", 80),
            ("validation", 20),
            ("test", 80),
        ]
    ):
        image = _draw_synthetic_nuclei_image(
            image_size=64,
            number_of_nuclei=number_of_nuclei,
            seed=index,
        )

        image_path = (
            tmp_path / f"patch_{index}.png"
        )

        Image.fromarray(image).save(image_path)

        records.append(
            {
                "path": str(image_path),
                "split": split,
                "label": (
                    0
                    if number_of_nuclei < 50
                    else 1
                ),
                "patient_id": f"patient_{index}",
                "slide_id": f"slide_{index}",
            }
        )

    metadata = pd.DataFrame.from_records(
        records
    )

    feature_frame = (
        extract_nuclei_feature_frame(
            metadata=metadata,
            config=NucleiSegmentationConfig(
                sample_size=64
            ),
        )
    )

    feature_columns = (
        get_nuclei_feature_columns(
            feature_frame
        )
    )

    assert len(feature_frame) == len(metadata)
    assert feature_columns
    assert not feature_frame[
        feature_columns
    ].isna().any().any()
