import numpy as np

from src.analysis.color_features import (
    ColorFeatureConfig,
    extract_color_features_from_array,
)


def test_color_features_are_finite() -> None:
    rng = np.random.default_rng(42)

    image = rng.integers(
        low=0,
        high=256,
        size=(32, 32, 3),
        dtype=np.uint8,
    )

    features = (
        extract_color_features_from_array(
            image_rgb=image
        )
    )

    assert features

    assert all(
        name.startswith("color_")
        for name in features
    )

    assert all(
        np.isfinite(value)
        for value in features.values()
    )


def test_color_features_ignore_pixel_positions() -> None:
    rng = np.random.default_rng(42)

    image = rng.integers(
        low=0,
        high=256,
        size=(24, 24, 3),
        dtype=np.uint8,
    )

    shuffled_pixels = (
        image.reshape(-1, 3)
        .copy()
    )

    rng.shuffle(
        shuffled_pixels,
        axis=0,
    )

    shuffled_image = (
        shuffled_pixels.reshape(
            image.shape
        )
    )

    original_features = (
        extract_color_features_from_array(
            image_rgb=image
        )
    )

    shuffled_features = (
        extract_color_features_from_array(
            image_rgb=shuffled_image
        )
    )

    assert original_features.keys() == (
        shuffled_features.keys()
    )

    for feature_name in original_features:
        assert np.isclose(
            original_features[feature_name],
            shuffled_features[feature_name],
            atol=1e-7,
        )


def test_histograms_are_normalized() -> None:
    image = np.full(
        shape=(16, 16, 3),
        fill_value=128,
        dtype=np.uint8,
    )

    config = ColorFeatureConfig(
        rgb_histogram_bins=8,
        hue_histogram_bins=10,
        saturation_value_histogram_bins=8,
    )

    features = (
        extract_color_features_from_array(
            image_rgb=image,
            config=config,
        )
    )

    red_histogram_values = [
        value
        for name, value in features.items()
        if name.startswith(
            "color_rgb_red_hist_"
        )
    ]

    assert np.isclose(
        sum(red_histogram_values),
        1.0,
    )