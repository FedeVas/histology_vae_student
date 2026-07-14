from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def generate_synthetic_histology_dataset(
    output_dir: str | Path,
    metadata_path: str | Path,
    num_patients: int = 12,
    slides_per_patient: int = 2,
    patches_per_slide: int = 8,
    image_size: int = 128,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Создаёт небольшой synthetic histology-like dataset.

    Набор предназначен только для проверки:
        metadata
        patient splitting
        Dataset
        DataLoader
        training pipeline

    Он не подходит для биологических выводов.
    """
    if num_patients < 3:
        raise ValueError("num_patients must be at least 3.")

    if slides_per_patient < 1:
        raise ValueError("slides_per_patient must be at least 1.")

    if patches_per_slide < 1:
        raise ValueError("patches_per_slide must be at least 1.")

    if image_size < 32:
        raise ValueError("image_size must be at least 32.")

    output_dir = Path(output_dir)
    metadata_path = Path(metadata_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []

    for patient_index in range(num_patients):
        patient_id = f"patient_{patient_index:03d}"

        # Условный паттерн нужен только для проверки downstream-кода.
        synthetic_label = patient_index % 2

        for slide_index in range(slides_per_patient):
            slide_id = (
                f"{patient_id}_slide_{slide_index:02d}"
            )

            slide_dir = output_dir / patient_id / slide_id
            slide_dir.mkdir(parents=True, exist_ok=True)

            slide_color_shift = rng.integers(
                low=-10,
                high=11,
                size=3,
            )

            for patch_index in range(patches_per_slide):
                patch_id = (
                    f"{slide_id}_patch_{patch_index:04d}"
                )

                image = _create_synthetic_patch(
                    image_size=image_size,
                    label=synthetic_label,
                    slide_color_shift=slide_color_shift,
                    rng=rng,
                )

                patch_path = slide_dir / f"{patch_id}.png"
                image.save(patch_path)

                records.append(
                    {
                        "path": patch_path.as_posix(),
                        "patient_id": patient_id,
                        "slide_id": slide_id,
                        "patch_id": patch_id,
                        "label": synthetic_label,
                    }
                )

    metadata = pd.DataFrame.from_records(records)
    metadata.to_csv(metadata_path, index=False)

    return metadata


def _create_synthetic_patch(
    image_size: int,
    label: int,
    slide_color_shift: np.ndarray,
    rng: np.random.Generator,
) -> Image.Image:
    """
    Создаёт упрощённый histology-like RGB patch.
    """
    background_color = np.asarray(
        [238, 205, 219],
        dtype=np.int16,
    )

    background_color += slide_color_shift.astype(np.int16)
    background_color = np.clip(
        background_color,
        0,
        255,
    ).astype(np.uint8)

    image_array = np.empty(
        (image_size, image_size, 3),
        dtype=np.uint8,
    )
    image_array[:] = background_color

    pixel_noise = rng.normal(
        loc=0.0,
        scale=4.0,
        size=image_array.shape,
    )

    image_array = np.clip(
        image_array.astype(np.float32) + pixel_noise,
        0,
        255,
    ).astype(np.uint8)

    image = Image.fromarray(image_array, mode="RGB")
    draw = ImageDraw.Draw(image, mode="RGBA")

    # Условная эозинофильная тканевая структура.
    number_of_tissue_regions = int(
        rng.integers(4, 9)
    )

    for _ in range(number_of_tissue_regions):
        x_center = int(rng.integers(0, image_size))
        y_center = int(rng.integers(0, image_size))

        width = int(rng.integers(20, 55))
        height = int(rng.integers(12, 40))

        bounding_box = (
            x_center - width,
            y_center - height,
            x_center + width,
            y_center + height,
        )

        draw.ellipse(
            bounding_box,
            fill=(224, 142, 177, 35),
            outline=(194, 108, 151, 45),
            width=1,
        )

    # Label=1 получает более высокую условную клеточность.
    if label == 1:
        number_of_nuclei = int(
            rng.integers(90, 150)
        )
        radius_range = (2, 5)
    else:
        number_of_nuclei = int(
            rng.integers(35, 85)
        )
        radius_range = (2, 4)

    for _ in range(number_of_nuclei):
        x_center = int(rng.integers(0, image_size))
        y_center = int(rng.integers(0, image_size))

        radius_x = int(
            rng.integers(radius_range[0], radius_range[1] + 1)
        )
        radius_y = int(
            rng.integers(radius_range[0], radius_range[1] + 2)
        )

        nucleus_color = (
            int(rng.integers(62, 105)),
            int(rng.integers(35, 75)),
            int(rng.integers(105, 155)),
            int(rng.integers(145, 220)),
        )

        draw.ellipse(
            (
                x_center - radius_x,
                y_center - radius_y,
                x_center + radius_x,
                y_center + radius_y,
            ),
            fill=nucleus_color,
        )

    return image