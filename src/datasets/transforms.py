from __future__ import annotations

import random
import torch
from collections.abc import Callable

from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as transform_functional


class RandomQuarterTurn:
    """
    Поворачивает изображение на 0, 90, 180 или 270 градусов.

    Для гистологии ориентация патча обычно не несёт
    фиксированного пространственного смысла.
    """

    def __init__(self, probability: float = 1.0) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Probability must be between 0 and 1.")

        self.probability = probability

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.probability:
            return image

        quarter_turns = random.randint(0, 3)
        angle = quarter_turns * 90

        return transform_functional.rotate(
            image,
            angle=angle,
        )


def build_train_transforms(
    image_size: int,
    horizontal_flip_probability: float = 0.5,
    vertical_flip_probability: float = 0.5,
    use_random_quarter_turn: bool = True,
    color_mode: str = "rgb",
) -> Callable:
    """
    Transforms для train split.

    На первом этапе не используем сильную цветовую аугментацию,
    чтобы отдельно оценить влияние H&E окрашивания.
    """
    augmentation_steps: list[Callable] = [
        transforms.Resize(
            size=(image_size, image_size),
            antialias=True,
        ),
        *_build_color_transforms(
            color_mode=color_mode,
        ),
        transforms.RandomHorizontalFlip(
            p=horizontal_flip_probability,
        ),
        transforms.RandomVerticalFlip(
            p=vertical_flip_probability,
        ),
    ]

    if use_random_quarter_turn:
        augmentation_steps.append(RandomQuarterTurn())

    augmentation_steps.append(transforms.ToTensor())

    return transforms.Compose(augmentation_steps)


def build_evaluation_transforms(
    image_size: int,
    color_mode: str = "rgb",
) -> Callable:
    """
    Детерминированные transforms для validation и test.
    """
    return transforms.Compose(
        [
            transforms.Resize(
                size=(image_size, image_size),
                antialias=True,
            ),
            transforms.ToTensor(),
            *_build_color_transforms(
                color_mode=color_mode,
            ),
        ]
    )


def _validate_color_mode(
    color_mode: str,
) -> str:
    normalized_color_mode = (
        str(color_mode).strip().lower()
    )

    supported_modes = {
        "rgb",
        "grayscale",
    }

    if normalized_color_mode not in supported_modes:
        raise ValueError(
            f"Unsupported color mode: "
            f"{color_mode!r}. "
            f"Available modes: "
            f"{sorted(supported_modes)}."
        )

    return normalized_color_mode


def _build_color_transforms(
    color_mode: str,
) -> list:
    normalized_color_mode = (
        _validate_color_mode(
            color_mode
        )
    )

    if normalized_color_mode == "rgb":
        return []

    return [
        transforms.Grayscale(
            num_output_channels=3,
        )
    ]

class ColorDenoisingPairTransform:
    """
    Создаёт пару:

        input_tensor:
            геометрически преобразованное изображение
            с дополнительным ColorJitter;

        target_tensor:
            то же геометрическое преобразование,
            но без изменения цвета.

    Input и target всегда имеют одинаковую геометрию.
    """

    def __init__(
        self,
        image_size: int,
        horizontal_flip_probability: float = 0.5,
        vertical_flip_probability: float = 0.5,
        random_quarter_turn: bool = True,
        brightness: float = 0.15,
        contrast: float = 0.15,
        saturation: float = 0.20,
        hue: float = 0.03,
    ) -> None:
        if image_size <= 0:
            raise ValueError(
                "image_size must be positive."
            )

        for probability_name, probability in {
            "horizontal_flip_probability": (
                horizontal_flip_probability
            ),
            "vertical_flip_probability": (
                vertical_flip_probability
            ),
        }.items():
            if not 0.0 <= probability <= 1.0:
                raise ValueError(
                    f"{probability_name} must be "
                    "between 0 and 1."
                )

        for parameter_name, parameter_value in {
            "brightness": brightness,
            "contrast": contrast,
            "saturation": saturation,
            "hue": hue,
        }.items():
            if parameter_value < 0:
                raise ValueError(
                    f"{parameter_name} must be "
                    "non-negative."
                )

        if hue > 0.5:
            raise ValueError(
                "hue must not exceed 0.5."
            )

        self.image_size = int(image_size)

        self.horizontal_flip_probability = float(
            horizontal_flip_probability
        )

        self.vertical_flip_probability = float(
            vertical_flip_probability
        )

        self.random_quarter_turn = bool(
            random_quarter_turn
        )

        self.color_jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(
        self,
        image: Image.Image,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        clean_image = image.convert("RGB")

        clean_image = transform_functional.resize(
            clean_image,
            size=[
                self.image_size,
                self.image_size,
            ],
        )

        if (
            torch.rand(1).item()
            < self.horizontal_flip_probability
        ):
            clean_image = transform_functional.hflip(
                clean_image
            )

        if (
            torch.rand(1).item()
            < self.vertical_flip_probability
        ):
            clean_image = transform_functional.vflip(
                clean_image
            )

        if self.random_quarter_turn:
            number_of_turns = int(
                torch.randint(
                    low=0,
                    high=4,
                    size=(1,),
                ).item()
            )

            if number_of_turns:
                clean_image = transform_functional.rotate(
                    clean_image,
                    angle=90 * number_of_turns,
                )

        corrupted_image = self.color_jitter(
            clean_image
        )

        input_tensor = transform_functional.to_tensor(
            corrupted_image
        )

        target_tensor = transform_functional.to_tensor(
            clean_image
        )

        return input_tensor, target_tensor


def build_color_denoising_pair_transform(
    image_size: int,
    horizontal_flip_probability: float = 0.5,
    vertical_flip_probability: float = 0.5,
    random_quarter_turn: bool = True,
    brightness: float = 0.15,
    contrast: float = 0.15,
    saturation: float = 0.20,
    hue: float = 0.03,
) -> ColorDenoisingPairTransform:
    return ColorDenoisingPairTransform(
        image_size=image_size,
        horizontal_flip_probability=(
            horizontal_flip_probability
        ),
        vertical_flip_probability=(
            vertical_flip_probability
        ),
        random_quarter_turn=random_quarter_turn,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )