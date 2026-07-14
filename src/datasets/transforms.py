from __future__ import annotations

import random
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
        ]
    )