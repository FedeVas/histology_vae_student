from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from src.utils.device import RuntimeDevice


@torch.inference_mode()
def create_reconstruction_grid(
    model: torch.nn.Module,
    data_loader: DataLoader,
    runtime: RuntimeDevice,
    number_of_images: int = 8,
) -> torch.Tensor:
    """
    Создаёт grid из трёх строк:

        originals
        reconstructions
        absolute errors
    """
    if number_of_images <= 0:
        raise ValueError(
            "number_of_images must be greater than zero."
        )

    try:
        batch = next(iter(data_loader))
    except StopIteration as error:
        raise RuntimeError(
            "Cannot create preview from an empty DataLoader."
        ) from error

    images = batch["image"].to(
        runtime.device,
        non_blocking=runtime.pin_memory,
    )

    number_of_images = min(
        number_of_images,
        images.shape[0],
    )

    images = images[:number_of_images]

    was_training = model.training
    model.eval()

    output = model(
        images,
        sample_posterior=False,
    )

    originals = images.detach().cpu()
    reconstructions = (
        output.reconstruction.detach().cpu()
    )

    absolute_errors = torch.abs(
        originals - reconstructions
    )

    preview_images = torch.cat(
        [
            originals,
            reconstructions,
            absolute_errors,
        ],
        dim=0,
    )

    grid = make_grid(
        preview_images,
        nrow=number_of_images,
        padding=2,
    )

    if was_training:
        model.train()

    return grid


def save_reconstruction_grid(
    grid: torch.Tensor,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_image(
        grid,
        output_path,
    )