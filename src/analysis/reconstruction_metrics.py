from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
)


@dataclass(frozen=True)
class ReconstructionMetrics:
    mse: float
    mae: float
    psnr: float
    ssim: float
    number_of_images: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "mse": self.mse,
            "mae": self.mae,
            "psnr": self.psnr,
            "ssim": self.ssim,
            "number_of_images": self.number_of_images,
        }


class ReconstructionMetricAccumulator:
    """
    Накапливает reconstruction metrics для всего dataset.

    Изображения ожидаются в формате:

        batch x channels x height x width

    и в диапазоне [0, 1].
    """

    def __init__(self) -> None:
        self.mse_sum = 0.0
        self.mae_sum = 0.0
        self.psnr_sum = 0.0
        self.ssim_sum = 0.0
        self.number_of_images = 0

    def update(
        self,
        target: torch.Tensor,
        reconstruction: torch.Tensor,
    ) -> None:
        self._validate_inputs(
            target=target,
            reconstruction=reconstruction,
        )

        target = target.detach().float().cpu()
        reconstruction = reconstruction.detach().float().cpu()

        squared_error = (
            reconstruction - target
        ).pow(2)

        absolute_error = (
            reconstruction - target
        ).abs()

        mse_per_image = squared_error.mean(
            dim=(1, 2, 3)
        )

        mae_per_image = absolute_error.mean(
            dim=(1, 2, 3)
        )

        target_numpy = (
            target
            .permute(0, 2, 3, 1)
            .numpy()
        )

        reconstruction_numpy = (
            reconstruction
            .permute(0, 2, 3, 1)
            .numpy()
        )

        batch_size = target.shape[0]

        self.mse_sum += float(
            mse_per_image.sum().item()
        )

        self.mae_sum += float(
            mae_per_image.sum().item()
        )

        for image_index in range(batch_size):
            target_image = target_numpy[image_index]
            reconstruction_image = (
                reconstruction_numpy[image_index]
            )

            psnr = peak_signal_noise_ratio(
                target_image,
                reconstruction_image,
                data_range=1.0,
            )

            ssim = structural_similarity(
                target_image,
                reconstruction_image,
                data_range=1.0,
                channel_axis=-1,
            )

            self.psnr_sum += float(psnr)
            self.ssim_sum += float(ssim)

        self.number_of_images += batch_size

    def compute(self) -> ReconstructionMetrics:
        if self.number_of_images == 0:
            raise RuntimeError(
                "No images were added to the metric accumulator."
            )

        denominator = float(self.number_of_images)

        return ReconstructionMetrics(
            mse=self.mse_sum / denominator,
            mae=self.mae_sum / denominator,
            psnr=self.psnr_sum / denominator,
            ssim=self.ssim_sum / denominator,
            number_of_images=self.number_of_images,
        )

    @staticmethod
    def _validate_inputs(
        target: torch.Tensor,
        reconstruction: torch.Tensor,
    ) -> None:
        if target.shape != reconstruction.shape:
            raise ValueError(
                "Target and reconstruction must have equal shapes. "
                f"target={tuple(target.shape)}, "
                f"reconstruction={tuple(reconstruction.shape)}."
            )

        if target.ndim != 4:
            raise ValueError(
                "Images must have four dimensions: "
                "batch, channels, height, width."
            )

        for tensor_name, tensor in (
            ("target", target),
            ("reconstruction", reconstruction),
        ):
            if not torch.isfinite(tensor).all():
                raise ValueError(
                    f"{tensor_name} contains non-finite values."
                )

            if tensor.min().item() < 0.0:
                raise ValueError(
                    f"{tensor_name} contains values below zero."
                )

            if tensor.max().item() > 1.0:
                raise ValueError(
                    f"{tensor_name} contains values above one."
                )