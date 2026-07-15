from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as functional


@dataclass
class VAELossOutput:
    """
    Компоненты VAE loss.
    """

    total_loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    kl_loss: torch.Tensor

    reconstruction_loss_per_pixel: torch.Tensor
    kl_loss_per_dimension: torch.Tensor

    beta: float


def compute_vae_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 1.0,
    reconstruction_type: str = "mse",
) -> VAELossOutput:
    """
    Вычисляет loss:

        total_loss =
            reconstruction_loss
            + beta * kl_loss

    Reconstruction и KL сначала суммируются внутри каждого
    объекта batch, после чего усредняются по batch.

    Это соответствует стандартной per-sample форме ELBO.
    """
    _validate_loss_inputs(
        reconstruction=reconstruction,
        target=target,
        mu=mu,
        log_var=log_var,
        beta=beta,
    )

    reconstruction_type = (
        reconstruction_type.strip().lower()
    )

    if reconstruction_type == "mse":
        elementwise_reconstruction_loss = (
            functional.mse_loss(
                reconstruction,
                target,
                reduction="none",
            )
        )

    elif reconstruction_type == "l1":
        elementwise_reconstruction_loss = (
            functional.l1_loss(
                reconstruction,
                target,
                reduction="none",
            )
        )

    else:
        raise ValueError(
            "reconstruction_type must be 'mse' or 'l1'. "
            f"Received: {reconstruction_type!r}"
        )

    reconstruction_loss_per_sample = (
        elementwise_reconstruction_loss
        .flatten(start_dim=1)
        .sum(dim=1)
    )

    reconstruction_loss = (
        reconstruction_loss_per_sample.mean()
    )

    kl_loss_per_sample = -0.5 * torch.sum(
        1.0
        + log_var
        - mu.pow(2)
        - log_var.exp(),
        dim=1,
    )

    kl_loss = kl_loss_per_sample.mean()

    total_loss = (
        reconstruction_loss
        + float(beta) * kl_loss
    )

    number_of_pixels_per_sample = target[0].numel()
    number_of_latent_dimensions = mu.shape[1]

    reconstruction_loss_per_pixel = (
        reconstruction_loss
        / number_of_pixels_per_sample
    )

    kl_loss_per_dimension = (
        kl_loss
        / number_of_latent_dimensions
    )

    return VAELossOutput(
        total_loss=total_loss,
        reconstruction_loss=reconstruction_loss,
        kl_loss=kl_loss,
        reconstruction_loss_per_pixel=(
            reconstruction_loss_per_pixel
        ),
        kl_loss_per_dimension=kl_loss_per_dimension,
        beta=float(beta),
    )


def linear_kl_beta(
    current_epoch: int,
    warmup_epochs: int,
    maximum_beta: float,
) -> float:
    """
    Линейно увеличивает beta от 0 до maximum_beta.

    current_epoch считается от нуля.

    Пример для warmup_epochs=10:

        epoch 0  -> beta 0.0
        epoch 5  -> beta 0.5 * maximum_beta
        epoch 10 -> beta 1.0 * maximum_beta
    """
    if current_epoch < 0:
        raise ValueError(
            "current_epoch must be non-negative."
        )

    if warmup_epochs < 0:
        raise ValueError(
            "warmup_epochs must be non-negative."
        )

    if maximum_beta < 0:
        raise ValueError(
            "maximum_beta must be non-negative."
        )

    if warmup_epochs == 0:
        return float(maximum_beta)

    warmup_progress = min(
        current_epoch / warmup_epochs,
        1.0,
    )

    return float(
        maximum_beta * warmup_progress
    )


def _validate_loss_inputs(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float,
) -> None:
    if reconstruction.shape != target.shape:
        raise ValueError(
            "Reconstruction and target must have equal shapes. "
            f"reconstruction={tuple(reconstruction.shape)}, "
            f"target={tuple(target.shape)}."
        )

    if mu.shape != log_var.shape:
        raise ValueError(
            "mu and log_var must have equal shapes. "
            f"mu={tuple(mu.shape)}, "
            f"log_var={tuple(log_var.shape)}."
        )

    if mu.ndim != 2:
        raise ValueError(
            "mu and log_var must have shape "
            "batch_size x latent_dim."
        )

    if reconstruction.shape[0] != mu.shape[0]:
        raise ValueError(
            "Image batch size and latent batch size must match."
        )

    if beta < 0:
        raise ValueError(
            "beta must be non-negative."
        )