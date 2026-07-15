from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as functional

from src.models.output import (
    RepresentationModelOutput,
)


@dataclass
class ModelLossOutput:
    total_loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    kl_loss: torch.Tensor

    reconstruction_loss_per_pixel: torch.Tensor
    kl_loss_per_dimension: torch.Tensor

    beta: float


# Обратная совместимость с предыдущим названием.
VAELossOutput = ModelLossOutput


def compute_model_loss(
    output: RepresentationModelOutput,
    target: torch.Tensor,
    model_type: str,
    beta: float,
    reconstruction_type: str = "mse",
) -> ModelLossOutput:
    """
    Выбирает loss в зависимости от типа модели.
    """
    normalized_model_type = (
        model_type.strip().lower()
    )

    if normalized_model_type == "autoencoder":
        return compute_autoencoder_loss(
            reconstruction=output.reconstruction,
            target=target,
            reconstruction_type=(
                reconstruction_type
            ),
        )

    if normalized_model_type == "vae":
        if output.mu is None:
            raise ValueError(
                "VAE output does not contain mu."
            )

        if output.log_var is None:
            raise ValueError(
                "VAE output does not contain log_var."
            )

        return compute_vae_loss(
            reconstruction=output.reconstruction,
            target=target,
            mu=output.mu,
            log_var=output.log_var,
            beta=beta,
            reconstruction_type=(
                reconstruction_type
            ),
        )

    raise ValueError(
        f"Unknown model type: {model_type!r}"
    )


def compute_autoencoder_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    reconstruction_type: str = "mse",
) -> ModelLossOutput:
    """
    Loss обычного autoencoder:

        total = reconstruction loss
    """
    elementwise_loss = (
        _compute_elementwise_reconstruction_loss(
            reconstruction=reconstruction,
            target=target,
            reconstruction_type=(
                reconstruction_type
            ),
        )
    )

    reconstruction_loss = (
        elementwise_loss
        .flatten(start_dim=1)
        .sum(dim=1)
        .mean()
    )

    number_of_pixels = target[0].numel()

    zero = reconstruction_loss.new_zeros(())

    return ModelLossOutput(
        total_loss=reconstruction_loss,
        reconstruction_loss=(
            reconstruction_loss
        ),
        kl_loss=zero,
        reconstruction_loss_per_pixel=(
            reconstruction_loss
            / number_of_pixels
        ),
        kl_loss_per_dimension=zero,
        beta=0.0,
    )


def compute_vae_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 1.0,
    reconstruction_type: str = "mse",
) -> ModelLossOutput:
    _validate_vae_inputs(
        reconstruction=reconstruction,
        target=target,
        mu=mu,
        log_var=log_var,
        beta=beta,
    )

    elementwise_loss = (
        _compute_elementwise_reconstruction_loss(
            reconstruction=reconstruction,
            target=target,
            reconstruction_type=(
                reconstruction_type
            ),
        )
    )

    reconstruction_loss = (
        elementwise_loss
        .flatten(start_dim=1)
        .sum(dim=1)
        .mean()
    )

    kl_loss = (
        -0.5
        * (
            1.0
            + log_var
            - mu.pow(2)
            - log_var.exp()
        )
        .sum(dim=1)
        .mean()
    )

    total_loss = (
        reconstruction_loss
        + float(beta) * kl_loss
    )

    return ModelLossOutput(
        total_loss=total_loss,
        reconstruction_loss=(
            reconstruction_loss
        ),
        kl_loss=kl_loss,
        reconstruction_loss_per_pixel=(
            reconstruction_loss
            / target[0].numel()
        ),
        kl_loss_per_dimension=(
            kl_loss / mu.shape[1]
        ),
        beta=float(beta),
    )


def _compute_elementwise_reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    reconstruction_type: str,
) -> torch.Tensor:
    if reconstruction.shape != target.shape:
        raise ValueError(
            "Reconstruction and target must "
            "have equal shapes."
        )

    normalized_type = (
        reconstruction_type.strip().lower()
    )

    if normalized_type == "mse":
        return functional.mse_loss(
            reconstruction,
            target,
            reduction="none",
        )

    if normalized_type == "l1":
        return functional.l1_loss(
            reconstruction,
            target,
            reduction="none",
        )

    raise ValueError(
        "reconstruction_type must be "
        f"'mse' or 'l1', got {normalized_type!r}."
    )


def linear_kl_beta(
    current_epoch: int,
    warmup_epochs: int,
    maximum_beta: float,
) -> float:
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

    progress = min(
        current_epoch / warmup_epochs,
        1.0,
    )

    return float(
        maximum_beta * progress
    )


def _validate_vae_inputs(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float,
) -> None:
    if reconstruction.shape != target.shape:
        raise ValueError(
            "Reconstruction and target must "
            "have equal shapes."
        )

    if mu.shape != log_var.shape:
        raise ValueError(
            "mu and log_var must have equal shapes."
        )

    if mu.ndim != 2:
        raise ValueError(
            "mu and log_var must have shape "
            "batch x latent_dim."
        )

    if reconstruction.shape[0] != mu.shape[0]:
        raise ValueError(
            "Image and latent batch sizes must match."
        )

    if beta < 0:
        raise ValueError(
            "beta must be non-negative."
        )