from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch


@dataclass(frozen=True)
class LatentDiagnostics:
    number_of_dimensions: int
    number_of_active_units: int
    active_fraction: float
    number_of_low_kl_dimensions: int
    mean_kl_per_dimension: float

    variance_of_mu: torch.Tensor
    mean_kl_by_dimension: torch.Tensor
    mean_absolute_mu_by_dimension: torch.Tensor
    mean_log_var_by_dimension: torch.Tensor

    active_unit_mask: torch.Tensor
    low_kl_mask: torch.Tensor

    def summary(self) -> dict[str, float | int]:
        return {
            "number_of_dimensions": (
                self.number_of_dimensions
            ),
            "number_of_active_units": (
                self.number_of_active_units
            ),
            "active_fraction": self.active_fraction,
            "number_of_low_kl_dimensions": (
                self.number_of_low_kl_dimensions
            ),
            "mean_kl_per_dimension": (
                self.mean_kl_per_dimension
            ),
        }


def compute_latent_diagnostics(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    active_unit_variance_threshold: float = 0.001,
    low_kl_threshold: float = 0.001,
) -> LatentDiagnostics:
    """
    Анализирует использование latent dimensions.

    Active unit:
        Var_x[mu_j(x)] > active_unit_variance_threshold

    Low-KL dimension:
        mean_x[KL_j(x)] < low_kl_threshold
    """
    if mu.shape != log_var.shape:
        raise ValueError(
            "mu and log_var must have equal shapes."
        )

    if mu.ndim != 2:
        raise ValueError(
            "mu and log_var must have shape "
            "number_of_samples x latent_dim."
        )

    if mu.shape[0] < 2:
        raise ValueError(
            "At least two samples are required."
        )

    if active_unit_variance_threshold < 0:
        raise ValueError(
            "active_unit_variance_threshold "
            "must be non-negative."
        )

    if low_kl_threshold < 0:
        raise ValueError(
            "low_kl_threshold must be non-negative."
        )

    mu = mu.detach().float().cpu()
    log_var = log_var.detach().float().cpu()

    variance_of_mu = mu.var(
        dim=0,
        unbiased=False,
    )

    kl_by_sample_and_dimension = -0.5 * (
        1.0
        + log_var
        - mu.pow(2)
        - log_var.exp()
    )

    mean_kl_by_dimension = (
        kl_by_sample_and_dimension.mean(dim=0)
    )

    active_unit_mask = (
        variance_of_mu
        > active_unit_variance_threshold
    )

    low_kl_mask = (
        mean_kl_by_dimension
        < low_kl_threshold
    )

    number_of_dimensions = int(mu.shape[1])

    number_of_active_units = int(
        active_unit_mask.sum().item()
    )

    number_of_low_kl_dimensions = int(
        low_kl_mask.sum().item()
    )

    return LatentDiagnostics(
        number_of_dimensions=number_of_dimensions,
        number_of_active_units=number_of_active_units,
        active_fraction=(
            number_of_active_units
            / number_of_dimensions
        ),
        number_of_low_kl_dimensions=(
            number_of_low_kl_dimensions
        ),
        mean_kl_per_dimension=float(
            mean_kl_by_dimension.mean().item()
        ),
        variance_of_mu=variance_of_mu,
        mean_kl_by_dimension=mean_kl_by_dimension,
        mean_absolute_mu_by_dimension=(
            mu.abs().mean(dim=0)
        ),
        mean_log_var_by_dimension=(
            log_var.mean(dim=0)
        ),
        active_unit_mask=active_unit_mask,
        low_kl_mask=low_kl_mask,
    )


def build_latent_statistics_frame(
    diagnostics: LatentDiagnostics,
) -> pd.DataFrame:
    """
    Создаёт таблицу со статистикой каждой latent dimension.
    """
    latent_dimensions = list(
        range(diagnostics.number_of_dimensions)
    )

    return pd.DataFrame(
        {
            "latent_dimension": latent_dimensions,
            "variance_of_mu": (
                diagnostics.variance_of_mu.numpy()
            ),
            "mean_kl": (
                diagnostics
                .mean_kl_by_dimension
                .numpy()
            ),
            "mean_absolute_mu": (
                diagnostics
                .mean_absolute_mu_by_dimension
                .numpy()
            ),
            "mean_log_var": (
                diagnostics
                .mean_log_var_by_dimension
                .numpy()
            ),
            "active_unit": (
                diagnostics
                .active_unit_mask
                .numpy()
            ),
            "low_kl_dimension": (
                diagnostics
                .low_kl_mask
                .numpy()
            ),
        }
    )