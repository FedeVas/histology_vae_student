from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch


@dataclass(frozen=True)
class LatentDiagnostics:
    number_of_dimensions: int
    number_of_active_units: int
    active_fraction: float

    number_of_low_kl_dimensions: int | None
    mean_kl_per_dimension: float | None

    variance_of_embedding: torch.Tensor
    mean_kl_by_dimension: torch.Tensor
    mean_absolute_embedding: torch.Tensor
    mean_log_var_by_dimension: torch.Tensor

    active_unit_mask: torch.Tensor
    low_kl_mask: torch.Tensor

    def summary(
        self,
    ) -> dict[str, float | int | None]:
        return {
            "number_of_dimensions": (
                self.number_of_dimensions
            ),
            "number_of_active_units": (
                self.number_of_active_units
            ),
            "active_fraction": (
                self.active_fraction
            ),
            "number_of_low_kl_dimensions": (
                self.number_of_low_kl_dimensions
            ),
            "mean_kl_per_dimension": (
                self.mean_kl_per_dimension
            ),
        }


def compute_latent_diagnostics(
    latent_vectors: torch.Tensor,
    log_var: torch.Tensor | None = None,
    active_unit_variance_threshold: float = 0.001,
    low_kl_threshold: float = 0.001,
) -> LatentDiagnostics:
    """
    Для Autoencoder анализирует variance latent vectors.

    Для VAE дополнительно вычисляет KL по dimensions.
    """
    if latent_vectors.ndim != 2:
        raise ValueError(
            "latent_vectors must have shape "
            "samples x latent_dim."
        )

    if latent_vectors.shape[0] < 2:
        raise ValueError(
            "At least two samples are required."
        )

    if active_unit_variance_threshold < 0:
        raise ValueError(
            "active_unit_variance_threshold "
            "must be non-negative."
        )

    latent_vectors = (
        latent_vectors.detach().float().cpu()
    )

    variance = latent_vectors.var(
        dim=0,
        unbiased=False,
    )

    active_mask = (
        variance
        > active_unit_variance_threshold
    )

    number_of_dimensions = int(
        latent_vectors.shape[1]
    )

    number_of_active_units = int(
        active_mask.sum().item()
    )

    if log_var is None:
        nan_values = torch.full(
            size=(number_of_dimensions,),
            fill_value=float("nan"),
        )

        mean_kl_by_dimension = nan_values.clone()
        mean_log_var = nan_values.clone()

        low_kl_mask = torch.zeros(
            number_of_dimensions,
            dtype=torch.bool,
        )

        number_of_low_kl_dimensions = None
        mean_kl_per_dimension = None

    else:
        if latent_vectors.shape != log_var.shape:
            raise ValueError(
                "latent_vectors and log_var "
                "must have equal shapes."
            )

        log_var = (
            log_var.detach().float().cpu()
        )

        kl_by_sample_and_dimension = -0.5 * (
            1.0
            + log_var
            - latent_vectors.pow(2)
            - log_var.exp()
        )

        mean_kl_by_dimension = (
            kl_by_sample_and_dimension.mean(
                dim=0
            )
        )

        mean_log_var = log_var.mean(dim=0)

        low_kl_mask = (
            mean_kl_by_dimension
            < low_kl_threshold
        )

        number_of_low_kl_dimensions = int(
            low_kl_mask.sum().item()
        )

        mean_kl_per_dimension = float(
            mean_kl_by_dimension.mean().item()
        )

    return LatentDiagnostics(
        number_of_dimensions=(
            number_of_dimensions
        ),
        number_of_active_units=(
            number_of_active_units
        ),
        active_fraction=(
            number_of_active_units
            / number_of_dimensions
        ),
        number_of_low_kl_dimensions=(
            number_of_low_kl_dimensions
        ),
        mean_kl_per_dimension=(
            mean_kl_per_dimension
        ),
        variance_of_embedding=variance,
        mean_kl_by_dimension=(
            mean_kl_by_dimension
        ),
        mean_absolute_embedding=(
            latent_vectors.abs().mean(dim=0)
        ),
        mean_log_var_by_dimension=(
            mean_log_var
        ),
        active_unit_mask=active_mask,
        low_kl_mask=low_kl_mask,
    )


def build_latent_statistics_frame(
    diagnostics: LatentDiagnostics,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "latent_dimension": range(
                diagnostics.number_of_dimensions
            ),
            "variance_of_embedding": (
                diagnostics
                .variance_of_embedding
                .numpy()
            ),
            "mean_kl": (
                diagnostics
                .mean_kl_by_dimension
                .numpy()
            ),
            "mean_absolute_embedding": (
                diagnostics
                .mean_absolute_embedding
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