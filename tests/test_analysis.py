import pytest
import torch

from src.analysis.latent import (
    build_latent_statistics_frame,
    compute_latent_diagnostics,
)
from src.analysis.reconstruction_metrics import (
    ReconstructionMetricAccumulator,
)


def test_reconstruction_metrics_are_finite() -> None:
    generator = torch.Generator()
    generator.manual_seed(42)

    target = torch.rand(
        4,
        3,
        16,
        16,
        generator=generator,
    )

    reconstruction = torch.clamp(
        target + 0.05,
        min=0.0,
        max=1.0,
    )

    accumulator = (
        ReconstructionMetricAccumulator()
    )

    accumulator.update(
        target=target,
        reconstruction=reconstruction,
    )

    metrics = accumulator.compute()

    assert metrics.number_of_images == 4
    assert metrics.mse > 0.0
    assert metrics.mae > 0.0
    assert metrics.psnr > 0.0
    assert -1.0 <= metrics.ssim <= 1.0


def test_latent_diagnostics_detect_active_unit() -> None:
    mu = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.5, 0.0],
        ],
        dtype=torch.float32,
    )

    log_var = torch.zeros_like(mu)

    diagnostics = compute_latent_diagnostics(
        mu=mu,
        log_var=log_var,
        active_unit_variance_threshold=0.01,
        low_kl_threshold=0.001,
    )

    assert diagnostics.number_of_dimensions == 2
    assert diagnostics.number_of_active_units == 1
    assert diagnostics.active_unit_mask.tolist() == [
        True,
        False,
    ]

    assert diagnostics.low_kl_mask.tolist() == [
        False,
        True,
    ]


def test_latent_statistics_frame_has_one_row_per_dimension() -> None:
    mu = torch.randn(8, 5)
    log_var = torch.zeros_like(mu)

    diagnostics = compute_latent_diagnostics(
        mu=mu,
        log_var=log_var,
    )

    frame = build_latent_statistics_frame(
        diagnostics
    )

    assert len(frame) == 5

    assert set(frame.columns) == {
        "latent_dimension",
        "variance_of_mu",
        "mean_kl",
        "mean_absolute_mu",
        "mean_log_var",
        "active_unit",
        "low_kl_dimension",
    }


def test_invalid_latent_shapes_raise_error() -> None:
    with pytest.raises(ValueError):
        compute_latent_diagnostics(
            mu=torch.zeros(4, 5),
            log_var=torch.zeros(4, 6),
        )