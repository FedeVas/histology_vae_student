from __future__ import annotations

from dataclasses import dataclass

from src.models.losses import VAELossOutput


@dataclass(frozen=True)
class EpochMetrics:
    total_loss: float
    reconstruction_loss: float
    kl_loss: float
    reconstruction_loss_per_pixel: float
    kl_loss_per_dimension: float
    number_of_samples: int
    number_of_batches: int


class EpochMetricAccumulator:
    """
    Накапливает batch losses с учётом размера каждого batch.
    """

    def __init__(self) -> None:
        self.total_loss = 0.0
        self.reconstruction_loss = 0.0
        self.kl_loss = 0.0
        self.reconstruction_loss_per_pixel = 0.0
        self.kl_loss_per_dimension = 0.0

        self.number_of_samples = 0
        self.number_of_batches = 0

    def update(
        self,
        loss_output: VAELossOutput,
        batch_size: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        self.total_loss += (
            loss_output.total_loss.detach().item()
            * batch_size
        )

        self.reconstruction_loss += (
            loss_output.reconstruction_loss.detach().item()
            * batch_size
        )

        self.kl_loss += (
            loss_output.kl_loss.detach().item()
            * batch_size
        )

        self.reconstruction_loss_per_pixel += (
            loss_output.reconstruction_loss_per_pixel
            .detach()
            .item()
            * batch_size
        )

        self.kl_loss_per_dimension += (
            loss_output.kl_loss_per_dimension
            .detach()
            .item()
            * batch_size
        )

        self.number_of_samples += batch_size
        self.number_of_batches += 1

    def compute(self) -> EpochMetrics:
        if self.number_of_samples == 0:
            raise RuntimeError(
                "No samples were accumulated during the epoch."
            )

        denominator = float(self.number_of_samples)

        return EpochMetrics(
            total_loss=self.total_loss / denominator,
            reconstruction_loss=(
                self.reconstruction_loss / denominator
            ),
            kl_loss=self.kl_loss / denominator,
            reconstruction_loss_per_pixel=(
                self.reconstruction_loss_per_pixel
                / denominator
            ),
            kl_loss_per_dimension=(
                self.kl_loss_per_dimension
                / denominator
            ),
            number_of_samples=self.number_of_samples,
            number_of_batches=self.number_of_batches,
        )