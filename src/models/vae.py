from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from src.models.decoder import ConvolutionalDecoder
from src.models.encoder import ConvolutionalEncoder


@dataclass
class VAEOutput:
    """
    Результат одного forward pass VAE.
    """

    reconstruction: torch.Tensor
    mu: torch.Tensor
    log_var: torch.Tensor
    z: torch.Tensor


class ConvolutionalVAE(nn.Module):
    """
    Convolutional Variational Autoencoder.
    """

    def __init__(
        self,
        image_channels: int,
        image_size: int,
        hidden_channels: Sequence[int],
        latent_dim: int,
        log_var_min: float = -10.0,
        log_var_max: float = 10.0,
    ) -> None:
        super().__init__()

        if log_var_min >= log_var_max:
            raise ValueError(
                "log_var_min must be smaller than log_var_max."
            )

        self.image_channels = image_channels
        self.image_size = image_size
        self.hidden_channels = tuple(hidden_channels)
        self.latent_dim = latent_dim

        self.log_var_min = float(log_var_min)
        self.log_var_max = float(log_var_max)

        self.encoder = ConvolutionalEncoder(
            input_channels=self.image_channels,
            image_size=self.image_size,
            hidden_channels=self.hidden_channels,
            latent_dim=self.latent_dim,
        )

        self.decoder = ConvolutionalDecoder(
            output_channels=self.image_channels,
            image_size=self.image_size,
            hidden_channels=self.hidden_channels,
            latent_dim=self.latent_dim,
        )

    def encode(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mu, log_var = self.encoder(inputs)

        log_var = torch.clamp(
            log_var,
            min=self.log_var_min,
            max=self.log_var_max,
        )

        return mu, log_var

    @staticmethod
    def reparameterize(
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        """
        Выполняет reparameterization:

            z = mu + sigma * epsilon

            epsilon ~ Normal(0, I)
            sigma = exp(0.5 * log_var)
        """
        if mu.shape != log_var.shape:
            raise ValueError(
                "mu and log_var must have equal shapes. "
                f"mu={tuple(mu.shape)}, "
                f"log_var={tuple(log_var.shape)}."
            )

        standard_deviation = torch.exp(
            0.5 * log_var
        )

        epsilon = torch.randn_like(
            standard_deviation
        )

        return mu + standard_deviation * epsilon

    def decode(
        self,
        latent_vectors: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(latent_vectors)

    def forward(
        self,
        inputs: torch.Tensor,
        sample_posterior: bool | None = None,
    ) -> VAEOutput:
        """
        В training mode по умолчанию используется sampling.

        В evaluation mode по умолчанию используется posterior mean,
        что делает reconstruction детерминированной.
        """
        mu, log_var = self.encode(inputs)

        if sample_posterior is None:
            sample_posterior = self.training

        if sample_posterior:
            z = self.reparameterize(
                mu=mu,
                log_var=log_var,
            )
        else:
            z = mu

        reconstruction = self.decode(z)

        return VAEOutput(
            reconstruction=reconstruction,
            mu=mu,
            log_var=log_var,
            z=z,
        )

    @torch.no_grad()
    def sample_prior(
        self,
        number_of_samples: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Генерирует изображения из стандартного normal prior.
        """
        if number_of_samples <= 0:
            raise ValueError(
                "number_of_samples must be greater than zero."
            )

        if device is None:
            device = next(self.parameters()).device

        latent_vectors = torch.randn(
            number_of_samples,
            self.latent_dim,
            device=device,
        )

        return self.decode(latent_vectors)