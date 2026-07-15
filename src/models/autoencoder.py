from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from src.models.decoder import (
    ConvolutionalDecoder,
)
from src.models.deterministic_encoder import (
    DeterministicConvolutionalEncoder,
)
from src.models.output import (
    RepresentationModelOutput,
)


class ConvolutionalAutoencoder(nn.Module):
    """
    Детерминированный convolutional autoencoder.
    """

    def __init__(
        self,
        image_channels: int,
        image_size: int,
        hidden_channels: Sequence[int],
        latent_dim: int,
    ) -> None:
        super().__init__()

        self.image_channels = int(image_channels)
        self.image_size = int(image_size)
        self.hidden_channels = tuple(
            int(channel)
            for channel in hidden_channels
        )
        self.latent_dim = int(latent_dim)

        self.encoder = (
            DeterministicConvolutionalEncoder(
                input_channels=self.image_channels,
                image_size=self.image_size,
                hidden_channels=self.hidden_channels,
                latent_dim=self.latent_dim,
            )
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
    ) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(
        self,
        latent_vectors: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(latent_vectors)

    def forward(
        self,
        inputs: torch.Tensor,
        sample_posterior: bool | None = None,
    ) -> RepresentationModelOutput:
        """
        sample_posterior принимается для совместимости
        с интерфейсом VAE, но для Autoencoder игнорируется.
        """
        del sample_posterior

        embedding = self.encode(inputs)
        reconstruction = self.decode(embedding)

        return RepresentationModelOutput(
            reconstruction=reconstruction,
            embedding=embedding,
            z=embedding,
            mu=None,
            log_var=None,
        )