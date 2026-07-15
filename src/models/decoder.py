from __future__ import annotations

from math import prod
from typing import Sequence

import torch
from torch import nn

from src.models.blocks import UpsampleBlock


class ConvolutionalDecoder(nn.Module):
    """
    Convolutional decoder для Variational Autoencoder.

    Decoder преобразует latent vector z обратно
    в RGB-изображение.
    """

    def __init__(
        self,
        output_channels: int,
        image_size: int,
        hidden_channels: Sequence[int],
        latent_dim: int,
    ) -> None:
        super().__init__()

        self._validate_arguments(
            output_channels=output_channels,
            image_size=image_size,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
        )

        self.output_channels = output_channels
        self.image_size = image_size
        self.hidden_channels = tuple(hidden_channels)
        self.latent_dim = latent_dim

        self.upsample_factor = 2 ** len(self.hidden_channels)

        self.encoded_spatial_size = (
            self.image_size // self.upsample_factor
        )

        self.feature_shape = (
            self.hidden_channels[-1],
            self.encoded_spatial_size,
            self.encoded_spatial_size,
        )

        self.flattened_feature_dim = prod(self.feature_shape)

        self.input_projection = nn.Linear(
            in_features=self.latent_dim,
            out_features=self.flattened_feature_dim,
        )

        reversed_channels = list(
            reversed(self.hidden_channels)
        )

        upsampling_blocks: list[nn.Module] = []

        for input_channels, output_channels_for_block in zip(
            reversed_channels[:-1],
            reversed_channels[1:],
        ):
            upsampling_blocks.append(
                UpsampleBlock(
                    input_channels=input_channels,
                    output_channels=output_channels_for_block,
                )
            )

        self.upsampling_backbone = nn.Sequential(
            *upsampling_blocks
        )

        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=reversed_channels[-1],
                out_channels=self.output_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        latent_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        latent_vectors:
            Tensor формы:

                batch_size x latent_dim

        Returns
        -------
        Tensor
            Reconstruction формы:

                batch_size x channels x image_size x image_size
        """
        if latent_vectors.ndim != 2:
            raise ValueError(
                "Decoder input must have two dimensions: "
                "batch and latent dimension. "
                f"Received shape: {tuple(latent_vectors.shape)}"
            )

        if latent_vectors.shape[1] != self.latent_dim:
            raise ValueError(
                "Unexpected latent dimension. "
                f"Expected {self.latent_dim}, "
                f"received {latent_vectors.shape[1]}."
            )

        projected_features = self.input_projection(
            latent_vectors
        )

        features = projected_features.view(
            latent_vectors.shape[0],
            *self.feature_shape,
        )

        features = self.upsampling_backbone(features)
        reconstruction = self.output_layer(features)

        expected_output_shape = (
            latent_vectors.shape[0],
            self.output_channels,
            self.image_size,
            self.image_size,
        )

        if tuple(reconstruction.shape) != expected_output_shape:
            raise RuntimeError(
                "Unexpected decoder output shape. "
                f"Expected {expected_output_shape}, "
                f"received {tuple(reconstruction.shape)}."
            )

        return reconstruction

    @staticmethod
    def _validate_arguments(
        output_channels: int,
        image_size: int,
        hidden_channels: Sequence[int],
        latent_dim: int,
    ) -> None:
        if output_channels <= 0:
            raise ValueError(
                "output_channels must be greater than zero."
            )

        if image_size <= 0:
            raise ValueError(
                "image_size must be greater than zero."
            )

        if latent_dim <= 0:
            raise ValueError(
                "latent_dim must be greater than zero."
            )

        if not hidden_channels:
            raise ValueError(
                "hidden_channels must contain at least one value."
            )

        if any(channel <= 0 for channel in hidden_channels):
            raise ValueError(
                "All hidden channel values must be greater than zero."
            )

        upsample_factor = 2 ** len(hidden_channels)

        if image_size % upsample_factor != 0:
            raise ValueError(
                "image_size must be divisible by the total "
                "upsample factor. "
                f"image_size={image_size}, "
                f"upsample_factor={upsample_factor}."
            )