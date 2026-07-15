from __future__ import annotations

from math import prod
from typing import Sequence

import torch
from torch import nn

from src.models.blocks import DownsampleBlock


class DeterministicConvolutionalEncoder(nn.Module):
    """
    Encoder обычного convolutional autoencoder.

    В отличие от VAE encoder возвращает один
    детерминированный latent vector.
    """

    def __init__(
        self,
        input_channels: int,
        image_size: int,
        hidden_channels: Sequence[int],
        latent_dim: int,
    ) -> None:
        super().__init__()

        self._validate_arguments(
            input_channels=input_channels,
            image_size=image_size,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
        )

        self.input_channels = int(input_channels)
        self.image_size = int(image_size)
        self.hidden_channels = tuple(
            int(channel)
            for channel in hidden_channels
        )
        self.latent_dim = int(latent_dim)

        downsample_factor = (
            2 ** len(self.hidden_channels)
        )

        self.encoded_spatial_size = (
            self.image_size // downsample_factor
        )

        blocks: list[nn.Module] = []
        current_channels = self.input_channels

        for output_channels in self.hidden_channels:
            blocks.append(
                DownsampleBlock(
                    input_channels=current_channels,
                    output_channels=output_channels,
                )
            )

            current_channels = output_channels

        self.backbone = nn.Sequential(*blocks)

        self.feature_shape = (
            self.hidden_channels[-1],
            self.encoded_spatial_size,
            self.encoded_spatial_size,
        )

        flattened_feature_dim = prod(
            self.feature_shape
        )

        self.embedding_head = nn.Linear(
            in_features=flattened_feature_dim,
            out_features=self.latent_dim,
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_input(inputs)

        features = self.backbone(inputs)

        expected_shape = (
            inputs.shape[0],
            *self.feature_shape,
        )

        if tuple(features.shape) != expected_shape:
            raise RuntimeError(
                "Unexpected encoder feature shape. "
                f"Expected {expected_shape}, "
                f"received {tuple(features.shape)}."
            )

        flattened_features = features.flatten(
            start_dim=1
        )

        return self.embedding_head(
            flattened_features
        )

    def _validate_input(
        self,
        inputs: torch.Tensor,
    ) -> None:
        if inputs.ndim != 4:
            raise ValueError(
                "Encoder input must have shape "
                "batch x channels x height x width."
            )

        if inputs.shape[1] != self.input_channels:
            raise ValueError(
                "Unexpected number of channels. "
                f"Expected {self.input_channels}, "
                f"received {inputs.shape[1]}."
            )

        if tuple(inputs.shape[-2:]) != (
            self.image_size,
            self.image_size,
        ):
            raise ValueError(
                "Unexpected image size. "
                f"Expected {self.image_size} x "
                f"{self.image_size}, "
                f"received {tuple(inputs.shape[-2:])}."
            )

    @staticmethod
    def _validate_arguments(
        input_channels: int,
        image_size: int,
        hidden_channels: Sequence[int],
        latent_dim: int,
    ) -> None:
        if input_channels <= 0:
            raise ValueError(
                "input_channels must be positive."
            )

        if image_size <= 0:
            raise ValueError(
                "image_size must be positive."
            )

        if latent_dim <= 0:
            raise ValueError(
                "latent_dim must be positive."
            )

        if not hidden_channels:
            raise ValueError(
                "hidden_channels must not be empty."
            )

        if any(
            channel <= 0
            for channel in hidden_channels
        ):
            raise ValueError(
                "All hidden channels must be positive."
            )

        downsample_factor = (
            2 ** len(hidden_channels)
        )

        if image_size % downsample_factor != 0:
            raise ValueError(
                "image_size must be divisible by "
                f"{downsample_factor}."
            )