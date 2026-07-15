from __future__ import annotations

from math import prod
from typing import Sequence

import torch
from torch import nn

from src.models.blocks import DownsampleBlock


class ConvolutionalEncoder(nn.Module):
    """
    Convolutional encoder для Variational Autoencoder.

    Encoder преобразует изображение в параметры приближённого
    posterior-распределения:

        q(z | x) = Normal(mu, sigma^2)

    Возвращает:
        mu
        log_var
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

        self.input_channels = input_channels
        self.image_size = image_size
        self.hidden_channels = tuple(hidden_channels)
        self.latent_dim = latent_dim

        self.downsample_factor = 2 ** len(self.hidden_channels)

        self.encoded_spatial_size = (
            self.image_size // self.downsample_factor
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

        self.flattened_feature_dim = prod(self.feature_shape)

        self.mu_head = nn.Linear(
            in_features=self.flattened_feature_dim,
            out_features=self.latent_dim,
        )

        self.log_var_head = nn.Linear(
            in_features=self.flattened_feature_dim,
            out_features=self.latent_dim,
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        inputs:
            Tensor формы:

                batch_size x channels x height x width

        Returns
        -------
        tuple[Tensor, Tensor]
            mu и log_var формы:

                batch_size x latent_dim
        """
        self._validate_input_tensor(inputs)

        features = self.backbone(inputs)

        expected_feature_shape = (
            inputs.shape[0],
            *self.feature_shape,
        )

        if tuple(features.shape) != expected_feature_shape:
            raise RuntimeError(
                "Unexpected encoder feature shape. "
                f"Expected {expected_feature_shape}, "
                f"received {tuple(features.shape)}."
            )

        flattened_features = features.flatten(start_dim=1)

        mu = self.mu_head(flattened_features)
        log_var = self.log_var_head(flattened_features)

        return mu, log_var

    def _validate_input_tensor(
        self,
        inputs: torch.Tensor,
    ) -> None:
        if inputs.ndim != 4:
            raise ValueError(
                "Encoder input must have four dimensions: "
                "batch, channels, height, width. "
                f"Received shape: {tuple(inputs.shape)}"
            )

        if inputs.shape[1] != self.input_channels:
            raise ValueError(
                "Unexpected number of image channels. "
                f"Expected {self.input_channels}, "
                f"received {inputs.shape[1]}."
            )

        if tuple(inputs.shape[-2:]) != (
            self.image_size,
            self.image_size,
        ):
            raise ValueError(
                "Unexpected image size. "
                f"Expected {self.image_size} x {self.image_size}, "
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
                "input_channels must be greater than zero."
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

        downsample_factor = 2 ** len(hidden_channels)

        if image_size % downsample_factor != 0:
            raise ValueError(
                "image_size must be divisible by the total "
                "downsample factor. "
                f"image_size={image_size}, "
                f"downsample_factor={downsample_factor}."
            )

        encoded_spatial_size = image_size // downsample_factor

        if encoded_spatial_size < 1:
            raise ValueError(
                "Encoder contains too many downsampling blocks "
                "for the selected image size."
            )