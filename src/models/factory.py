from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from src.models.autoencoder import (
    ConvolutionalAutoencoder,
)
from src.models.vae import (
    ConvolutionalVAE,
)


SupportedModelType = Literal[
    "autoencoder",
    "vae",
]


def get_model_type(
    config: dict,
) -> SupportedModelType:
    model_type = str(
        config["model"].get("type", "vae")
    ).strip().lower()

    supported_types = {
        "autoencoder",
        "vae",
    }

    if model_type not in supported_types:
        raise ValueError(
            f"Unsupported model type: {model_type!r}. "
            f"Available types: {sorted(supported_types)}"
        )

    return model_type  # type: ignore[return-value]


def validate_model_training_config(
    config: dict,
) -> None:
    model_type = get_model_type(config)

    beta = float(
        config["training"]["beta"]
    )

    if beta < 0:
        raise ValueError(
            "training.beta must be non-negative."
        )

    if model_type == "autoencoder" and beta != 0.0:
        raise ValueError(
            "Autoencoder must use training.beta: 0.0. "
            "KL regularization is not part of the "
            "deterministic autoencoder objective."
        )


def build_model_from_config(
    config: dict,
    target_device: torch.device | None = None,
) -> nn.Module:
    """
    Создаёт Autoencoder или VAE из YAML-конфигурации.
    """
    validate_model_training_config(config)

    data_config = config["data"]
    model_config = config["model"]

    model_type = get_model_type(config)

    common_arguments = {
        "image_channels": int(
            data_config["channels"]
        ),
        "image_size": int(
            data_config["image_size"]
        ),
        "hidden_channels": [
            int(channel)
            for channel in model_config[
                "hidden_channels"
            ]
        ],
        "latent_dim": int(
            model_config["latent_dim"]
        ),
    }

    if model_type == "autoencoder":
        model: nn.Module = (
            ConvolutionalAutoencoder(
                **common_arguments,
            )
        )

    else:
        model = ConvolutionalVAE(
            **common_arguments,
            log_var_min=float(
                model_config["log_var_min"]
            ),
            log_var_max=float(
                model_config["log_var_max"]
            ),
        )

    if target_device is not None:
        model = model.to(target_device)

    return model


def build_vae_from_config(
    config: dict,
    target_device: torch.device | None = None,
) -> nn.Module:
    """
    Оставлено для совместимости со старым кодом.

    Новый код должен использовать build_model_from_config.
    """
    return build_model_from_config(
        config=config,
        target_device=target_device,
    )