from __future__ import annotations

import torch

from src.models.vae import ConvolutionalVAE


def build_vae_from_config(
    config: dict,
    target_device: torch.device | None = None,
) -> ConvolutionalVAE:
    """
    Создаёт ConvolutionalVAE из YAML-конфигурации.
    """
    data_config = config["data"]
    model_config = config["model"]

    model = ConvolutionalVAE(
        image_channels=int(data_config["channels"]),
        image_size=int(data_config["image_size"]),
        hidden_channels=[
            int(channel)
            for channel in model_config["hidden_channels"]
        ],
        latent_dim=int(model_config["latent_dim"]),
        log_var_min=float(model_config["log_var_min"]),
        log_var_max=float(model_config["log_var_max"]),
    )

    if target_device is not None:
        model = model.to(target_device)

    return model