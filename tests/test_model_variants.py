import torch

from src.models.autoencoder import (
    ConvolutionalAutoencoder,
)
from src.models.factory import (
    build_model_from_config,
)
from src.models.losses import (
    compute_model_loss,
)


def create_autoencoder() -> (
    ConvolutionalAutoencoder
):
    return ConvolutionalAutoencoder(
        image_channels=3,
        image_size=32,
        hidden_channels=[8, 16],
        latent_dim=6,
    )


def test_autoencoder_output_shapes() -> None:
    model = create_autoencoder()

    images = torch.rand(
        4,
        3,
        32,
        32,
    )

    output = model(images)

    assert output.reconstruction.shape == (
        4,
        3,
        32,
        32,
    )

    assert output.embedding.shape == (
        4,
        6,
    )

    assert output.z.shape == (
        4,
        6,
    )

    assert output.mu is None
    assert output.log_var is None


def test_autoencoder_loss_supports_backward() -> None:
    model = create_autoencoder()

    images = torch.rand(
        4,
        3,
        32,
        32,
    )

    output = model(images)

    loss_output = compute_model_loss(
        output=output,
        target=images,
        model_type="autoencoder",
        beta=0.0,
        reconstruction_type="mse",
    )

    loss_output.total_loss.backward()

    assert loss_output.total_loss.item() > 0
    assert loss_output.kl_loss.item() == 0
    assert loss_output.beta == 0.0

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]

    assert gradients

    assert all(
        torch.isfinite(gradient).all()
        for gradient in gradients
    )


def test_factory_builds_autoencoder() -> None:
    config = {
        "data": {
            "channels": 3,
            "image_size": 32,
        },
        "model": {
            "type": "autoencoder",
            "latent_dim": 6,
            "hidden_channels": [8, 16],
            "log_var_min": -10.0,
            "log_var_max": 10.0,
        },
        "training": {
            "beta": 0.0,
        },
    }

    model = build_model_from_config(config)

    assert isinstance(
        model,
        ConvolutionalAutoencoder,
    )