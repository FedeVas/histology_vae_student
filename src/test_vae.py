import pytest
import torch

from src.models.decoder import ConvolutionalDecoder
from src.models.encoder import ConvolutionalEncoder
from src.models.losses import (
    compute_vae_loss,
    linear_kl_beta,
)
from src.models.vae import ConvolutionalVAE


IMAGE_SIZE = 32
IMAGE_CHANNELS = 3
HIDDEN_CHANNELS = [8, 16]
LATENT_DIM = 6
BATCH_SIZE = 4


def create_test_model() -> ConvolutionalVAE:
    return ConvolutionalVAE(
        image_channels=IMAGE_CHANNELS,
        image_size=IMAGE_SIZE,
        hidden_channels=HIDDEN_CHANNELS,
        latent_dim=LATENT_DIM,
        log_var_min=-10.0,
        log_var_max=10.0,
    )


def test_encoder_returns_expected_shapes() -> None:
    encoder = ConvolutionalEncoder(
        input_channels=IMAGE_CHANNELS,
        image_size=IMAGE_SIZE,
        hidden_channels=HIDDEN_CHANNELS,
        latent_dim=LATENT_DIM,
    )

    images = torch.rand(
        BATCH_SIZE,
        IMAGE_CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    mu, log_var = encoder(images)

    assert mu.shape == (
        BATCH_SIZE,
        LATENT_DIM,
    )

    assert log_var.shape == (
        BATCH_SIZE,
        LATENT_DIM,
    )


def test_decoder_returns_expected_shape_and_range() -> None:
    decoder = ConvolutionalDecoder(
        output_channels=IMAGE_CHANNELS,
        image_size=IMAGE_SIZE,
        hidden_channels=HIDDEN_CHANNELS,
        latent_dim=LATENT_DIM,
    )

    latent_vectors = torch.randn(
        BATCH_SIZE,
        LATENT_DIM,
    )

    reconstruction = decoder(latent_vectors)

    assert reconstruction.shape == (
        BATCH_SIZE,
        IMAGE_CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    assert reconstruction.min().item() >= 0.0
    assert reconstruction.max().item() <= 1.0


def test_vae_returns_expected_shapes() -> None:
    model = create_test_model()

    images = torch.rand(
        BATCH_SIZE,
        IMAGE_CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    output = model(images)

    assert output.reconstruction.shape == images.shape

    assert output.mu.shape == (
        BATCH_SIZE,
        LATENT_DIM,
    )

    assert output.log_var.shape == (
        BATCH_SIZE,
        LATENT_DIM,
    )

    assert output.z.shape == (
        BATCH_SIZE,
        LATENT_DIM,
    )


def test_vae_evaluation_is_deterministic() -> None:
    model = create_test_model()
    model.eval()

    images = torch.rand(
        BATCH_SIZE,
        IMAGE_CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    with torch.no_grad():
        first_output = model(images)
        second_output = model(images)

    torch.testing.assert_close(
        first_output.z,
        first_output.mu,
    )

    torch.testing.assert_close(
        second_output.z,
        second_output.mu,
    )

    torch.testing.assert_close(
        first_output.reconstruction,
        second_output.reconstruction,
    )


def test_vae_loss_supports_backward() -> None:
    model = create_test_model()
    model.train()

    images = torch.rand(
        BATCH_SIZE,
        IMAGE_CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    output = model(images)

    loss_output = compute_vae_loss(
        reconstruction=output.reconstruction,
        target=images,
        mu=output.mu,
        log_var=output.log_var,
        beta=1.0,
        reconstruction_type="mse",
    )

    loss_output.total_loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
        and parameter.grad is not None
    ]

    assert gradients

    assert all(
        torch.isfinite(gradient).all()
        for gradient in gradients
    )

    assert loss_output.total_loss.item() >= 0.0
    assert loss_output.reconstruction_loss.item() >= 0.0
    assert loss_output.kl_loss.item() >= 0.0


@pytest.mark.parametrize(
    ("epoch", "expected_beta"),
    [
        (0, 0.0),
        (2, 0.2),
        (5, 0.5),
        (10, 1.0),
        (20, 1.0),
    ],
)
def test_linear_kl_beta(
    epoch: int,
    expected_beta: float,
) -> None:
    actual_beta = linear_kl_beta(
        current_epoch=epoch,
        warmup_epochs=10,
        maximum_beta=1.0,
    )

    assert actual_beta == pytest.approx(
        expected_beta
    )


def test_invalid_image_size_raises_error() -> None:
    with pytest.raises(ValueError):
        ConvolutionalVAE(
            image_channels=3,
            image_size=30,
            hidden_channels=[8, 16, 32],
            latent_dim=6,
        )