from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.models.vae import ConvolutionalVAE
from src.training.checkpoints import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from src.training.engine import (
    train_one_epoch,
    validate_one_epoch,
)
from src.utils.device import resolve_device

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.models.vae import ConvolutionalVAE
from src.training.checkpoints import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from src.training.engine import (
    train_one_epoch,
    validate_one_epoch,
)
from src.utils.device import resolve_device


class TinyImageDataset(Dataset):
    def __init__(
        self,
        number_of_images: int = 8,
        image_size: int = 16,
        seed: int = 42,
    ) -> None:
        generator = torch.Generator()
        generator.manual_seed(seed)

        self.images = torch.rand(
            number_of_images,
            3,
            image_size,
            image_size,
            generator=generator,
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict:
        return {
            "image": self.images[index],
        }


def create_tiny_model() -> ConvolutionalVAE:
    return ConvolutionalVAE(
        image_channels=3,
        image_size=16,
        hidden_channels=[4, 8],
        latent_dim=4,
        log_var_min=-10.0,
        log_var_max=10.0,
    )


def create_tiny_loader() -> DataLoader:
    return DataLoader(
        TinyImageDataset(),
        batch_size=4,
        shuffle=False,
    )


def test_train_epoch_updates_model_parameters() -> None:
    model = create_tiny_model()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    runtime = resolve_device(
        accelerator="cpu",
    )

    first_parameter_before = (
        next(model.parameters())
        .detach()
        .clone()
    )

    metrics, global_step = train_one_epoch(
        model=model,
        data_loader=create_tiny_loader(),
        optimizer=optimizer,
        runtime=runtime,
        beta=1.0,
        reconstruction_type="mse",
        gradient_clip_norm=5.0,
        scaler=None,
        writer=None,
        global_step=0,
        log_every_n_steps=1,
        max_batches=1,
    )

    first_parameter_after = (
        next(model.parameters())
        .detach()
        .clone()
    )

    assert not torch.equal(
        first_parameter_before,
        first_parameter_after,
    )

    assert global_step == 1
    assert metrics.number_of_batches == 1
    assert metrics.number_of_samples == 4
    assert metrics.total_loss >= 0.0
    assert metrics.reconstruction_loss >= 0.0
    assert metrics.kl_loss >= 0.0


def test_validation_epoch_returns_finite_metrics() -> None:
    model = create_tiny_model()

    runtime = resolve_device(
        accelerator="cpu",
    )

    metrics = validate_one_epoch(
        model=model,
        data_loader=create_tiny_loader(),
        runtime=runtime,
        beta=1.0,
        reconstruction_type="mse",
        max_batches=1,
    )

    assert metrics.number_of_batches == 1
    assert metrics.number_of_samples == 4

    assert torch.isfinite(
        torch.tensor(metrics.total_loss)
    )

    assert torch.isfinite(
        torch.tensor(
            metrics.reconstruction_loss_per_pixel
        )
    )


def test_checkpoint_roundtrip(
    tmp_path: Path,
) -> None:
    model = create_tiny_model()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    checkpoint_path = (
        tmp_path / "checkpoint.pt"
    )

    save_training_checkpoint(
        output_path=checkpoint_path,
        epoch=3,
        model=model,
        optimizer=optimizer,
        best_validation_loss=12.5,
        epochs_without_improvement=2,
        global_step=40,
        history=[
            {
                "epoch": 4,
                "validation_total_loss": 12.5,
            }
        ],
        config={
            "project": {
                "seed": 42,
            }
        },
        scaler=None,
    )

    restored_model = create_tiny_model()

    restored_optimizer = torch.optim.AdamW(
        restored_model.parameters(),
        lr=1e-3,
    )

    checkpoint = load_training_checkpoint(
        checkpoint_path=checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
        scaler=None,
        device="cpu",
    )

    assert checkpoint["epoch"] == 3
    assert checkpoint["global_step"] == 40
    assert (
        checkpoint["best_validation_loss"]
        == 12.5
    )

    original_state = model.state_dict()
    restored_state = restored_model.state_dict()

    assert original_state.keys() == restored_state.keys()

    for parameter_name in original_state:
        torch.testing.assert_close(
            original_state[parameter_name],
            restored_state[parameter_name],
        )
class TinyImageDataset(Dataset):
    def __init__(
        self,
        number_of_images: int = 8,
        image_size: int = 16,
        seed: int = 42,
    ) -> None:
        generator = torch.Generator()
        generator.manual_seed(seed)

        self.images = torch.rand(
            number_of_images,
            3,
            image_size,
            image_size,
            generator=generator,
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict:
        return {
            "image": self.images[index],
        }


def create_tiny_model() -> ConvolutionalVAE:
    return ConvolutionalVAE(
        image_channels=3,
        image_size=16,
        hidden_channels=[4, 8],
        latent_dim=4,
        log_var_min=-10.0,
        log_var_max=10.0,
    )


def create_tiny_loader() -> DataLoader:
    return DataLoader(
        TinyImageDataset(),
        batch_size=4,
        shuffle=False,
    )


def test_train_epoch_updates_model_parameters() -> None:
    model = create_tiny_model()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    runtime = resolve_device(
        accelerator="cpu",
    )

    first_parameter_before = (
        next(model.parameters())
        .detach()
        .clone()
    )

    metrics, global_step = train_one_epoch(
        model=model,
        data_loader=create_tiny_loader(),
        optimizer=optimizer,
        runtime=runtime,
        beta=1.0,
        reconstruction_type="mse",
        gradient_clip_norm=5.0,
        scaler=None,
        writer=None,
        global_step=0,
        log_every_n_steps=1,
        max_batches=1,
    )

    first_parameter_after = (
        next(model.parameters())
        .detach()
        .clone()
    )

    assert not torch.equal(
        first_parameter_before,
        first_parameter_after,
    )

    assert global_step == 1
    assert metrics.number_of_batches == 1
    assert metrics.number_of_samples == 4
    assert metrics.total_loss >= 0.0
    assert metrics.reconstruction_loss >= 0.0
    assert metrics.kl_loss >= 0.0


def test_validation_epoch_returns_finite_metrics() -> None:
    model = create_tiny_model()

    runtime = resolve_device(
        accelerator="cpu",
    )

    metrics = validate_one_epoch(
        model=model,
        data_loader=create_tiny_loader(),
        runtime=runtime,
        beta=1.0,
        reconstruction_type="mse",
        max_batches=1,
    )

    assert metrics.number_of_batches == 1
    assert metrics.number_of_samples == 4

    assert torch.isfinite(
        torch.tensor(metrics.total_loss)
    )

    assert torch.isfinite(
        torch.tensor(
            metrics.reconstruction_loss_per_pixel
        )
    )


def test_checkpoint_roundtrip(
    tmp_path: Path,
) -> None:
    model = create_tiny_model()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    checkpoint_path = (
        tmp_path / "checkpoint.pt"
    )

    save_training_checkpoint(
        output_path=checkpoint_path,
        epoch=3,
        model=model,
        optimizer=optimizer,
        best_validation_loss=12.5,
        epochs_without_improvement=2,
        global_step=40,
        history=[
            {
                "epoch": 4,
                "validation_total_loss": 12.5,
            }
        ],
        config={
            "project": {
                "seed": 42,
            }
        },
        scaler=None,
    )

    restored_model = create_tiny_model()

    restored_optimizer = torch.optim.AdamW(
        restored_model.parameters(),
        lr=1e-3,
    )

    checkpoint = load_training_checkpoint(
        checkpoint_path=checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
        scaler=None,
        device="cpu",
    )

    assert checkpoint["epoch"] == 3
    assert checkpoint["global_step"] == 40
    assert (
        checkpoint["best_validation_loss"]
        == 12.5
    )

    original_state = model.state_dict()
    restored_state = restored_model.state_dict()

    assert original_state.keys() == restored_state.keys()

    for parameter_name in original_state:
        torch.testing.assert_close(
            original_state[parameter_name],
            restored_state[parameter_name],
        )