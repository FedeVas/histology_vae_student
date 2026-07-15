from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.models.losses import compute_vae_loss
from src.models.vae import ConvolutionalVAE
from src.training.metrics import (
    EpochMetricAccumulator,
    EpochMetrics,
)
from src.utils.device import RuntimeDevice


def train_one_epoch(
    model: ConvolutionalVAE,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    runtime: RuntimeDevice,
    beta: float,
    reconstruction_type: str,
    gradient_clip_norm: float | None,
    scaler: Any | None = None,
    writer: SummaryWriter | None = None,
    global_step: int = 0,
    log_every_n_steps: int = 10,
    max_batches: int | None = None,
) -> tuple[EpochMetrics, int]:
    """
    Выполняет одну train epoch.
    """
    if log_every_n_steps <= 0:
        raise ValueError(
            "log_every_n_steps must be greater than zero."
        )

    model.train()

    accumulator = EpochMetricAccumulator()

    for batch_index, batch in enumerate(data_loader):
        if (
            max_batches is not None
            and batch_index >= max_batches
        ):
            break

        images = _move_images_to_device(
            batch=batch,
            runtime=runtime,
        )

        optimizer.zero_grad(set_to_none=True)

        with _autocast_context(runtime):
            output = model(
                images,
                sample_posterior=True,
            )

            loss_output = compute_vae_loss(
                reconstruction=output.reconstruction,
                target=images,
                mu=output.mu,
                log_var=output.log_var,
                beta=beta,
                reconstruction_type=reconstruction_type,
            )

        gradient_norm: float | None = None

        if scaler is not None:
            scaler.scale(
                loss_output.total_loss
            ).backward()

            scaler.unscale_(optimizer)

            gradient_norm = _clip_gradients(
                model=model,
                maximum_norm=gradient_clip_norm,
            )

            scaler.step(optimizer)
            scaler.update()

        else:
            loss_output.total_loss.backward()

            gradient_norm = _clip_gradients(
                model=model,
                maximum_norm=gradient_clip_norm,
            )

            optimizer.step()

        accumulator.update(
            loss_output=loss_output,
            batch_size=images.shape[0],
        )

        global_step += 1

        if (
            writer is not None
            and global_step % log_every_n_steps == 0
        ):
            writer.add_scalar(
                "batch/train_total_loss",
                loss_output.total_loss.detach().item(),
                global_step,
            )

            writer.add_scalar(
                "batch/train_reconstruction_loss_per_pixel",
                loss_output
                .reconstruction_loss_per_pixel
                .detach()
                .item(),
                global_step,
            )

            writer.add_scalar(
                "batch/train_kl_loss_per_dimension",
                loss_output
                .kl_loss_per_dimension
                .detach()
                .item(),
                global_step,
            )

            writer.add_scalar(
                "batch/train_beta",
                beta,
                global_step,
            )

            if gradient_norm is not None:
                writer.add_scalar(
                    "batch/gradient_norm",
                    gradient_norm,
                    global_step,
                )

    return accumulator.compute(), global_step


@torch.inference_mode()
def validate_one_epoch(
    model: ConvolutionalVAE,
    data_loader: DataLoader,
    runtime: RuntimeDevice,
    beta: float,
    reconstruction_type: str,
    max_batches: int | None = None,
) -> EpochMetrics:
    """
    Выполняет deterministic validation epoch.

    Для reconstruction используется posterior mean z = mu.
    """
    model.eval()

    accumulator = EpochMetricAccumulator()

    for batch_index, batch in enumerate(data_loader):
        if (
            max_batches is not None
            and batch_index >= max_batches
        ):
            break

        images = _move_images_to_device(
            batch=batch,
            runtime=runtime,
        )

        with _autocast_context(runtime):
            output = model(
                images,
                sample_posterior=False,
            )

            loss_output = compute_vae_loss(
                reconstruction=output.reconstruction,
                target=images,
                mu=output.mu,
                log_var=output.log_var,
                beta=beta,
                reconstruction_type=reconstruction_type,
            )

        accumulator.update(
            loss_output=loss_output,
            batch_size=images.shape[0],
        )

    return accumulator.compute()


def _move_images_to_device(
    batch: dict,
    runtime: RuntimeDevice,
) -> torch.Tensor:
    if "image" not in batch:
        raise KeyError(
            "DataLoader batch does not contain an 'image' key."
        )

    images = batch["image"]

    if not isinstance(images, torch.Tensor):
        raise TypeError(
            "batch['image'] must be a torch.Tensor."
        )

    return images.to(
        runtime.device,
        non_blocking=runtime.pin_memory,
    )


def _autocast_context(
    runtime: RuntimeDevice,
):
    if runtime.mixed_precision:
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )

    return nullcontext()


def _clip_gradients(
    model: torch.nn.Module,
    maximum_norm: float | None,
) -> float | None:
    if maximum_norm is None:
        return None

    if maximum_norm <= 0:
        raise ValueError(
            "gradient_clip_norm must be greater than zero "
            "or null."
        )

    gradient_norm = clip_grad_norm_(
        parameters=model.parameters(),
        max_norm=maximum_norm,
    )

    if not torch.isfinite(gradient_norm):
        raise RuntimeError(
            "Non-finite gradient norm was detected."
        )

    return float(gradient_norm.detach().item())