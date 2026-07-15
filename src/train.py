from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from src.datasets.factory import (
    build_data_loaders,
    build_datasets,
    prepare_metadata,
)
from src.datasets.split import get_split_summary
from src.models.losses import linear_kl_beta
from src.models.factory import build_vae_from_config
from src.models.vae import ConvolutionalVAE
from src.training.checkpoints import (
    load_training_checkpoint,
    save_model_state,
    save_training_checkpoint,
)
from src.training.engine import (
    train_one_epoch,
    validate_one_epoch,
)
from src.training.metrics import EpochMetrics
from src.training.visualization import (
    create_reconstruction_grid,
    save_reconstruction_grid,
)
from src.utils.config import load_config
from src.utils.device import (
    RuntimeDevice,
    resolve_device,
)
from src.utils.reproducibility import seed_everything


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a convolutional VAE on histology patches."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/vae_base.yaml"),
        help="Path to the YAML configuration file.",
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional path to a training checkpoint.",
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run only two epochs with a small number of batches."
        ),
    )

    return parser.parse_args()


def create_optimizer(
    model: torch.nn.Module,
    config: dict,
) -> torch.optim.Optimizer:
    training_config = config["training"]

    return torch.optim.AdamW(
        model.parameters(),
        lr=float(
            training_config["learning_rate"]
        ),
        weight_decay=float(
            training_config["weight_decay"]
        ),
    )


def create_gradient_scaler(
    runtime: RuntimeDevice,
):
    if not runtime.mixed_precision:
        return None

    return torch.amp.GradScaler("cuda")


def create_run_directory(
    config: dict,
) -> Path:
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    run_name = str(
        config["logging"]["run_name"]
    )

    run_directory = (
        Path(config["project"]["output_dir"])
        / "runs"
        / f"{timestamp}_{run_name}"
    )

    run_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    return run_directory


def save_history(
    history: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(history).to_csv(
        output_path,
        index=False,
    )


def metrics_to_history_row(
    epoch: int,
    train_beta: float,
    train_metrics: EpochMetrics,
    validation_metrics: EpochMetrics,
) -> dict[str, Any]:
    return {
        "epoch": epoch + 1,
        "train_beta": train_beta,

        "train_total_loss": (
            train_metrics.total_loss
        ),
        "train_reconstruction_loss": (
            train_metrics.reconstruction_loss
        ),
        "train_kl_loss": (
            train_metrics.kl_loss
        ),
        "train_reconstruction_loss_per_pixel": (
            train_metrics
            .reconstruction_loss_per_pixel
        ),
        "train_kl_loss_per_dimension": (
            train_metrics
            .kl_loss_per_dimension
        ),

        "validation_total_loss": (
            validation_metrics.total_loss
        ),
        "validation_reconstruction_loss": (
            validation_metrics
            .reconstruction_loss
        ),
        "validation_kl_loss": (
            validation_metrics.kl_loss
        ),
        "validation_reconstruction_loss_per_pixel": (
            validation_metrics
            .reconstruction_loss_per_pixel
        ),
        "validation_kl_loss_per_dimension": (
            validation_metrics
            .kl_loss_per_dimension
        ),
    }


def log_epoch_metrics(
    writer: SummaryWriter,
    epoch: int,
    train_beta: float,
    train_metrics: EpochMetrics,
    validation_metrics: EpochMetrics,
) -> None:
    step = epoch + 1

    writer.add_scalars(
        "epoch/total_loss",
        {
            "train": train_metrics.total_loss,
            "validation": (
                validation_metrics.total_loss
            ),
        },
        step,
    )

    writer.add_scalars(
        "epoch/reconstruction_loss_per_pixel",
        {
            "train": (
                train_metrics
                .reconstruction_loss_per_pixel
            ),
            "validation": (
                validation_metrics
                .reconstruction_loss_per_pixel
            ),
        },
        step,
    )

    writer.add_scalars(
        "epoch/kl_loss_per_dimension",
        {
            "train": (
                train_metrics
                .kl_loss_per_dimension
            ),
            "validation": (
                validation_metrics
                .kl_loss_per_dimension
            ),
        },
        step,
    )

    writer.add_scalar(
        "epoch/train_beta",
        train_beta,
        step,
    )


def print_epoch_summary(
    epoch: int,
    total_epochs: int,
    train_beta: float,
    train_metrics: EpochMetrics,
    validation_metrics: EpochMetrics,
    improved: bool,
) -> None:
    improvement_marker = (
        " [BEST]"
        if improved
        else ""
    )

    print(
        f"Epoch {epoch + 1:03d}/{total_epochs:03d} "
        f"| beta={train_beta:.4f}"
    )

    print(
        "  train      "
        f"total={train_metrics.total_loss:.4f} "
        f"recon/pixel="
        f"{train_metrics.reconstruction_loss_per_pixel:.6f} "
        f"kl/dim="
        f"{train_metrics.kl_loss_per_dimension:.6f}"
    )

    print(
        "  validation "
        f"total={validation_metrics.total_loss:.4f} "
        f"recon/pixel="
        f"{validation_metrics.reconstruction_loss_per_pixel:.6f} "
        f"kl/dim="
        f"{validation_metrics.kl_loss_per_dimension:.6f}"
        f"{improvement_marker}"
    )


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)

    seed_everything(
        seed=int(config["project"]["seed"]),
        deterministic=bool(
            config["device"]["deterministic"]
        ),
    )

    runtime = resolve_device(
        accelerator=str(
            config["device"]["accelerator"]
        ),
        mixed_precision=config["training"][
            "mixed_precision"
        ],
        pin_memory=config["data"]["pin_memory"],
    )

    metadata = prepare_metadata(config)

    datasets = build_datasets(
        config=config,
        metadata=metadata,
    )

    data_loaders = build_data_loaders(
        config=config,
        datasets=datasets,
        pin_memory=runtime.pin_memory,
    )

    print("=" * 72)
    print("HISTOLOGY VAE — TRAINING")
    print("=" * 72)
    print(f"Device: {runtime.device}")
    print(f"Mixed precision: {runtime.mixed_precision}")
    print()
    print(get_split_summary(metadata).to_string(index=False))
    print()

    model = build_vae_from_config(
        config=config,
        target_device=runtime.device,
    )

    optimizer = create_optimizer(
        model=model,
        config=config,
    )

    scaler = create_gradient_scaler(runtime)

    training_config = config["training"]
    early_stopping_config = (
        training_config["early_stopping"]
    )

    total_epochs = int(
        training_config["epochs"]
    )

    max_train_batches = training_config[
        "max_train_batches"
    ]

    max_validation_batches = training_config[
        "max_validation_batches"
    ]

    if max_train_batches is not None:
        max_train_batches = int(
            max_train_batches
        )

    if max_validation_batches is not None:
        max_validation_batches = int(
            max_validation_batches
        )

    if arguments.smoke_test:
        total_epochs = min(total_epochs, 2)
        max_train_batches = 2
        max_validation_batches = 1

        print(
            "Smoke-test mode enabled: "
            "2 epochs, 2 train batches, "
            "1 validation batch."
        )
        print()

    start_epoch = 0
    global_step = 0
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    if arguments.resume is None:
        run_directory = create_run_directory(config)

        shutil.copy2(
            arguments.config,
            run_directory / "config.yaml",
        )

    else:
        checkpoint_path = arguments.resume.resolve()

        run_directory = (
            checkpoint_path.parent.parent
        )

        checkpoint = load_training_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=runtime.device,
        )

        start_epoch = int(
            checkpoint["epoch"]
        ) + 1

        global_step = int(
            checkpoint["global_step"]
        )

        best_validation_loss = float(
            checkpoint["best_validation_loss"]
        )

        epochs_without_improvement = int(
            checkpoint.get(
                "epochs_without_improvement",
                0,
            )
        )

        history = list(
            checkpoint.get("history", [])
        )

        print(
            f"Resuming from epoch {start_epoch + 1}."
        )
        print()

    checkpoints_directory = (
        run_directory / "checkpoints"
    )

    reconstructions_directory = (
        run_directory / "reconstructions"
    )

    history_path = (
        run_directory / "history.csv"
    )

    writer = SummaryWriter(
        log_dir=str(
            run_directory / "tensorboard"
        ),
        purge_step=(
            global_step
            if arguments.resume is not None
            else None
        ),
    )

    maximum_beta = float(
        training_config["beta"]
    )

    warmup_epochs = int(
        training_config["kl_warmup_epochs"]
    )

    gradient_clip_norm = training_config[
        "gradient_clip_norm"
    ]

    if gradient_clip_norm is not None:
        gradient_clip_norm = float(
            gradient_clip_norm
        )

    reconstruction_type = str(
        training_config["reconstruction_loss"]
    )

    patience = int(
        early_stopping_config["patience"]
    )

    minimum_improvement = float(
        early_stopping_config["min_delta"]
    )

    early_stopping_enabled = bool(
        early_stopping_config["enabled"]
    )

    try:
        for epoch in range(
            start_epoch,
            total_epochs,
        ):
            train_beta = linear_kl_beta(
                current_epoch=epoch,
                warmup_epochs=warmup_epochs,
                maximum_beta=maximum_beta,
            )

            train_metrics, global_step = train_one_epoch(
                model=model,
                data_loader=data_loaders.train,
                optimizer=optimizer,
                runtime=runtime,
                beta=train_beta,
                reconstruction_type=reconstruction_type,
                gradient_clip_norm=gradient_clip_norm,
                scaler=scaler,
                writer=writer,
                global_step=global_step,
                log_every_n_steps=int(
                    training_config[
                        "log_every_n_steps"
                    ]
                ),
                max_batches=max_train_batches,
            )

            # Validation всегда использует maximum_beta.
            # Благодаря этому validation objective сравним
            # между разными эпохами KL warm-up.
            validation_metrics = validate_one_epoch(
                model=model,
                data_loader=data_loaders.validation,
                runtime=runtime,
                beta=maximum_beta,
                reconstruction_type=reconstruction_type,
                max_batches=max_validation_batches,
            )

            history_row = metrics_to_history_row(
                epoch=epoch,
                train_beta=train_beta,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
            )

            history.append(history_row)

            save_history(
                history=history,
                output_path=history_path,
            )

            log_epoch_metrics(
                writer=writer,
                epoch=epoch,
                train_beta=train_beta,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
            )

            preview_frequency = int(
                training_config[
                    "preview_every_n_epochs"
                ]
            )

            should_save_preview = (
                (epoch + 1) % preview_frequency == 0
                or epoch == total_epochs - 1
            )

            if should_save_preview:
                reconstruction_grid = (
                    create_reconstruction_grid(
                        model=model,
                        data_loader=(
                            data_loaders.validation
                        ),
                        runtime=runtime,
                        number_of_images=int(
                            training_config[
                                "preview_images"
                            ]
                        ),
                    )
                )

                preview_path = (
                    reconstructions_directory
                    / f"epoch_{epoch + 1:03d}.png"
                )

                save_reconstruction_grid(
                    grid=reconstruction_grid,
                    output_path=preview_path,
                )

                writer.add_image(
                    "reconstructions/"
                    "original_reconstruction_error",
                    reconstruction_grid,
                    epoch + 1,
                )

            improved = (
                validation_metrics.total_loss
                < best_validation_loss
                - minimum_improvement
            )

            if improved:
                best_validation_loss = (
                    validation_metrics.total_loss
                )

                epochs_without_improvement = 0

                save_training_checkpoint(
                    output_path=(
                        checkpoints_directory
                        / "best_checkpoint.pt"
                    ),
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    best_validation_loss=(
                        best_validation_loss
                    ),
                    epochs_without_improvement=(
                        epochs_without_improvement
                    ),
                    global_step=global_step,
                    history=history,
                    config=config,
                    scaler=scaler,
                )

                save_model_state(
                    output_path=(
                        checkpoints_directory
                        / "best_model_state.pt"
                    ),
                    model=model,
                )

            else:
                epochs_without_improvement += 1

            save_training_checkpoint(
                output_path=(
                    checkpoints_directory
                    / "last_checkpoint.pt"
                ),
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                best_validation_loss=(
                    best_validation_loss
                ),
                epochs_without_improvement=(
                    epochs_without_improvement
                ),
                global_step=global_step,
                history=history,
                config=config,
                scaler=scaler,
            )

            print_epoch_summary(
                epoch=epoch,
                total_epochs=total_epochs,
                train_beta=train_beta,
                train_metrics=train_metrics,
                validation_metrics=(
                    validation_metrics
                ),
                improved=improved,
            )

            writer.flush()

            if (
                early_stopping_enabled
                and epochs_without_improvement
                >= patience
            ):
                print()
                print(
                    "Early stopping: validation loss "
                    f"did not improve for {patience} epochs."
                )
                break

    finally:
        writer.close()

    print()
    print("=" * 72)
    print("Training completed.")
    print(f"Run directory: {run_directory.resolve()}")
    print(f"Best validation loss: {best_validation_loss:.6f}")
    print("=" * 72)


if __name__ == "__main__":
    main()