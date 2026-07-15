from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_training_checkpoint(
    output_path: str | Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    best_validation_loss: float,
    epochs_without_improvement: int,
    global_step: int,
    history: list[dict[str, Any]],
    config: dict,
    scaler: Any | None = None,
) -> None:
    """
    Сохраняет checkpoint, достаточный для продолжения обучения.
    """
    checkpoint = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_validation_loss": float(
            best_validation_loss
        ),
        "epochs_without_improvement": int(
            epochs_without_improvement
        ),
        "global_step": int(global_step),
        "history": history,
        "config": config,
        "scaler_state_dict": (
            scaler.state_dict()
            if scaler is not None
            else None
        ),
    }

    _atomic_torch_save(
        object_to_save=checkpoint,
        output_path=output_path,
    )


def save_model_state(
    output_path: str | Path,
    model: torch.nn.Module,
) -> None:
    """
    Сохраняет только model.state_dict для inference.
    """
    _atomic_torch_save(
        object_to_save=model.state_dict(),
        output_path=output_path,
    )


def load_training_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """
    Загружает training checkpoint.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint was not found: "
            f"{checkpoint_path.resolve()}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    required_keys = {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "best_validation_loss",
        "global_step",
    }

    missing_keys = required_keys.difference(
        checkpoint.keys()
    )

    if missing_keys:
        raise ValueError(
            "Checkpoint is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    scaler_state = checkpoint.get(
        "scaler_state_dict"
    )

    if (
        scaler is not None
        and scaler_state is not None
    ):
        scaler.load_state_dict(scaler_state)

    return checkpoint


def _atomic_torch_save(
    object_to_save: Any,
    output_path: str | Path,
) -> None:
    """
    Сначала сохраняет временный файл, затем заменяет итоговый.

    Это снижает вероятность оставить повреждённый checkpoint,
    если сохранение было прервано.
    """
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    torch.save(
        object_to_save,
        temporary_path,
    )

    temporary_path.replace(output_path)