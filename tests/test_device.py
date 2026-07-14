import torch

from src.utils.device import resolve_device


def test_explicit_cpu_selection() -> None:
    runtime = resolve_device(
        accelerator="cpu",
        mixed_precision="auto",
        pin_memory="auto",
    )

    assert runtime.device == torch.device("cpu")
    assert runtime.accelerator == "cpu"
    assert runtime.mixed_precision is False
    assert runtime.pin_memory is False


def test_auto_returns_supported_device() -> None:
    runtime = resolve_device(accelerator="auto")

    assert runtime.device.type in {"cpu", "cuda", "mps"}


def test_invalid_accelerator_raises_error() -> None:
    try:
        resolve_device(accelerator="quantum_gpu")
    except ValueError:
        return

    raise AssertionError("Expected ValueError for an invalid accelerator.")