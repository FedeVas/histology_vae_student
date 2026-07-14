from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RuntimeDevice:
    """
    Параметры вычислительного устройства для текущего запуска.
    """

    device: torch.device
    accelerator: str
    mixed_precision: bool
    pin_memory: bool

    def summary(self) -> str:
        return (
            f"device={self.device}, "
            f"mixed_precision={self.mixed_precision}, "
            f"pin_memory={self.pin_memory}"
        )


def is_mps_available() -> bool:
    """
    Проверяет доступность Apple Metal Performance Shaders.
    """
    mps_backend = getattr(torch.backends, "mps", None)

    return bool(
        mps_backend is not None
        and mps_backend.is_built()
        and mps_backend.is_available()
    )


def _resolve_auto_bool(
    value: bool | str,
    automatic_value: bool,
    option_name: str,
) -> bool:
    """
    Преобразует bool или строку 'auto' в итоговое логическое значение.
    """
    if isinstance(value, bool):
        return value

    normalized_value = str(value).strip().lower()

    if normalized_value == "auto":
        return automatic_value

    if normalized_value == "true":
        return True

    if normalized_value == "false":
        return False

    raise ValueError(
        f"{option_name} must be true, false or auto, got: {value!r}"
    )


def resolve_device(
    accelerator: str = "auto",
    mixed_precision: bool | str = "auto",
    pin_memory: bool | str = "auto",
) -> RuntimeDevice:
    """
    Определяет вычислительное устройство.

    При accelerator='auto' используется приоритет:

        CUDA -> MPS -> CPU

    Parameters
    ----------
    accelerator:
        auto, cpu, cuda или mps.
    mixed_precision:
        true, false или auto.
        В режиме auto включается только на CUDA.
    pin_memory:
        true, false или auto.
        В режиме auto включается только на CUDA.
    """
    accelerator = accelerator.strip().lower()

    supported_accelerators = {"auto", "cpu", "cuda", "mps"}

    if accelerator not in supported_accelerators:
        raise ValueError(
            f"Unknown accelerator {accelerator!r}. "
            f"Available values: {sorted(supported_accelerators)}"
        )

    cuda_available = torch.cuda.is_available()
    mps_available = is_mps_available()

    if accelerator == "auto":
        if cuda_available:
            selected_accelerator = "cuda"
        elif mps_available:
            selected_accelerator = "mps"
        else:
            selected_accelerator = "cpu"

    elif accelerator == "cuda":
        if not cuda_available:
            raise RuntimeError(
                "CUDA was explicitly requested, but CUDA is not available. "
                "Use accelerator='cpu' or accelerator='auto'."
            )
        selected_accelerator = "cuda"

    elif accelerator == "mps":
        if not mps_available:
            raise RuntimeError(
                "MPS was explicitly requested, but MPS is not available. "
                "Use accelerator='cpu' or accelerator='auto'."
            )
        selected_accelerator = "mps"

    else:
        selected_accelerator = "cpu"

    device = torch.device(selected_accelerator)

    use_mixed_precision = _resolve_auto_bool(
        value=mixed_precision,
        automatic_value=device.type == "cuda",
        option_name="mixed_precision",
    )

    use_pin_memory = _resolve_auto_bool(
        value=pin_memory,
        automatic_value=device.type == "cuda",
        option_name="pin_memory",
    )

    if use_mixed_precision and device.type != "cuda":
        raise ValueError(
            "In the first project implementation, mixed precision is "
            "supported only for CUDA."
        )

    return RuntimeDevice(
        device=device,
        accelerator=selected_accelerator,
        mixed_precision=use_mixed_precision,
        pin_memory=use_pin_memory,
    )