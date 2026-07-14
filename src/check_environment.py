import platform
import sys
from pathlib import Path

import torch
from torch import nn

from src.utils.config import load_config
from src.utils.device import RuntimeDevice, resolve_device
from src.utils.reproducibility import seed_everything


CONFIG_PATH = Path("configs/vae_base.yaml")


def print_runtime_information(runtime: RuntimeDevice) -> None:
    """
    Печатает информацию о Python, PyTorch и выбранном устройстве.
    """
    print("=" * 60)
    print("HISTOLOGY VAE — ENVIRONMENT CHECK")
    print("=" * 60)

    print(f"Python:          {sys.version.split()[0]}")
    print(f"Platform:        {platform.platform()}")
    print(f"PyTorch:         {torch.__version__}")
    print(f"CUDA available:  {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version:    {torch.version.cuda}")
        print(f"GPU count:       {torch.cuda.device_count()}")
        print(f"GPU name:        {torch.cuda.get_device_name(0)}")

    print(f"Selected device: {runtime.device}")
    print(f"Mixed precision: {runtime.mixed_precision}")
    print(f"Pin memory:      {runtime.pin_memory}")
    print("=" * 60)


def run_tensor_smoke_test(
    runtime: RuntimeDevice,
    channels: int,
    image_size: int,
) -> None:
    """
    Проверяет:
    1. создание tensor;
    2. перенос модели на устройство;
    3. forward pass;
    4. вычисление loss;
    5. backward pass;
    6. optimizer step.
    """
    batch_size = 4

    test_batch = torch.randn(
        batch_size,
        channels,
        image_size,
        image_size,
        device=runtime.device,
    )

    model = nn.Sequential(
        nn.Conv2d(
            in_channels=channels,
            out_channels=8,
            kernel_size=3,
            padding=1,
        ),
        nn.SiLU(),
        nn.AdaptiveAvgPool2d(output_size=1),
        nn.Flatten(),
        nn.Linear(8, 1),
    ).to(runtime.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    optimizer.zero_grad(set_to_none=True)

    predictions = model(test_batch)
    loss = predictions.square().mean()

    loss.backward()
    optimizer.step()

    print("Tensor smoke test")
    print(f"Input shape:     {tuple(test_batch.shape)}")
    print(f"Output shape:    {tuple(predictions.shape)}")
    print(f"Loss:            {loss.item():.6f}")
    print("Forward pass:    OK")
    print("Backward pass:   OK")
    print("Optimizer step:  OK")


def main() -> None:
    config = load_config(CONFIG_PATH)

    seed_everything(
        seed=int(config["project"]["seed"]),
        deterministic=bool(config["device"]["deterministic"]),
    )

    runtime = resolve_device(
        accelerator=str(config["device"]["accelerator"]),
        mixed_precision=config["training"]["mixed_precision"],
        pin_memory=config["data"]["pin_memory"],
    )

    print_runtime_information(runtime)

    run_tensor_smoke_test(
        runtime=runtime,
        channels=int(config["data"]["channels"]),
        image_size=int(config["data"]["image_size"]),
    )

    print("=" * 60)
    print("Environment check completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()