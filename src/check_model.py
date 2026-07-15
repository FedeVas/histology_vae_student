from __future__ import annotations

from pathlib import Path

import torch

from src.models.losses import (
    compute_vae_loss,
    linear_kl_beta,
)
from src.models.vae import ConvolutionalVAE
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.reproducibility import seed_everything


CONFIG_PATH = Path("configs/vae_base.yaml")


def count_trainable_parameters(
    model: torch.nn.Module,
) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def calculate_gradient_norm(
    model: torch.nn.Module,
) -> float:
    squared_gradient_sum = torch.zeros(
        (),
        device=next(model.parameters()).device,
    )

    number_of_gradients = 0

    for parameter in model.parameters():
        if parameter.grad is None:
            continue

        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError(
                "Model contains non-finite gradients."
            )

        squared_gradient_sum += (
            parameter.grad.detach().pow(2).sum()
        )

        number_of_gradients += 1

    if number_of_gradients == 0:
        raise RuntimeError(
            "No gradients were calculated."
        )

    return squared_gradient_sum.sqrt().item()


def main() -> None:
    config = load_config(CONFIG_PATH)

    seed = int(config["project"]["seed"])

    seed_everything(
        seed=seed,
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

    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]

    model = ConvolutionalVAE(
        image_channels=int(data_config["channels"]),
        image_size=int(data_config["image_size"]),
        hidden_channels=[
            int(channel)
            for channel in model_config["hidden_channels"]
        ],
        latent_dim=int(model_config["latent_dim"]),
        log_var_min=float(
            model_config["log_var_min"]
        ),
        log_var_max=float(
            model_config["log_var_max"]
        ),
    ).to(runtime.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(
            training_config["weight_decay"]
        ),
    )

    batch_size = 4

    images = torch.rand(
        batch_size,
        int(data_config["channels"]),
        int(data_config["image_size"]),
        int(data_config["image_size"]),
        device=runtime.device,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    output = model(images)

    beta = linear_kl_beta(
        current_epoch=0,
        warmup_epochs=int(
            training_config["kl_warmup_epochs"]
        ),
        maximum_beta=float(
            training_config["beta"]
        ),
    )

    loss_output = compute_vae_loss(
        reconstruction=output.reconstruction,
        target=images,
        mu=output.mu,
        log_var=output.log_var,
        beta=beta,
        reconstruction_type=str(
            training_config["reconstruction_loss"]
        ),
    )

    loss_output.total_loss.backward()

    gradient_norm = calculate_gradient_norm(model)

    optimizer.step()

    if not torch.isfinite(
        loss_output.total_loss
    ):
        raise RuntimeError(
            "Total loss is not finite."
        )

    if output.reconstruction.min().item() < 0.0:
        raise RuntimeError(
            "Reconstruction contains values below zero."
        )

    if output.reconstruction.max().item() > 1.0:
        raise RuntimeError(
            "Reconstruction contains values above one."
        )

    model.eval()

    with torch.no_grad():
        deterministic_output_1 = model(
            images,
            sample_posterior=False,
        )

        deterministic_output_2 = model(
            images,
            sample_posterior=False,
        )

    if not torch.equal(
        deterministic_output_1.reconstruction,
        deterministic_output_2.reconstruction,
    ):
        raise RuntimeError(
            "Deterministic evaluation produced different outputs."
        )

    print("=" * 64)
    print("HISTOLOGY VAE — MODEL CHECK")
    print("=" * 64)

    print(f"Device:                      {runtime.device}")
    print(
        "Trainable parameters:        "
        f"{count_trainable_parameters(model):,}"
    )

    print()
    print("Tensor shapes")
    print(f"Input:                       {tuple(images.shape)}")
    print(f"Mu:                          {tuple(output.mu.shape)}")
    print(
        f"Log variance:                "
        f"{tuple(output.log_var.shape)}"
    )
    print(f"Latent sample:               {tuple(output.z.shape)}")
    print(
        f"Reconstruction:              "
        f"{tuple(output.reconstruction.shape)}"
    )

    print()
    print("Loss components")
    print(
        f"Beta:                        "
        f"{loss_output.beta:.6f}"
    )
    print(
        f"Total loss:                  "
        f"{loss_output.total_loss.item():.6f}"
    )
    print(
        f"Reconstruction loss:         "
        f"{loss_output.reconstruction_loss.item():.6f}"
    )
    print(
        f"Reconstruction loss/pixel:   "
        f"{loss_output.reconstruction_loss_per_pixel.item():.6f}"
    )
    print(
        f"KL loss:                     "
        f"{loss_output.kl_loss.item():.6f}"
    )
    print(
        f"KL loss/dimension:           "
        f"{loss_output.kl_loss_per_dimension.item():.6f}"
    )
    print(
        f"Gradient norm:               "
        f"{gradient_norm:.6f}"
    )

    print()
    print("Checks")
    print("Forward pass:                OK")
    print("VAE reparameterization:      OK")
    print("Loss calculation:            OK")
    print("Backward pass:               OK")
    print("Optimizer step:              OK")
    print("Deterministic evaluation:    OK")

    print("=" * 64)
    print("Model check completed successfully.")
    print("=" * 64)


if __name__ == "__main__":
    main()