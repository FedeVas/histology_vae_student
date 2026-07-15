from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from src.analysis.evaluation import evaluate_representation_model
from src.analysis.latent import (
    build_latent_statistics_frame,
    compute_latent_diagnostics,
)
from src.analysis.plots import (
    create_pca_embedding_plot,
)
from src.datasets.factory import (
    build_data_loaders,
    build_datasets,
    prepare_metadata,
)
from src.models.factory import (
    build_model_from_config,
    get_model_type,
)
from src.training.visualization import (
    create_reconstruction_grid,
    save_reconstruction_grid,
)
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.reproducibility import seed_everything


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained histology VAE."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--split",
        choices=[
            "train",
            "validation",
            "test",
        ],
        default="test",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
    )

    return parser.parse_args()


def load_model_weights(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint was not found: "
            f"{checkpoint_path.resolve()}"
        )

    loaded_object: Any = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    if (
        isinstance(loaded_object, dict)
        and "model_state_dict" in loaded_object
    ):
        state_dict = loaded_object[
            "model_state_dict"
        ]

    elif isinstance(loaded_object, dict):
        state_dict = loaded_object

    else:
        raise TypeError(
            "Checkpoint must contain a state_dict "
            "or a training checkpoint dictionary."
        )

    model.load_state_dict(state_dict)


def resolve_output_directory(
    checkpoint_path: Path,
    split_name: str,
    explicit_output_directory: Path | None,
) -> Path:
    if explicit_output_directory is not None:
        return explicit_output_directory

    # .../run/checkpoints/best_checkpoint.pt
    run_directory = (
        checkpoint_path.resolve()
        .parent
        .parent
    )

    return (
        run_directory
        / "evaluation"
        / split_name
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
        mixed_precision=False,
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

    selected_loader = getattr(
        data_loaders,
        arguments.split,
    )

    model = build_model_from_config(
        config=config,
        target_device=runtime.device,
    )

    load_model_weights(
        model=model,
        checkpoint_path=arguments.checkpoint,
        device=runtime.device,
    )

    evaluation_result = evaluate_representation_model(
        model=model,
        data_loader=selected_loader,
        runtime=runtime,
        split_name=arguments.split,
        max_batches=arguments.max_batches,
    )

    evaluation_config = config.get(
        "evaluation",
        {},
    )

    diagnostics = compute_latent_diagnostics(
        latent_vectors=evaluation_result.latent_vectors,
        log_var=evaluation_result.log_var,        
        active_unit_variance_threshold=float(
            evaluation_config.get(
                "active_unit_variance_threshold",
                0.001,
            )
        ),
        low_kl_threshold=float(
            evaluation_config.get(
                "low_kl_threshold",
                0.001,
            )
        ),
    )

    output_directory = resolve_output_directory(
        checkpoint_path=arguments.checkpoint,
        split_name=arguments.split,
        explicit_output_directory=(
            arguments.output_dir
        ),
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluation_result.embeddings.to_csv(
        output_directory / "embeddings.csv",
        index=False,
    )

    latent_statistics = (
        build_latent_statistics_frame(
            diagnostics
        )
    )

    latent_statistics.to_csv(
        output_directory
        / "latent_statistics.csv",
        index=False,
    )

    pca_coordinates, explained_variance = (
        create_pca_embedding_plot(
            embeddings=(
                evaluation_result.embeddings
            ),
            output_path=(
                output_directory
                / "pca_embedding.png"
            ),
            seed=int(
                config["project"]["seed"]
            ),
        )
    )

    pca_coordinates.to_csv(
        output_directory
        / "pca_coordinates.csv",
        index=False,
    )

    reconstruction_grid = (
        create_reconstruction_grid(
            model=model,
            data_loader=selected_loader,
            runtime=runtime,
            number_of_images=int(
                evaluation_config.get(
                    "preview_images",
                    8,
                )
            ),
        )
    )

    save_reconstruction_grid(
        grid=reconstruction_grid,
        output_path=(
            output_directory
            / "reconstruction_preview.png"
        ),
    )

    metrics_summary = {
        "model" :{
            "type": get_model_type(config),
            "beta": float(config["training"]["beta"]),
            "latent_dim": int(config["model"]["latent_dim"])
        },
        "split": arguments.split,
        "checkpoint": str(
            arguments.checkpoint.resolve()
        ),
        "reconstruction": (
            evaluation_result
            .reconstruction_metrics
            .to_dict()
        ),
        "latent": diagnostics.summary(),
        "pca": {
            "explained_variance_ratio_pc1": (
                float(explained_variance[0])
            ),
            "explained_variance_ratio_pc2": (
                float(explained_variance[1])
            ),
        },
    }

    with (
        output_directory
        / "metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics_summary,
            file,
            indent=2,
        )

    reconstruction_metrics = (
        evaluation_result.reconstruction_metrics
    )

    print("=" * 68)
    print("HISTOLOGY VAE — EVALUATION")
    print("=" * 68)

    print(f"Device:              {runtime.device}")
    print(f"Split:               {arguments.split}")
    print(
        f"Number of images:    "
        f"{reconstruction_metrics.number_of_images}"
    )

    print()
    print("Reconstruction metrics")
    print(
        f"MSE:                 "
        f"{reconstruction_metrics.mse:.6f}"
    )
    print(
        f"MAE:                 "
        f"{reconstruction_metrics.mae:.6f}"
    )
    print(
        f"PSNR:                "
        f"{reconstruction_metrics.psnr:.4f}"
    )
    print(
        f"SSIM:                "
        f"{reconstruction_metrics.ssim:.4f}"
    )

    print()
    print("Latent diagnostics")
    print(
        f"Latent dimensions:   "
        f"{diagnostics.number_of_dimensions}"
    )
    print(
        f"Active units:        "
        f"{diagnostics.number_of_active_units}"
    )
    print(
        f"Active fraction:     "
        f"{diagnostics.active_fraction:.2%}"
    )
    print()
    print(
        f"Results:             "
        f"{output_directory.resolve()}"
    )

    if diagnostics.number_of_active_units == 0:
        print()
        print(
            "WARNING: no active latent units were detected. "
            "This may indicate posterior collapse."
        )
    if diagnostics.mean_kl_per_dimension is None:
        print("Mean KL/dimension:   not applicable")
    else:
        print(
            f"Mean KL/dimension:   "
            f"{diagnostics.mean_kl_per_dimension:.6f}"
        )
    if diagnostics.number_of_low_kl_dimensions is None:
        print("Low-KL dimensions:   not applicable")
    else:
        print(
            f"Low-KL dimensions:   "
            f"{diagnostics.number_of_low_kl_dimensions}"
        )
    print("=" * 68)


if __name__ == "__main__":
    main()