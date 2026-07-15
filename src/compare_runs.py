from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare trained representation models."
        )
    )

    parser.add_argument(
        "--run-dirs",
        type=Path,
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/comparisons/"
            "synthetic_baselines"
        ),
    )

    return parser.parse_args()


def load_run_summary(
    run_directory: Path,
) -> dict:
    config_path = (
        run_directory / "config.yaml"
    )

    history_path = (
        run_directory / "history.csv"
    )

    metrics_path = (
        run_directory
        / "evaluation"
        / "test"
        / "metrics.json"
    )

    for required_path in (
        config_path,
        history_path,
        metrics_path,
    ):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required file not found: "
                f"{required_path.resolve()}"
            )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    with metrics_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metrics = json.load(file)

    history = pd.read_csv(history_path)

    best_row_index = history[
        "validation_total_loss"
    ].idxmin()

    best_row = history.loc[best_row_index]

    model_type = str(
        config["model"].get("type", "vae")
    )

    beta = float(
        config["training"]["beta"]
    )

    if model_type == "autoencoder":
        display_name = "Autoencoder"
    elif beta == 1.0:
        display_name = "VAE beta=1"
    else:
        display_name = f"Beta-VAE beta={beta:g}"

    reconstruction = metrics[
        "reconstruction"
    ]

    latent = metrics["latent"]

    return {
        "run_name": run_directory.name,
        "model": display_name,
        "model_type": model_type,
        "beta": beta,
        "latent_dim": int(
            config["model"]["latent_dim"]
        ),
        "best_epoch": int(
            best_row["epoch"]
        ),
        "best_validation_total_loss": float(
            best_row[
                "validation_total_loss"
            ]
        ),
        "test_mse": float(
            reconstruction["mse"]
        ),
        "test_mae": float(
            reconstruction["mae"]
        ),
        "test_psnr": float(
            reconstruction["psnr"]
        ),
        "test_ssim": float(
            reconstruction["ssim"]
        ),
        "active_units": int(
            latent["number_of_active_units"]
        ),
        "active_fraction": float(
            latent["active_fraction"]
        ),
        "mean_kl_per_dimension": (
            latent.get(
                "mean_kl_per_dimension"
            )
        ),
    }


def save_bar_plot(
    comparison: pd.DataFrame,
    value_column: str,
    title: str,
    y_label: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    axis.bar(
        comparison["model"],
        comparison[value_column],
    )

    axis.set_title(title)
    axis.set_ylabel(y_label)
    axis.tick_params(
        axis="x",
        rotation=15,
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    arguments = parse_arguments()

    summaries = [
        load_run_summary(run_directory)
        for run_directory
        in arguments.run_dirs
    ]

    comparison = pd.DataFrame(
        summaries
    )

    arguments.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        arguments.output_dir
        / "comparison.csv",
        index=False,
    )

    save_bar_plot(
        comparison=comparison,
        value_column="test_mse",
        title=(
            "Test reconstruction MSE"
        ),
        y_label="MSE",
        output_path=(
            arguments.output_dir
            / "comparison_mse.png"
        ),
    )

    save_bar_plot(
        comparison=comparison,
        value_column="test_ssim",
        title=(
            "Test reconstruction SSIM"
        ),
        y_label="SSIM",
        output_path=(
            arguments.output_dir
            / "comparison_ssim.png"
        ),
    )

    save_bar_plot(
        comparison=comparison,
        value_column="active_fraction",
        title=(
            "Fraction of active latent units"
        ),
        y_label="Active fraction",
        output_path=(
            arguments.output_dir
            / "comparison_active_units.png"
        ),
    )

    print()
    print("Model comparison")
    print(
        comparison[
            [
                "model",
                "test_mse",
                "test_psnr",
                "test_ssim",
                "active_units",
                "active_fraction",
                "mean_kl_per_dimension",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        "Results: "
        f"{arguments.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()