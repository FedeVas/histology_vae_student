from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.analysis.reconstruction_metrics import (
    ReconstructionMetricAccumulator,
    ReconstructionMetrics,
)
from src.utils.device import RuntimeDevice


@dataclass(frozen=True)
class RepresentationEvaluationResult:
    reconstruction_metrics: ReconstructionMetrics
    embeddings: pd.DataFrame

    latent_vectors: torch.Tensor
    log_var: torch.Tensor | None


@torch.inference_mode()
def evaluate_representation_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    runtime: RuntimeDevice,
    split_name: str,
    max_batches: int | None = None,
) -> RepresentationEvaluationResult:
    model.eval()

    metric_accumulator = (
        ReconstructionMetricAccumulator()
    )

    latent_batches: list[torch.Tensor] = []
    log_var_batches: list[torch.Tensor] = []

    paths: list[str] = []
    patient_ids: list[str] = []
    slide_ids: list[str] = []
    patch_ids: list[str] = []
    labels: list[int] = []

    has_log_var: bool | None = None

    for batch_index, batch in enumerate(
        data_loader
    ):
        if (
            max_batches is not None
            and batch_index >= max_batches
        ):
            break

        images = batch["image"].to(
            runtime.device,
            non_blocking=runtime.pin_memory,
        )

        output = model(
            images,
            sample_posterior=False,
        )

        metric_accumulator.update(
            target=images,
            reconstruction=(
                output.reconstruction
            ),
        )

        latent_batches.append(
            output.embedding
            .detach()
            .float()
            .cpu()
        )

        current_has_log_var = (
            output.log_var is not None
        )

        if has_log_var is None:
            has_log_var = current_has_log_var

        elif has_log_var != current_has_log_var:
            raise RuntimeError(
                "Model output inconsistently contains "
                "log_var across batches."
            )

        if output.log_var is not None:
            log_var_batches.append(
                output.log_var
                .detach()
                .float()
                .cpu()
            )

        batch_size = images.shape[0]

        paths.extend(
            str(value)
            for value in batch["path"]
        )

        patient_ids.extend(
            str(value)
            for value in batch["patient_id"]
        )

        slide_ids.extend(
            str(value)
            for value in batch["slide_id"]
        )

        if "patch_id" in batch:
            patch_ids.extend(
                str(value)
                for value in batch["patch_id"]
            )
        else:
            patch_ids.extend(
                ["" for _ in range(batch_size)]
            )

        batch_labels = batch["label"]

        if isinstance(
            batch_labels,
            torch.Tensor,
        ):
            labels.extend(
                int(value)
                for value
                in batch_labels.tolist()
            )
        else:
            labels.extend(
                int(value)
                for value in batch_labels
            )

    if not latent_batches:
        raise RuntimeError(
            "Evaluation produced no batches."
        )

    latent_vectors = torch.cat(
        latent_batches,
        dim=0,
    )

    if log_var_batches:
        log_var = torch.cat(
            log_var_batches,
            dim=0,
        )
    else:
        log_var = None

    metadata_frame = pd.DataFrame(
        {
            "path": paths,
            "patient_id": patient_ids,
            "slide_id": slide_ids,
            "patch_id": patch_ids,
            "label": labels,
            "split": split_name,
        }
    )

    latent_columns = [
        f"latent_{dimension:03d}"
        for dimension in range(
            latent_vectors.shape[1]
        )
    ]

    latent_frame = pd.DataFrame(
        latent_vectors.numpy(),
        columns=latent_columns,
    )

    embeddings = pd.concat(
        [
            metadata_frame.reset_index(
                drop=True
            ),
            latent_frame.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    return RepresentationEvaluationResult(
        reconstruction_metrics=(
            metric_accumulator.compute()
        ),
        embeddings=embeddings,
        latent_vectors=latent_vectors,
        log_var=log_var,
    )


# Совместимость с предыдущим кодом.
VAEEvaluationResult = (
    RepresentationEvaluationResult
)
evaluate_vae = evaluate_representation_model