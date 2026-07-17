from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datasets.patch_dataset import HistologyPatchDataset
from src.models.pretrained_encoder import (
    DEFAULT_DINOV2_IMAGE_SIZE,
    build_dinov2_preprocessing_transform,
    build_pretrained_encoder,
    get_pretrained_encoder_info,
)
from src.utils.device import resolve_device
from src.utils.reproducibility import seed_everything

"""
Извлекает embeddings из замороженного self-supervised encoder
(по умолчанию DINOv2) для того же CRC metadata, которое
используется VAE-пайплайном этого проекта.

Результат — CSV в том же формате, что и src/evaluate.py
(колонки, начинающиеся с feature-prefix, plus метаданные), поэтому
его можно напрямую передать в:

    python -m src.run_linear_probe \\
        --train-embeddings TRAIN_CSV \\
        --validation-embeddings VALIDATION_CSV \\
        --test-embeddings TEST_CSV \\
        --feature-prefix dinov2_ \\
        --output-dir OUTPUT_DIRECTORY

Это позволяет напрямую сравнить DINOv2 с RGB VAE, grayscale VAE,
color-denoising VAE и RGB-HSV color baseline на одних и тех же
внешних метриках (balanced accuracy, macro-F1, retrieval).
"""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract frozen pretrained encoder "
            "embeddings (e.g. DINOv2) for CRC "
            "histology patches."
        )
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path("."),
    )

    parser.add_argument(
        "--split",
        choices=[
            "train",
            "validation",
            "test",
        ],
        required=True,
    )

    parser.add_argument(
        "--encoder-name",
        type=str,
        default="dinov2_vits14",
        help=(
            "One of: dinov2_vits14, dinov2_vitb14, "
            "dinov2_vitl14."
        ),
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_DINOV2_IMAGE_SIZE,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


@torch.inference_mode()
def extract_embeddings(
    encoder: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    feature_prefix: str,
) -> pd.DataFrame:
    embedding_batches: list[torch.Tensor] = []

    sample_ids: list[str] = []
    paths: list[str] = []
    patient_ids: list[str] = []
    slide_ids: list[str] = []
    patch_ids: list[str] = []
    class_codes: list[str] = []
    class_names: list[str] = []
    labels: list[int] = []

    for batch in data_loader:
        images = batch["image"].to(device)

        embeddings = encoder(images).float().cpu()
        embedding_batches.append(embeddings)

        batch_size = images.shape[0]

        optional_fields = (
            ("sample_id", sample_ids),
            ("class_code", class_codes),
            ("class_name", class_names),
        )

        for field_name, target_list in optional_fields:
            if field_name in batch:
                target_list.extend(
                    str(value)
                    for value in batch[field_name]
                )
            else:
                target_list.extend(
                    ["" for _ in range(batch_size)]
                )

        paths.extend(
            str(value) for value in batch["path"]
        )
        patient_ids.extend(
            str(value)
            for value in batch["patient_id"]
        )
        slide_ids.extend(
            str(value) for value in batch["slide_id"]
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

        if isinstance(batch_labels, torch.Tensor):
            labels.extend(
                int(value)
                for value in batch_labels.tolist()
            )
        else:
            labels.extend(
                int(value) for value in batch_labels
            )

    if not embedding_batches:
        raise RuntimeError(
            "Embedding extraction produced no batches."
        )

    embeddings_tensor = torch.cat(
        embedding_batches,
        dim=0,
    )

    feature_columns = [
        f"{feature_prefix}{dimension:03d}"
        for dimension in range(
            embeddings_tensor.shape[1]
        )
    ]

    feature_frame = pd.DataFrame(
        embeddings_tensor.numpy(),
        columns=feature_columns,
    )

    metadata_frame = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "path": paths,
            "patient_id": patient_ids,
            "slide_id": slide_ids,
            "patch_id": patch_ids,
            "class_code": class_codes,
            "class_name": class_names,
            "label": labels,
        }
    )

    return pd.concat(
        [
            metadata_frame.reset_index(drop=True),
            feature_frame.reset_index(drop=True),
        ],
        axis=1,
    )


def main() -> None:
    arguments = parse_arguments()

    seed_everything(
        seed=arguments.seed,
        deterministic=True,
    )

    runtime = resolve_device(
        accelerator=arguments.accelerator,
        mixed_precision=False,
        pin_memory="auto",
    )

    encoder_info = get_pretrained_encoder_info(
        arguments.encoder_name
    )

    print(
        f"Loading {encoder_info.name} "
        f"(embedding_dim={encoder_info.embedding_dim}) "
        "on "
        f"{runtime.device}..."
    )

    encoder = build_pretrained_encoder(
        encoder_name=arguments.encoder_name,
        device=runtime.device,
    )

    transform = (
        build_dinov2_preprocessing_transform(
            image_size=arguments.image_size,
        )
    )

    metadata = pd.read_csv(arguments.metadata)

    dataset = HistologyPatchDataset(
        metadata=metadata,
        split=arguments.split,
        transform=transform,
        root_dir=arguments.root_dir,
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=arguments.num_workers,
        drop_last=False,
    )

    embeddings = extract_embeddings(
        encoder=encoder,
        data_loader=data_loader,
        device=runtime.device,
        feature_prefix=(
            f"{arguments.encoder_name}_"
        ),
    )

    embeddings["split"] = arguments.split

    arguments.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        arguments.output_dir
        / f"{arguments.split}_embeddings.csv"
    )

    embeddings.to_csv(output_path, index=False)

    summary = {
        "encoder_name": arguments.encoder_name,
        "embedding_dim": (
            encoder_info.embedding_dim
        ),
        "image_size": arguments.image_size,
        "split": arguments.split,
        "number_of_images": int(len(embeddings)),
        "feature_prefix": (
            f"{arguments.encoder_name}_"
        ),
        "device": str(runtime.device),
        "pretrained_on": "ImageNet-1k (no histology fine-tuning)",
    }

    with (
        arguments.output_dir
        / f"{arguments.split}_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(
        f"Saved {len(embeddings)} embeddings to "
        f"{output_path.resolve()}"
    )


if __name__ == "__main__":
    main()
