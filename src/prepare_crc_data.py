from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.data_audit import (
    run_crc_data_audit,
)
from src.datasets.crc import build_crc_metadata
from src.utils.config import load_config


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and audit NCT-CRC-HE datasets."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/crc_data_pilot.yaml"
        ),
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    config = load_config(arguments.config)

    project_config = config["project"]
    dataset_config = config["dataset"]
    pilot_config = dataset_config["pilot"]
    audit_config = dataset_config["audit"]

    if bool(pilot_config["enabled"]):
        train_pool_per_class = int(
            pilot_config[
                "train_pool_per_class"
            ]
        )

        external_test_per_class = int(
            pilot_config[
                "external_test_per_class"
            ]
        )
    else:
        train_pool_per_class = None
        external_test_per_class = None

    metadata = build_crc_metadata(
        train_root=dataset_config[
            "train_root"
        ],
        external_test_root=dataset_config[
            "external_test_root"
        ],
        validation_fraction=float(
            dataset_config[
                "internal_validation_fraction"
            ]
        ),
        seed=int(project_config["seed"]),
        train_pool_per_class=(
            train_pool_per_class
        ),
        external_test_per_class=(
            external_test_per_class
        ),
    )

    metadata_path = Path(
        dataset_config["metadata_csv"]
    )

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata.to_csv(
        metadata_path,
        index=False,
    )

    run_crc_data_audit(
        metadata=metadata,
        output_directory=dataset_config[
            "audit_dir"
        ],
        seed=int(project_config["seed"]),
        verify_images=bool(
            audit_config["verify_images"]
        ),
        verify_max_per_class_and_source=(
            audit_config[
                "verify_max_per_class_and_source"
            ]
        ),
        color_statistics_per_class_and_source=int(
            audit_config[
                "color_statistics_per_class_and_source"
            ]
        ),
        montage_images_per_class=int(
            audit_config[
                "montage_images_per_class"
            ]
        ),
    )

    print("=" * 68)
    print("CRC DATA PREPARATION")
    print("=" * 68)

    print(
        metadata.groupby(
            [
                "source",
                "split",
                "class_code",
            ]
        )
        .size()
        .rename("images")
        .to_string()
    )

    print()
    print(
        f"Metadata: "
        f"{metadata_path.resolve()}"
    )

    print(
        f"Audit:    "
        f"{Path(dataset_config['audit_dir']).resolve()}"
    )

    print()
    print(
        "Important: internal validation is patch-level; "
        "CRC-VAL-HE-7K is the external patient-disjoint test set."
    )

    print("=" * 68)


if __name__ == "__main__":
    main()