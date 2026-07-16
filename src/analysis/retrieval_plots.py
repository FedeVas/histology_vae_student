from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def save_retrieval_montages(
    neighbors: pd.DataFrame,
    output_directory: str | Path,
    neighbors_per_query: int = 5,
    queries_per_class: int = 2,
    seed: int = 42,
) -> list[Path]:
    """
    Сохраняет отдельный montage для каждого tissue class.

    В каждой строке:
        query external-test image
        + его ближайшие train neighbors.
    """
    if neighbors_per_query <= 0:
        raise ValueError(
            "neighbors_per_query must be positive."
        )

    if queries_per_class <= 0:
        raise ValueError(
            "queries_per_class must be positive."
        )

    required_columns = {
        "query_index",
        "query_path",
        "query_class_code",
        "neighbor_rank",
        "neighbor_path",
        "neighbor_class_code",
        "distance",
        "same_class",
    }

    missing_columns = (
        required_columns
        .difference(neighbors.columns)
    )

    if missing_columns:
        raise ValueError(
            "Neighbors frame is missing columns: "
            f"{sorted(missing_columns)}"
        )

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    query_table = (
        neighbors[
            [
                "query_index",
                "query_path",
                "query_class_code",
            ]
        ]
        .drop_duplicates(
            subset=["query_index"]
        )
        .reset_index(drop=True)
    )

    saved_paths: list[Path] = []

    grouped_classes = query_table.groupby(
        "query_class_code",
        sort=True,
    )

    for class_index, (
        class_code,
        class_queries,
    ) in enumerate(grouped_classes):
        number_to_sample = min(
            queries_per_class,
            len(class_queries),
        )

        selected_queries = (
            class_queries.sample(
                n=number_to_sample,
                random_state=(
                    seed + class_index
                ),
                replace=False,
            )
            .sort_values("query_index")
            .reset_index(drop=True)
        )

        figure, axes = plt.subplots(
            nrows=number_to_sample,
            ncols=neighbors_per_query + 1,
            figsize=(
                (neighbors_per_query + 1) * 2.5,
                number_to_sample * 2.7,
            ),
            squeeze=False,
        )

        for row_index, query_row in (
            selected_queries.iterrows()
        ):
            query_index = int(
                query_row["query_index"]
            )

            query_axis = axes[
                row_index,
                0,
            ]

            with Image.open(
                query_row["query_path"]
            ) as image:
                query_axis.imshow(
                    image.convert("RGB")
                )

            query_axis.set_title(
                f"Query\n{class_code}",
                fontsize=9,
            )

            query_axis.axis("off")

            query_neighbors = (
                neighbors[
                    neighbors["query_index"]
                    == query_index
                ]
                .sort_values("neighbor_rank")
                .head(neighbors_per_query)
                .reset_index(drop=True)
            )

            for neighbor_offset in range(
                neighbors_per_query
            ):
                axis = axes[
                    row_index,
                    neighbor_offset + 1,
                ]

                axis.axis("off")

                if (
                    neighbor_offset
                    >= len(query_neighbors)
                ):
                    continue

                neighbor_row = (
                    query_neighbors.iloc[
                        neighbor_offset
                    ]
                )

                with Image.open(
                    neighbor_row[
                        "neighbor_path"
                    ]
                ) as image:
                    axis.imshow(
                        image.convert("RGB")
                    )

                match_symbol = (
                    "✓"
                    if bool(
                        neighbor_row[
                            "same_class"
                        ]
                    )
                    else "✗"
                )

                axis.set_title(
                    (
                        f"#{int(neighbor_row['neighbor_rank'])} "
                        f"{neighbor_row['neighbor_class_code']} "
                        f"{match_symbol}\n"
                        f"d={float(neighbor_row['distance']):.3f}"
                    ),
                    fontsize=8,
                )

        figure.suptitle(
            (
                "External queries and nearest "
                f"train neighbors — {class_code}"
            ),
            fontsize=12,
        )

        figure.tight_layout()

        safe_class_code = re.sub(
            r"[^A-Za-z0-9_-]+",
            "_",
            str(class_code),
        )

        output_path = (
            output_directory
            / (
                f"retrieval_"
                f"{safe_class_code}.png"
            )
        )

        figure.savefig(
            output_path,
            dpi=170,
            bbox_inches="tight",
        )

        plt.close(figure)

        saved_paths.append(
            output_path
        )

    return saved_paths