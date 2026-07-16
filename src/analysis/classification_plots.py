from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_confusion_matrix_plot(
    matrix: np.ndarray,
    class_codes: list[str],
    output_path: str | Path,
    title: str,
    normalized: bool,
) -> None:
    if matrix.shape != (
        len(class_codes),
        len(class_codes),
    ):
        raise ValueError(
            "Confusion matrix shape does not match "
            "the number of class codes."
        )

    figure, axis = plt.subplots(
        figsize=(9, 8)
    )

    image = axis.imshow(matrix)

    figure.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
    )

    positions = np.arange(
        len(class_codes)
    )

    axis.set_xticks(positions)
    axis.set_yticks(positions)

    axis.set_xticklabels(
        class_codes,
        rotation=45,
        ha="right",
    )

    axis.set_yticklabels(class_codes)

    axis.set_xlabel("Predicted tissue class")
    axis.set_ylabel("True tissue class")
    axis.set_title(title)

    for row_index in range(
        matrix.shape[0]
    ):
        for column_index in range(
            matrix.shape[1]
        ):
            value = matrix[
                row_index,
                column_index,
            ]

            display_value = (
                f"{value:.2f}"
                if normalized
                else str(int(value))
            )

            axis.text(
                column_index,
                row_index,
                display_value,
                ha="center",
                va="center",
                fontsize=8,
            )

    figure.tight_layout()

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=170,
        bbox_inches="tight",
    )

    plt.close(figure)