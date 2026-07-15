from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def append_history_row(
    history_path: str | Path,
    row: dict[str, Any],
) -> None:
    """
    Добавляет одну эпоху в CSV.
    """
    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = history_path.exists()

    with history_path.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(row.keys()),
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def save_json(
    output_path: str | Path,
    data: dict[str, Any],
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )