from __future__ import annotations

import argparse
from pathlib import Path

from src.reporting.llm_report import (
    generate_llm_report,
    load_metrics_sources,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a grounded, LLM-assisted markdown "
            "summary from one or more metrics.json files."
        )
    )

    parser.add_argument(
        "--metrics",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help=(
            "Repeatable. Example: "
            "--metrics rgb_vae=outputs/rgb_vae/probe/metrics.json "
            "--metrics grayscale_vae=outputs/grayscale_vae/probe/metrics.json"
        ),
    )

    parser.add_argument(
        "--research-question",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-6",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def _parse_metrics_arguments(
    raw_arguments: list[str],
) -> dict[str, str]:
    parsed: dict[str, str] = {}

    for raw_argument in raw_arguments:
        if "=" not in raw_argument:
            raise ValueError(
                "Each --metrics argument must have the "
                f"form LABEL=PATH. Received: {raw_argument!r}"
            )

        label, path = raw_argument.split("=", 1)
        label = label.strip()
        path = path.strip()

        if not label or not path:
            raise ValueError(
                "Both LABEL and PATH must be non-empty in "
                f"--metrics {raw_argument!r}"
            )

        parsed[label] = path

    return parsed


def main() -> None:
    arguments = parse_arguments()

    metrics_sources = _parse_metrics_arguments(
        arguments.metrics
    )

    metrics_by_label = load_metrics_sources(
        metrics_sources
    )

    report_text = generate_llm_report(
        metrics_by_label=metrics_by_label,
        research_question=(
            arguments.research_question
        ),
        model=arguments.model,
    )

    arguments.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    arguments.output.write_text(
        report_text,
        encoding="utf-8",
    )

    print(
        f"LLM-assisted report saved to: "
        f"{arguments.output.resolve()}"
    )
    print(
        "Reminder: this is a drafting aid, not a "
        "substitute for the statistical analysis in "
        "src/analysis (bootstrap CIs, linear probes, "
        "retrieval metrics)."
    )


if __name__ == "__main__":
    main()
