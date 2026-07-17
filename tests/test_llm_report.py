import json
from pathlib import Path

import pytest

from src.reporting.llm_report import (
    build_report_prompt,
    find_unverified_numbers,
    generate_llm_report,
    load_metrics_sources,
)


def _write_metrics(path: Path, metrics: dict) -> None:
    path.write_text(
        json.dumps(metrics),
        encoding="utf-8",
    )


def test_load_metrics_sources_reads_json(
    tmp_path: Path,
) -> None:
    metrics_path = (
        tmp_path / "metrics.json"
    )

    _write_metrics(
        metrics_path,
        {"external_test": {"balanced_accuracy": 0.41}},
    )

    sources = load_metrics_sources(
        {"grayscale_vae": metrics_path}
    )

    assert (
        sources["grayscale_vae"]
        ["external_test"]
        ["balanced_accuracy"]
        == 0.41
    )


def test_load_metrics_sources_missing_file_raises(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        load_metrics_sources(
            {"missing": tmp_path / "does_not_exist.json"}
        )


def test_build_report_prompt_embeds_numbers() -> None:
    metrics_by_label = {
        "rgb_vae": {
            "external_test": {
                "balanced_accuracy": 0.3478
            }
        },
    }

    system_prompt, user_prompt = (
        build_report_prompt(metrics_by_label)
    )

    assert "ONLY the numbers" in system_prompt
    assert "0.3478" in user_prompt


def test_build_report_prompt_requires_sources() -> None:
    with pytest.raises(ValueError):
        build_report_prompt({})


def test_find_unverified_numbers_flags_invented_values() -> None:
    metrics_by_label = {
        "rgb_vae": {
            "external_test": {
                "balanced_accuracy": 0.3478,
                "macro_f1": 0.3256,
            }
        },
    }

    grounded_report = (
        "RGB VAE reaches balanced accuracy 0.3478 "
        "and macro-F1 0.3256."
    )

    hallucinated_report = (
        "RGB VAE reaches balanced accuracy 0.9999."
    )

    assert find_unverified_numbers(
        grounded_report,
        metrics_by_label,
    ) == []

    unverified = find_unverified_numbers(
        hallucinated_report,
        metrics_by_label,
    )

    assert "0.9999" in unverified


def test_generate_llm_report_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("anthropic")

    monkeypatch.delenv(
        "ANTHROPIC_API_KEY",
        raising=False,
    )

    with pytest.raises(RuntimeError):
        generate_llm_report(
            metrics_by_label={
                "rgb_vae": {
                    "external_test": {
                        "balanced_accuracy": 0.34
                    }
                }
            },
            api_key=None,
        )
