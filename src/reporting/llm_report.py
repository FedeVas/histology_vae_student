from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


"""
LLM-assisted experiment reporting.

Мотивация: вакансия явно упоминает "Integrate with LLM API for
process automation and support". Этот модуль — небольшой, но
настоящий пример такой интеграции: он не генерирует числа, а берёт
уже посчитанные, детерминированные metrics.json (из evaluate.py,
run_linear_probe.py, run_retrieval.py) и просит LLM написать связный
текстовый черновик сравнения экспериментов на их основе.

Дизайн сознательно консервативен:

    - LLM не видит сырых изображений и не принимает решений;
    - system prompt явно запрещает придумывать числа, которых нет
      во входных metrics.json;
    - после генерации отчёта числа в тексте проверяются против
      исходных JSON (grounding check), а не просто "доверяются" LLM;
    - это черновик для человека-исследователя, а не автоматический
      источник научных выводов.

Это соответствует Scope and limitations из README проекта: отчёт
не должен делать клинических утверждений и не заменяет пробинг
и bootstrap-анализ, уже реализованные в src/analysis.
"""


NUMBER_PATTERN = re.compile(
    r"-?\d+\.\d+"
)


@dataclass(frozen=True)
class MetricsSource:
    """
    Один источник метрик, например одна модель или один эксперимент.
    """

    label: str
    metrics_path: Path


def load_metrics_sources(
    sources: dict[str, str | Path],
) -> dict[str, dict[str, Any]]:
    """
    Загружает несколько metrics.json файлов по их меткам.

    Parameters
    ----------
    sources:
        Словарь {label: путь_к_metrics.json}, например:

        {
            "rgb_vae": "outputs/rgb_vae/probe/metrics.json",
            "grayscale_vae": "outputs/grayscale_vae/probe/metrics.json",
        }
    """
    if not sources:
        raise ValueError(
            "sources must contain at least one entry."
        )

    loaded: dict[str, dict[str, Any]] = {}

    for label, path in sources.items():
        metrics_path = Path(path)

        if not metrics_path.exists():
            raise FileNotFoundError(
                f"Metrics file was not found for "
                f"{label!r}: {metrics_path.resolve()}"
            )

        with metrics_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            loaded[label] = json.load(file)

    return loaded


def build_report_prompt(
    metrics_by_label: dict[str, dict[str, Any]],
    research_question: str | None = None,
) -> tuple[str, str]:
    """
    Строит (system_prompt, user_prompt) для LLM-отчёта.

    Числа в user_prompt — это единственный источник фактов,
    доступный модели.
    """
    if not metrics_by_label:
        raise ValueError(
            "metrics_by_label must not be empty."
        )

    system_prompt = (
        "You are a careful machine learning scientist "
        "writing an internal results summary.\n\n"
        "Hard rules:\n"
        "1. Use ONLY the numbers given to you in the "
        "JSON below. Never invent, round in a misleading "
        "way, or estimate a number that is not present.\n"
        "2. If a comparison the user asks about cannot be "
        "supported by the given numbers, say so explicitly "
        "instead of guessing.\n"
        "3. Do not make clinical claims. This project "
        "evaluates representation quality on public tissue-"
        "class patches; it does not predict patient outcomes "
        "and must not be described as doing so.\n"
        "4. Prefer precise, hedged scientific language over "
        "confident marketing language (e.g. 'is consistent "
        "with', not 'proves').\n"
        "5. When you state a number, copy it exactly as given."
    )

    metrics_json = json.dumps(
        metrics_by_label,
        indent=2,
        sort_keys=True,
    )

    question_line = (
        research_question
        if research_question
        else (
            "Summarize what these results show, "
            "including any disagreements between metrics "
            "and any result you cannot fully explain."
        )
    )

    user_prompt = (
        "Here are metrics.json outputs from several "
        "experiments in a histology representation-"
        "learning project:\n\n"
        f"{metrics_json}\n\n"
        f"Task: {question_line}\n\n"
        "Write a short markdown section (headings + "
        "bullet points) with:\n"
        "- one paragraph overview;\n"
        "- a bullet list of the clearest numeric "
        "differences between experiments;\n"
        "- one paragraph of caveats (sample size, "
        "single seed, patch-level evaluation, etc., "
        "only if such fields are present in the data)."
    )

    return system_prompt, user_prompt


def find_unverified_numbers(
    report_text: str,
    metrics_by_label: dict[str, dict[str, Any]],
) -> list[str]:
    """
    Простая grounding-проверка: находит decimal-числа в отчёте,
    которых нет (с точностью до округления) ни в одном исходном
    metrics.json.

    Это эвристика, а не строгое доказательство отсутствия
    галлюцинаций: она не проверяет счётчики (integers) и не
    понимает единицы измерения. Но она ловит наиболее опасный
    случай — придуманную LLM метрику.
    """
    known_numbers = _flatten_numeric_values(
        metrics_by_label
    )

    tolerance = 1e-6

    unverified: list[str] = []

    for match in NUMBER_PATTERN.finditer(
        report_text
    ):
        candidate = float(match.group())

        is_known = any(
            abs(candidate - known) < tolerance
            for known in known_numbers
        )

        if not is_known:
            unverified.append(match.group())

    return unverified


def generate_llm_report(
    metrics_by_label: dict[str, dict[str, Any]],
    research_question: str | None = None,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 1200,
    api_key: str | None = None,
) -> str:
    """
    Вызывает Anthropic API для генерации черновика отчёта.

    Требует установленного пакета `anthropic` и переменной
    окружения ANTHROPIC_API_KEY (или явного api_key).
    """
    try:
        import anthropic
    except ImportError as error:
        raise ImportError(
            "The 'anthropic' package is required for "
            "LLM-assisted reporting. Install it with: "
            "pip install anthropic"
        ) from error

    resolved_api_key = (
        api_key
        or os.environ.get("ANTHROPIC_API_KEY")
    )

    if not resolved_api_key:
        raise RuntimeError(
            "No Anthropic API key was found. Set the "
            "ANTHROPIC_API_KEY environment variable or "
            "pass api_key explicitly."
        )

    system_prompt, user_prompt = build_report_prompt(
        metrics_by_label=metrics_by_label,
        research_question=research_question,
    )

    client = anthropic.Anthropic(
        api_key=resolved_api_key
    )

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
    )

    report_text = "".join(
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text"
    )

    unverified_numbers = find_unverified_numbers(
        report_text=report_text,
        metrics_by_label=metrics_by_label,
    )

    if unverified_numbers:
        report_text += (
            "\n\n> **Grounding warning:** the following "
            "numbers appear in this draft but were not "
            "found in the supplied metrics.json files and "
            "should be verified manually before use: "
            + ", ".join(sorted(set(unverified_numbers)))
        )

    return report_text


def _flatten_numeric_values(
    value: Any,
) -> list[float]:
    numbers: list[float] = []

    if isinstance(value, bool):
        return numbers

    if isinstance(value, (int, float)):
        numbers.append(float(value))

    elif isinstance(value, dict):
        for item in value.values():
            numbers.extend(
                _flatten_numeric_values(item)
            )

    elif isinstance(value, list):
        for item in value:
            numbers.extend(
                _flatten_numeric_values(item)
            )

    return numbers
