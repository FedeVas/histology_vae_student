from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    Загружает YAML-конфигурацию проекта.

    Parameters
    ----------
    config_path:
        Путь до YAML-файла.

    Returns
    -------
    dict[str, Any]
        Конфигурация в виде словаря.

    Raises
    ------
    FileNotFoundError
        Если файл конфигурации не найден.
    ValueError
        Если YAML-файл пуст или его корневой объект не является словарём.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file was not found: {path.resolve()}"
        )

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Configuration file is empty: {path.resolve()}")

    if not isinstance(config, dict):
        raise ValueError(
            "The root object in the configuration file must be a dictionary."
        )

    return config