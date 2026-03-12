"""YAML config loading utilities."""

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_project_root() -> Path:
    """Walk up from this file to find trail_project/ root."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "configs").exists() and (parent / "src").exists():
            return parent
    raise RuntimeError("Could not locate trail_project root directory.")


PROJECT_ROOT = find_project_root()


def load_data_config(name: str) -> dict[str, Any]:
    return load_yaml(PROJECT_ROOT / "configs" / "data" / f"{name}.yaml")


def load_model_config(name: str) -> dict[str, Any]:
    return load_yaml(PROJECT_ROOT / "configs" / "model" / f"{name}.yaml")


def load_experiment_config(name: str) -> dict[str, Any]:
    return load_yaml(PROJECT_ROOT / "configs" / "experiment" / f"{name}.yaml")


def load_eval_config(name: str) -> dict[str, Any]:
    return load_yaml(PROJECT_ROOT / "configs" / "eval" / f"{name}.yaml")


def resolve_path(relative_path: str) -> Path:
    """Resolve a path relative to the project root."""
    return (PROJECT_ROOT / relative_path).resolve()
