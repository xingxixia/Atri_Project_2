from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config(path: str | Path) -> dict[str, Any]:
    with project_path(path).open("r", encoding="utf-8-sig") as f:
        if str(path).lower().endswith(".json"):
            data = json.load(f)
        else:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError(
                    f"{path} is YAML, but PyYAML is not installed. Use the JSON config files or install pyyaml."
                ) from exc
            data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a config mapping in {path}")
    return data


load_yaml = load_config
