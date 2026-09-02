from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Loads configuration from YAML or JSON file."""
    if path is None:
        yaml_default = ROOT / "configs" / "default.yaml"
        if yaml_default.exists():
            return load_config(yaml_default)
        json_default = ROOT / "config.json"
        if json_default.exists():
            return load_json(json_default)
        return {}

    path_obj = Path(path)
    if path_obj.suffix.lower() in {".yaml", ".yml"}:
        with open(path_obj, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return load_json(path_obj)


def load_styles(path: Optional[str | Path] = None) -> Dict[str, Any]:
    if path is None:
        path = ROOT / "styles.json"
    if Path(path).exists():
        return load_json(path)
    return {}
