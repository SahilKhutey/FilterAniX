import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config():
    return load_json(ROOT / "config.json")


def load_styles():
    return load_json(ROOT / "styles.json")
