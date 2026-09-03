from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.art.mathematical.config import (
    MathematicalAnimeStyle,
)


def main() -> None:

    style = MathematicalAnimeStyle.creator_anime()

    print("=" * 60)
    print("FilterAniX Mathematical Engine")
    print("MTH-01 Configuration")
    print("=" * 60)

    print(
        json.dumps(
            style.to_dict(),
            indent=2,
        )
    )

    print("=" * 60)
    print("Configuration VALID")
    print("=" * 60)


if __name__ == "__main__":
    main()
