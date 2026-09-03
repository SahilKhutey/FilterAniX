from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_temporal_plan(
    path: str | Path,
) -> dict[int, dict[str, Any]]:

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Temporal plan not found: {path}"
        )

    result: dict[int, dict[str, Any]] = {}

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for line in handle:

            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            index = int(
                item.get(
                    "frame_index",
                    item.get("frame", 0),
                )
            )

            result[index] = item

    return result
