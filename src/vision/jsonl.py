from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from .schema import validate_vision_frame
from .types import VisionFrame


def write_vision_jsonl(
    frames: Iterable[VisionFrame],
    output: str | Path,
) -> int:
    output = Path(output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    count = 0

    with output.open(
        "w",
        encoding="utf-8",
    ) as handle:

        for frame in frames:
            data = frame.to_dict()

            validate_vision_frame(data)

            handle.write(
                json.dumps(
                    data,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

            count += 1

    return count


def read_vision_jsonl(
    path: str | Path,
) -> Iterator[dict]:
    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}"
                ) from exc

            validate_vision_frame(data)

            yield data
