"""Print each sampled frame's live car-accident probability per pedestrian."""

from __future__ import annotations

import sys
from pathlib import Path

from jev_sees import Noul, Sees

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "assets" / "traffic.mp4"
# The objects share the frame from just after 42s through about 50s.
WINDOW = (504, 600)
QUESTION = "当前采样帧中每个行人正在发生 car accident 的实时概率"


def is_person(obj: dict) -> bool:
    text = f"{obj.get('label', '')} {obj.get('description', '')}".lower()
    return any(word in text for word in ("person", "pedestrian", "man", "woman", "boy", "girl"))


def questions(objects: list[dict]) -> dict[str, Noul]:
    return {
        obj["object_id"]: Noul(
            instructions=f"当前帧中，{obj['object_id']} 正在发生 car accident 吗？"
        )
        for obj in objects
        if is_person(obj)
    }


def main() -> int:
    start, end = WINDOW
    Sees()(
        VIDEO,
        QUESTION,
        questions,
        start=start,
        end=end,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
