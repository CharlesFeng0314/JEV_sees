"""Ask JEV what color the detected bus is."""

from __future__ import annotations

import sys
from pathlib import Path

from jev_sees import Choice, Sees

IMAGE = Path(__file__).resolve().parents[1] / "assets" / "bus.jpg"
QUESTION = "What color is the bus?"


def questions(objects: list[dict]) -> dict[str, Choice]:
    buses = [item for item in objects if "bus" in str(item.get("label", "")).lower()]
    return {
        item["object_id"]: Choice(
            instructions=f"What color is {item['object_id']}, the detected bus?",
            criteria={
                "yellow": None,
                "red": None,
                "white": None,
                "blue": None,
                "black": None,
                "uncertain": None,
            },
        )
        for item in buses
    }


def main() -> int:
    Sees()(
        IMAGE,
        QUESTION,
        questions,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
