"""Print each sampled frame's car-accident probability for every pedestrian."""

from __future__ import annotations

import sys
from pathlib import Path

from jev_sees import Sees

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "assets" / "traffic.mp4"
# The objects share the frame from just after 42s through about 50s.
WINDOW = (504, 600)
QUESTION = "Is this object currently involved in a car accident?"


def main() -> int:
    start, end = WINDOW
    Sees().for_each("pedestrian").probability(
        VIDEO,
        QUESTION,
        start=start,
        end=end,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
