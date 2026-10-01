"""Ask JEV how likely a car accident is for each pedestrian in a street clip.

The application supplies the official per-object JEV questions. Sampling,
tracking, rendering, and output files stay inside JEV Sees.
The clip is an excerpt of the Intel IoT sample ``person-bicycle-car-detection``
(CC BY 4.0). The call needs TYPESAFE_API_KEY; a JEV failure is raised.
"""

from __future__ import annotations

import sys
from pathlib import Path

from jev_sees import Noul, Sees

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "assets" / "traffic.mp4"
GIF = ROOT / "assets" / "demo.gif"
# The objects share the frame from just after 42s through about 50s.
WINDOW = (504, 600)
QUESTION = "视频中这个对象是正在或即将发生 car accident 的行人吗？"


def questions(objects: list[dict]) -> dict[str, Noul]:
    return {
        obj["object_id"]: Noul(instructions=f"{obj['object_id']}：{QUESTION}")
        for obj in objects
    }


def main() -> int:
    start, end = WINDOW
    Sees()(
        VIDEO,
        QUESTION,
        questions,
        save=GIF,
        start=start,
        end=end,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
