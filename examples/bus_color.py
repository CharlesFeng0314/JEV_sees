"""Ask JEV what color the bus is.

Prints the tracks first. The JEV call needs TYPESAFE_API_KEY; without it the
script still prints what the camera path saw. The picture is the Ultralytics
bus.jpg sample.
"""

from __future__ import annotations

import sys
from pathlib import Path

from jev_sees import Sees

IMAGE = Path(__file__).resolve().parents[1] / "assets" / "bus.jpg"


def main() -> int:
    sees = Sees(vocabulary=["bus", "person", "car"])
    tracks = sees.observe(IMAGE)
    for item in tracks:
        color = (item.get("attributes") or {}).get("color", "")
        print(f"{item['object_id']} {item['label']} {item['bbox_xyxy']} {color}")
    if not sees.api_key:
        print("TYPESAFE_API_KEY is unset; skipped the JEV call")
        return 0
    result = sees.ask(
        "What color is the bus?",
        ["yellow", "red", "white", "blue", "black", "uncertain"],
    )
    print(result.choice, result.confidence)
    print(result.probabilities)
    return 0


if __name__ == "__main__":
    sys.exit(main())
