"""Ask JEV what color the detected bus is."""

from __future__ import annotations

import sys
from pathlib import Path

from jev_sees import Choice, Sees, TypeSafeClient

IMAGE = Path(__file__).resolve().parents[1] / "assets" / "bus.jpg"
QUESTION = "What color is the bus?"


def main() -> int:
    sees = Sees()
    sees.observe(IMAGE)

    with TypeSafeClient() as client:
        response = client.system_one(
            state=sees.state(QUESTION),
            questions={
                "bus_color": Choice(
                    instructions=QUESTION,
                    criteria={
                        "yellow": None,
                        "red": None,
                        "white": None,
                        "blue": None,
                        "black": None,
                        "uncertain": None,
                    },
                )
            },
        )

    answer = response.choices["bus_color"]
    print(answer.choice, answer.confidence)
    print(answer.probabilities)
    return 0


if __name__ == "__main__":
    sys.exit(main())
