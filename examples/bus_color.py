"""Ask JEV what color the bus is.

Requires TYPESAFE_API_KEY. The picture is the Ultralytics bus.jpg sample.
"""

from pathlib import Path

from jev_sees import Sees

IMAGE = Path(__file__).resolve().parents[1] / "assets" / "bus.jpg"


def main() -> None:
    sees = Sees(vocabulary=["bus", "person", "car"])
    result = sees(
        IMAGE,
        "What color is the bus?",
        ["yellow", "red", "white", "blue", "black", "uncertain"],
    )
    print(result.choice, result.confidence)
    print(result.probabilities)


if __name__ == "__main__":
    main()
