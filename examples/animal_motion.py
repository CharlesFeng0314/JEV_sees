"""Classify every animal's motion state in an image or video."""

from __future__ import annotations

import sys
from pathlib import Path

from jev_sees import Sees

MEDIA = Path("path/to/animals.mp4")
QUESTION = "What is this animal's current motion state?"
CHOICES = ["stationary", "walking", "running", "uncertain"]


def main() -> int:
    Sees().for_each("animal").choice(
        MEDIA,
        QUESTION,
        choices=CHOICES,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
