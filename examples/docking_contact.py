"""Track a short docking clip, then ask whether the two vehicles are in contact.

Each frame updates object ids and pixel boxes. JEV is called once, on the
memory of the whole clip. Set TYPESAFE_API_KEY for that call. Without a key
the script still prints the tracks.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2

from jev_sees import Sees

VIDEO = Path(__file__).resolve().parents[1] / "assets" / "docking.mp4"
VOCABULARY = ["spacecraft", "space station", "cargo ship", "module"]
QUESTION = ["in_contact", "approaching", "separated", "uncertain"]


def main() -> None:
    capture = cv2.VideoCapture(str(VIDEO))
    if not capture.isOpened():
        raise SystemExit(f"Could not open {VIDEO}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 10.0
    step = max(1, int(round(fps / 5.0)))
    sees = Sees(vocabulary=VOCABULARY)
    index = 0
    kept = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % step == 0:
            tracks = sees.observe(frame[:, :, ::-1], vocabulary=VOCABULARY)
            kept += 1
            print(f"frame {index}")
            for item in tracks:
                print(f"  {item['object_id']} {item['label']} {item['bbox_xyxy']}")
        index += 1
    capture.release()
    print(f"observed {kept} frames")
    if not os.environ.get("TYPESAFE_API_KEY"):
        print("TYPESAFE_API_KEY is unset; skipped the JEV call")
        return
    result = sees.ask(
        "Are the spacecraft and the station in contact?",
        QUESTION,
    )
    print(result.choice, result.confidence)
    print(result.probabilities)


if __name__ == "__main__":
    sys.exit(main())
