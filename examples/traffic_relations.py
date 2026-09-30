"""Watch a street clip and ask JEV, several times a second, how likely a car accident is for each pedestrian.

One call per sampled frame carries every visible person as their own yes/no
question. JEV answers the batch together. The yes-probability on each answer
is the accident risk drawn on that person.

The clip is an excerpt of the Intel IoT sample ``person-bicycle-car-detection``
(CC BY 4.0). Without TYPESAFE_API_KEY the script still prints tracks and writes
the GIF.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from jev_sees import Sees
from jev_sees.tracking import box_gap

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "assets" / "traffic.mp4"
GIF = ROOT / "docs" / "demo.gif"
VOCABULARY = ["person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light"]
ACTORS = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
# The objects share the frame from just after 42s through about 50s at 12 fps.
WINDOW = (504, 600)
ASK_EVERY = 6
MAX_ACTORS = 8
GIF_WIDTH = 480
VEHICLES = {"car", "bus", "truck", "motorcycle"}
TITLE = "chance (%) of person having car accident"
PROMPT = "Each question is one pedestrian. Yes means a car accident involving that person is happening or imminent in this frame."


def main() -> int:
    capture = cv2.VideoCapture(str(VIDEO))
    if not capture.isOpened():
        raise SystemExit(f"Could not open {VIDEO}")
    sees = Sees(vocabulary=VOCABULARY)
    index = 0
    gif_frames: list[Image.Image] = []
    asked = 0
    person_of: dict[str, int] = {}
    start, end = WINDOW
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index == 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            index = start
            continue
        if index > end:
            break
        tracks = sees.observe(frame[:, :, ::-1], vocabulary=VOCABULARY)
        answers = None
        if index % ASK_EVERY == 0:
            questions = _questions(tracks)
            if questions:
                _print_tracks(index, tracks)
                if sees.api_key:
                    result = _ask(sees, questions)
                    answers = {
                        key: float(getattr(answer, "noul"))
                        for key, answer in result.answers.items()
                    }
                    asked += 1
                    for key, risk in answers.items():
                        print(f"  {key} accident {risk:.2f}")
                else:
                    print("  TYPESAFE_API_KEY is unset; skipped the JEV call")
                gif_frames.append(_draw(frame, tracks, answers, person_of))
        index += 1
    capture.release()
    if not gif_frames:
        raise SystemExit("No sampled frame contained a road user")
    _write_gif(gif_frames, GIF)
    print(f"observed window {start}-{end}, asked {asked} times, wrote {GIF}")
    return 0


def _ask(sees: Sees, questions: dict):
    from typesafe_sdk import TypeSafeError

    last = None
    for _ in range(3):
        try:
            return sees.ask(PROMPT, questions)
        except TypeSafeError as exc:
            last = exc
            print(f"  retry after {type(exc).__name__}")
    raise last


def _questions(tracks: list[dict]) -> dict[str, dict[str, str]]:
    people = [item for item in tracks if item.get("label") == "person" and item.get("bbox_xyxy")]
    vehicles = [item for item in tracks if item.get("label") in VEHICLES and item.get("bbox_xyxy")]
    people.sort(key=lambda item: _nearest_gap(item, vehicles or people))
    questions = {}
    for item in people[:MAX_ACTORS]:
        name = f"{item['object_id']} ({item['label']})"
        vehicle = _nearest(item, vehicles)
        if vehicle is None:
            hazard = "no car is visible"
        else:
            gap = box_gap(item["bbox_xyxy"], vehicle["bbox_xyxy"])
            hazard = f"the nearest vehicle is {vehicle['object_id']} ({vehicle['label']}), box gap {gap:.0f}px"
        questions[str(item["object_id"])] = {
            "yes": f"{name} is in a car accident now, or a car is about to hit them. {hazard}.",
            "no": f"{name} is clear of every car. {hazard}.",
        }
    return questions


def _nearest(item: dict, actors: list[dict]) -> dict | None:
    others = [other for other in actors if other.get("object_id") != item.get("object_id")]
    if not others:
        return None
    return min(others, key=lambda other: box_gap(item["bbox_xyxy"], other["bbox_xyxy"]))


def _nearest_gap(item: dict, actors: list[dict]) -> float:
    other = _nearest(item, actors)
    if other is None:
        return 1e9
    return box_gap(item["bbox_xyxy"], other["bbox_xyxy"])


def _print_tracks(index: int, tracks: list[dict]) -> None:
    print(f"frame {index}")
    for item in tracks:
        if item.get("label") not in ACTORS:
            continue
        print(f"  {item['object_id']} {item['label']} {item['bbox_xyxy']}")


def _draw(
    frame,
    tracks: list[dict],
    risks: dict[str, float] | None,
    person_of: dict[str, int],
) -> Image.Image:
    height, width = frame.shape[:2]
    scale = GIF_WIDTH / float(width)
    canvas = cv2.resize(frame, (GIF_WIDTH, max(1, int(round(height * scale)))))
    current: dict[int, str] = {}
    for item in tracks:
        if item.get("label") not in ACTORS or not item.get("bbox_xyxy"):
            continue
        x1, y1, x2, y2 = [int(round(float(value) * scale)) for value in item["bbox_xyxy"]]
        risk = None if risks is None else risks.get(str(item["object_id"]))
        if item["label"] == "person" and risk is not None:
            number = person_of.setdefault(str(item["object_id"]), len(person_of) + 1)
            current[number] = f"{risk * 100:.0f}"
            color = _risk_color(risk)
            caption = f"{risk:.0%}"
        elif item["label"] in VEHICLES:
            color = (160, 160, 160)
            caption = item["label"]
        else:
            color = (160, 160, 160)
            caption = ""
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        if caption:
            _chip(canvas, caption, x1, y1, color)
    rows = [f"person{number}: {current.get(number, '')}" for number in range(1, max(2, len(person_of)) + 1)]
    return _chrome(canvas, rows)


def _chrome(canvas: np.ndarray, rows: list[str]) -> Image.Image:
    title = np.zeros((36, GIF_WIDTH, 3), dtype=np.uint8)
    cv2.putText(title, TITLE, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    foot = np.zeros((22 * len(rows) + 12, GIF_WIDTH, 3), dtype=np.uint8)
    for index, row in enumerate(rows):
        cv2.putText(foot, row, (8, 22 + index * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    stacked = np.vstack([title, canvas, foot])
    return Image.fromarray(cv2.cvtColor(stacked, cv2.COLOR_BGR2RGB))


def _risk_color(risk: float) -> tuple[int, int, int]:
    if risk >= 0.6:
        return (40, 40, 220)
    if risk >= 0.3:
        return (0, 180, 255)
    return (40, 180, 60)


def _chip(canvas, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    scale = 0.5
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    top = y - th - baseline - 6
    if top < 0:
        top = min(y + 4, canvas.shape[0] - th - baseline - 4)
    left = min(max(0, x), canvas.shape[1] - tw - 6)
    cv2.rectangle(canvas, (left, top), (left + tw + 6, top + th + baseline + 4), color, -1)
    cv2.putText(
        canvas,
        text,
        (left + 3, top + th + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def _write_gif(frames: list[Image.Image], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    quant = [frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=48) for frame in frames]
    quant[0].save(path, save_all=True, append_images=quant[1:], duration=280, loop=0, optimize=True)


if __name__ == "__main__":
    sys.exit(main())
