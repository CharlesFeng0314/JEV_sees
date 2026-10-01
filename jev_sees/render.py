"""Draw boxes on a frame and write an image, video, or GIF."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def annotate(
    rgb: np.ndarray,
    tracks: list[dict],
    *,
    risks: dict[str, float] | None = None,
    captions: dict[str, str] | None = None,
    subject: str | None = None,
) -> np.ndarray:
    """Return an RGB frame with a box on each track."""

    canvas = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    risks = risks or {}
    captions = captions or {}
    for item in tracks:
        box = item.get("bbox_xyxy")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        object_id = str(item.get("object_id") or "")
        label = str(item.get("label") or "")
        x1, y1, x2, y2 = [int(round(float(value))) for value in box]
        risk = risks.get(object_id)
        if risk is not None:
            color = _risk_color(risk)
            text = f"{risk:.0%}"
        elif object_id in captions:
            color = (40, 180, 60)
            text = captions[object_id]
        elif subject and label == subject:
            color = (40, 180, 60)
            text = label
        else:
            color = (160, 160, 160)
            text = label
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        if text:
            _chip(canvas, text, x1, y1, color)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def write_visual(frames: list[np.ndarray], path: str | Path, *, fps: float) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError(f"No frame to write to {destination}")
    suffix = destination.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        bgr = cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(destination), bgr):
            raise ValueError(f"Could not write {destination}")
        return
    if suffix == ".gif":
        duration = max(40, int(round(1000 / max(fps, 0.1))))
        images = [Image.fromarray(frame) for frame in frames]
        quant = [image.convert("P", palette=Image.Palette.ADAPTIVE, colors=48) for image in images]
        quant[0].save(destination, save_all=True, append_images=quant[1:], duration=duration, loop=0, optimize=True)
        return
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(fps, 0.1),
        (width, height),
    )
    if not writer.isOpened():
        raise ValueError(f"Could not open a video writer for {destination}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _risk_color(risk: float) -> tuple[int, int, int]:
    if risk >= 0.6:
        return (40, 40, 220)
    if risk >= 0.3:
        return (0, 180, 255)
    return (40, 180, 60)


def _chip(canvas: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    scale = 0.5
    thickness = 1
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    top = y - height - baseline - 6
    if top < 0:
        top = min(y + 4, canvas.shape[0] - height - baseline - 4)
    left = min(max(0, x), max(0, canvas.shape[1] - width - 6))
    cv2.rectangle(canvas, (left, top), (left + width + 6, top + height + baseline + 4), color, -1)
    cv2.putText(
        canvas,
        text,
        (left + 3, top + height + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )
