"""Draw boxes on a frame and write an image, video, or GIF."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_BG = (24, 17, 10)
_PANEL = (34, 25, 15)
_LINE = (78, 66, 48)
_CYAN = (224, 201, 48)
_TEXT = (235, 239, 242)
_MUTED = (153, 143, 126)


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
            text = f"{label} | {object_id} | {risk:.0%}"
        elif object_id in captions:
            color = (40, 180, 60)
            text = f"{label} | {object_id} | {captions[object_id]}"
        elif subject and label == subject:
            color = (40, 180, 60)
            text = f"{label} | {object_id}"
        else:
            color = (160, 160, 160)
            text = f"{label} | {object_id}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        if text:
            _chip(canvas, text, x1, y1, color)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


class DashboardRenderer:
    """Stateful video renderer with stable answer rows in a right-side panel."""

    def __init__(
        self,
        *,
        title: str = "JEV SEES // LIVE FRAME ANALYSIS",
        panel_title: str = "SELECTED OBJECTS // PROBABILITY",
    ) -> None:
        self.title = title
        self.panel_title = panel_title
        self.entries: dict[str, dict] = {}

    def render(
        self,
        rgb: np.ndarray,
        tracks: list[dict],
        answers: dict[str, object],
        *,
        frame_index: int,
        video_time_s: float,
        state_chars: int | None = None,
    ) -> np.ndarray:
        self._update(tracks, answers)
        source = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
        height, width = source.shape[:2]
        header_h = 62
        footer_h = 26
        panel_w = max(300, min(420, int(round(width * 0.43))))
        canvas = np.full((header_h + height + footer_h, width + panel_w, 3), _BG, dtype=np.uint8)
        canvas[header_h : header_h + height, :width] = source
        self._header(canvas, width, panel_w, frame_index, video_time_s, tracks, answers)
        self._boxes(canvas, tracks, answers, y_offset=header_h, video_width=width)
        self._panel(canvas, width, header_h, height, panel_w)
        self._footer(canvas, width, panel_w, state_chars)
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    def _update(self, tracks: list[dict], answers: dict[str, object]) -> None:
        by_id = {str(item.get("object_id")): item for item in tracks if item.get("object_id")}
        for entry in self.entries.values():
            entry["visible"] = False
        for key, answer in answers.items():
            object_id = str(key)
            track = by_id.get(object_id, {})
            probability = getattr(answer, "noul", None)
            confidence = getattr(answer, "confidence", None)
            choice = getattr(answer, "choice", None)
            value = probability if probability is not None else confidence
            entry = self.entries.setdefault(
                object_id,
                {
                    "object_id": object_id,
                    "label": str(track.get("label") or "object"),
                    "history": [],
                    "peak": 0.0,
                    "samples": 0,
                },
            )
            entry["label"] = str(track.get("label") or entry["label"])
            entry["visible"] = object_id in by_id
            if choice is not None:
                entry["choice"] = str(choice)
            if value is not None:
                numeric = max(0.0, min(1.0, float(value)))
                entry["current"] = numeric
                entry["peak"] = max(float(entry["peak"]), numeric)
                entry["samples"] = int(entry["samples"]) + 1
                entry["history"] = [*entry["history"], numeric][-24:]

    def _header(
        self,
        canvas: np.ndarray,
        width: int,
        panel_w: int,
        frame_index: int,
        video_time_s: float,
        tracks: list[dict],
        answers: dict[str, object],
    ) -> None:
        cv2.line(canvas, (0, 61), (width + panel_w, 61), _CYAN, 1, cv2.LINE_AA)
        _text(canvas, self.title, 18, 27, 0.58, _TEXT, 1)
        _text(canvas, "STRUCTURED VISUAL JUDGMENT", 18, 49, 0.38, _MUTED, 1)
        right = f"FRAME {frame_index:05d}   T+{video_time_s:06.2f}s"
        _right_text(canvas, right, width + panel_w - 18, 27, 0.48, _CYAN)
        counts = f"DETECTED {len(tracks):02d}   ANSWERED {len(answers):02d}"
        _right_text(canvas, counts, width + panel_w - 18, 49, 0.38, _MUTED)

    def _boxes(
        self,
        canvas: np.ndarray,
        tracks: list[dict],
        answers: dict[str, object],
        *,
        y_offset: int,
        video_width: int,
    ) -> None:
        for item in tracks:
            box = item.get("bbox_xyxy")
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            object_id = str(item.get("object_id") or "")
            label = _safe_text(str(item.get("label") or "object"))
            answer = answers.get(object_id)
            risk = getattr(answer, "noul", None) if answer is not None else None
            choice = getattr(answer, "choice", None) if answer is not None else None
            color = _risk_color(float(risk)) if risk is not None else _CYAN
            x1, y1, x2, y2 = [int(round(float(value))) for value in box]
            y1 += y_offset
            y2 += y_offset
            x1, x2 = sorted((max(0, x1), min(video_width - 1, x2)))
            y1, y2 = sorted((max(y_offset, y1), min(canvas.shape[0] - 1, y2)))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
            _corners(canvas, x1, y1, x2, y2, color)
            if risk is not None:
                suffix = f"  {float(risk):.0%}"
            elif choice is not None:
                suffix = f"  {choice}"
            else:
                suffix = ""
            _tech_chip(canvas, f"{label}  {object_id}{suffix}", x1, y1, color, video_width)

    def _panel(self, canvas: np.ndarray, left: int, top: int, height: int, panel_w: int) -> None:
        canvas[top : top + height, left:] = _PANEL
        cv2.line(canvas, (left, top), (left, top + height), _LINE, 1, cv2.LINE_AA)
        _text(canvas, self.panel_title, left + 18, top + 28, 0.43, _TEXT, 1)
        _text(canvas, "ROWS STAY PINNED AFTER FIRST DETECTION", left + 18, top + 49, 0.31, _MUTED, 1)
        cv2.line(canvas, (left + 18, top + 61), (left + panel_w - 18, top + 61), _LINE, 1)
        if not self.entries:
            _text(canvas, "WAITING FOR SELECTED OBJECTS", left + 18, top + 96, 0.4, _MUTED, 1)
            return
        available = max(1, height - 72)
        row_h = max(38, min(92, available // max(1, len(self.entries))))
        for index, entry in enumerate(self.entries.values()):
            y = top + 72 + index * row_h
            if y + 48 > top + height:
                break
            self._entry(canvas, entry, left + 18, y, panel_w - 36, row_h)

    def _entry(self, canvas: np.ndarray, entry: dict, x: int, y: int, width: int, row_h: int) -> None:
        current = float(entry.get("current", 0.0))
        label = _safe_text(str(entry.get("label") or "object")).upper()
        status = "LIVE" if entry.get("visible") else "MEMORY"
        choice = entry.get("choice")
        color = _CYAN if choice is not None else _risk_color(current)
        if row_h < 58:
            _text(canvas, f"{entry['object_id']} / {label}", x, y + 15, 0.34, _TEXT, 1)
            _right_text(canvas, f"{current:05.1%}  {status}", x + width, y + 15, 0.34, color)
            cv2.line(canvas, (x, y + row_h - 5), (x + width, y + row_h - 5), _LINE, 1)
            return
        _text(canvas, f"{entry['object_id']}  /  {label}", x, y + 16, 0.39, _TEXT, 1)
        _right_text(canvas, status, x + width, y + 16, 0.34, color if status == "LIVE" else _MUTED)
        primary = str(choice).upper() if choice is not None else f"{current:05.1%}"
        _text(canvas, primary, x, y + 45, 0.66, color, 2)
        _text(
            canvas,
            (
                f"CONF {current:05.1%}   N {int(entry.get('samples', 0)):02d}"
                if choice is not None
                else f"PEAK {float(entry.get('peak', 0.0)):05.1%}   N {int(entry.get('samples', 0)):02d}"
            ),
            x + 104,
            y + 41,
            0.34,
            _MUTED,
            1,
        )
        bar_y = y + min(row_h - 14, 61)
        cv2.rectangle(canvas, (x, bar_y), (x + width, bar_y + 5), _LINE, -1)
        cv2.rectangle(canvas, (x, bar_y), (x + int(round(width * current)), bar_y + 5), color, -1)
        history = entry.get("history") or []
        if len(history) >= 2 and row_h >= 82:
            _sparkline(canvas, history, x, y + 67, width, max(8, row_h - 76), color)

    def _footer(self, canvas: np.ndarray, width: int, panel_w: int, state_chars: int | None) -> None:
        y = canvas.shape[0] - 9
        _text(canvas, "ONE FRAME // MANY OBJECTS // ONE JEV CALL", 18, y, 0.33, _MUTED, 1)
        state = "STATE -- / 24K" if state_chars is None else f"STATE {state_chars / 1000:.1f}K / 24K"
        _right_text(canvas, state, width + panel_w - 18, y, 0.33, _CYAN)


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


def _text(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(
        canvas,
        _safe_text(text),
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _right_text(
    canvas: np.ndarray,
    text: str,
    right: int,
    y: int,
    scale: float,
    color: tuple[int, int, int],
) -> None:
    safe = _safe_text(text)
    width = cv2.getTextSize(safe, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
    _text(canvas, safe, right - width, y, scale, color, 1)


def _safe_text(text: str) -> str:
    return text.encode("ascii", "replace").decode("ascii")


def _corners(
    canvas: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
) -> None:
    span = max(7, min(18, (x2 - x1) // 4, (y2 - y1) // 4))
    for start, end in (
        ((x1, y1), (x1 + span, y1)),
        ((x1, y1), (x1, y1 + span)),
        ((x2, y1), (x2 - span, y1)),
        ((x2, y1), (x2, y1 + span)),
        ((x1, y2), (x1 + span, y2)),
        ((x1, y2), (x1, y2 - span)),
        ((x2, y2), (x2 - span, y2)),
        ((x2, y2), (x2, y2 - span)),
    ):
        cv2.line(canvas, start, end, color, 2, cv2.LINE_AA)


def _tech_chip(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    right_limit: int,
) -> None:
    safe = _safe_text(text)
    scale = 0.39
    (width, height), baseline = cv2.getTextSize(safe, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    top = y - height - baseline - 10
    if top < 62:
        top = y + 4
    left = min(max(0, x), max(0, right_limit - width - 16))
    cv2.rectangle(canvas, (left, top), (left + width + 12, top + height + baseline + 7), _BG, -1)
    cv2.rectangle(canvas, (left, top), (left + width + 12, top + height + baseline + 7), color, 1)
    _text(canvas, safe, left + 6, top + height + 1, scale, _TEXT, 1)


def _sparkline(
    canvas: np.ndarray,
    values: list[float],
    x: int,
    y: int,
    width: int,
    height: int,
    color: tuple[int, int, int],
) -> None:
    if width < 2 or height < 2:
        return
    points = []
    count = len(values)
    for index, value in enumerate(values):
        px = x + int(round(index * width / max(1, count - 1)))
        py = y + height - int(round(max(0.0, min(1.0, float(value))) * height))
        points.append((px, py))
    cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, color, 1, cv2.LINE_AA)


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
