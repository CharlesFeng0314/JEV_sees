"""Tables, JSON, and Excel for one call."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_PRINT_ORDER = ("object_id", "label", "risk", "choice", "confidence", "peak_frame")
_HIDDEN = {"bbox_xyxy", "_score"}


class JudgmentLog:
    """Fold per-sample answers into one row per object, plus a timeline."""

    def __init__(self) -> None:
        self.peaks: dict[str, dict[str, Any]] = {}
        self.timeline: list[dict[str, Any]] = []
        self.scene: list[dict[str, Any]] = []
        self.seen: dict[str, dict[str, Any]] = {}
        self.had_questions = False

    def see(self, tracks: list[dict[str, Any]]) -> None:
        for item in tracks:
            object_id = str(item.get("object_id") or "")
            if not object_id:
                continue
            self.seen[object_id] = {
                "object_id": object_id,
                "label": item.get("label"),
                "bbox_xyxy": _bbox(item),
            }

    def add(self, frame: int | None, tracks: list[dict[str, Any]], answers: dict[str, Any]) -> None:
        if not answers:
            return
        by_id = {str(item.get("object_id")): item for item in tracks if item.get("object_id")}
        for key, answer in answers.items():
            track = by_id.get(str(key))
            risk = _number(answer, "noul")
            choice = getattr(answer, "choice", None)
            confidence = _number(answer, "confidence")
            if track is not None and risk is not None:
                self._peak_risk(frame, str(key), track, risk)
            elif track is not None and choice is not None:
                self._peak_choice(frame, str(key), track, str(choice), confidence)
            elif track is None and len(answers) == 1 and choice is not None:
                self.scene.append(_scene_choice(frame, str(choice), confidence))
            elif track is None and len(answers) == 1 and risk is not None:
                self.scene.append(_scene_risk(frame, risk))

    def export_timeline(self) -> list[dict[str, Any]]:
        if self.timeline:
            return list(self.timeline)
        return list(self.scene)

    def rows(self) -> list[dict[str, Any]]:
        if self.peaks:
            rows = []
            for key in sorted(self.peaks):
                row = {name: value for name, value in self.peaks[key].items() if name not in {"_score"}}
                rows.append(row)
            return rows
        if self.scene:
            return [_best_scene(self.scene)]
        return [dict(self.seen[key]) for key in sorted(self.seen)]

    def _peak_risk(self, frame: int | None, object_id: str, track: dict[str, Any], risk: float) -> None:
        entry = {
            "frame": frame,
            "object_id": object_id,
            "label": track.get("label"),
            "risk": risk,
            "bbox_xyxy": _bbox(track),
        }
        self.timeline.append({key: value for key, value in entry.items() if value is not None or key == "frame"})
        if frame is None:
            self.timeline[-1].pop("frame", None)
        previous = self.peaks.get(object_id)
        if previous is not None and risk < float(previous["risk"]):
            return
        row = {
            "object_id": object_id,
            "label": track.get("label"),
            "risk": risk,
            "bbox_xyxy": _bbox(track),
        }
        if frame is not None:
            row["peak_frame"] = frame
        self.peaks[object_id] = row

    def _peak_choice(
        self,
        frame: int | None,
        object_id: str,
        track: dict[str, Any],
        choice: str,
        confidence: float | None,
    ) -> None:
        score = 0.0 if confidence is None else confidence
        sample = {
            "object_id": object_id,
            "label": track.get("label"),
            "choice": choice,
            "confidence": confidence,
            "bbox_xyxy": _bbox(track),
        }
        if frame is not None:
            sample["frame"] = frame
        self.timeline.append(sample)
        previous = self.peaks.get(object_id)
        if previous is not None and score < float(previous.get("_score", -1)):
            return
        row = {
            "object_id": object_id,
            "label": track.get("label"),
            "choice": choice,
            "confidence": confidence,
            "bbox_xyxy": _bbox(track),
            "_score": score,
        }
        if frame is not None:
            row["peak_frame"] = frame
        self.peaks[object_id] = row


def format_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No matching object."
    columns = [name for name in _PRINT_ORDER if any(name in row for row in rows)]
    for row in rows:
        for name in row:
            if name not in columns and name not in _HIDDEN:
                columns.append(name)
    if not columns:
        return "No matching object."
    cells = [[_cell(row.get(name)) for name in columns] for row in rows]
    widths = [len(name) for name in columns]
    for line in cells:
        for index, value in enumerate(line):
            widths[index] = max(widths[index], len(value))
    head = "  ".join(name.ljust(widths[index]) for index, name in enumerate(columns))
    body = ["  ".join(value.ljust(widths[index]) for index, value in enumerate(line)) for line in cells]
    return "\n".join([head, *body])


def write_json(path: str | Path, question: str, rows: list[dict[str, Any]], timeline: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"question": question, "rows": rows, "timeline": timeline}
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_excel(path: str | Path, rows: list[dict[str, Any]], timeline: list[dict[str, Any]]) -> None:
    from openpyxl import Workbook

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    book = Workbook()
    _fill_sheet(book.active, "Summary", rows)
    _fill_sheet(book.create_sheet("Timeline"), "Timeline", timeline)
    book.save(destination)


def _fill_sheet(sheet: Any, title: str, rows: list[dict[str, Any]]) -> None:
    sheet.title = title
    columns: list[str] = []
    for row in rows:
        for name in row:
            if name not in columns and name not in {"_score"}:
                columns.append(name)
    if not columns:
        columns = ["object_id"]
    sheet.append(columns)
    for row in rows:
        sheet.append([_excel_cell(row.get(name)) for name in columns])


def _best_scene(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if "risk" in samples[0]:
        best = max(samples, key=lambda item: float(item["risk"]))
        row: dict[str, Any] = {"risk": best["risk"]}
    else:
        best = max(samples, key=lambda item: float(item.get("confidence") or 0.0))
        row = {"choice": best.get("choice"), "confidence": best.get("confidence")}
    if best.get("frame") is not None:
        row["peak_frame"] = best["frame"]
    return row


def _scene_choice(frame: int | None, choice: str, confidence: float | None) -> dict[str, Any]:
    row: dict[str, Any] = {"choice": choice, "confidence": confidence}
    if frame is not None:
        row["frame"] = frame
    return row


def _scene_risk(frame: int | None, risk: float) -> dict[str, Any]:
    row: dict[str, Any] = {"risk": risk}
    if frame is not None:
        row["frame"] = frame
    return row


def _bbox(item: dict[str, Any] | None) -> list[float] | None:
    if not item:
        return None
    box = item.get("bbox_xyxy")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    return [float(value) for value in box]


def _number(answer: Any, name: str) -> float | None:
    value = getattr(answer, name, None)
    if value is None:
        return None
    return float(value)


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _excel_cell(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return ",".join(str(part) for part in value)
    return value
