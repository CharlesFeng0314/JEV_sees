"""Thin view over the official JEV response, plus the table for one call."""

from __future__ import annotations

from typing import Any

from .report import format_table, frame_rows


class Result:
    """Official ``SystemOneResponse`` plus the rows of one image or video call."""

    def __init__(
        self,
        response: Any = None,
        *,
        rows: list[dict[str, Any]] | None = None,
        timeline: list[dict[str, Any]] | None = None,
        frames: list[dict[str, Any]] | None = None,
        question: str = "",
    ):
        self.response = response
        self.rows = list(rows or [])
        self.timeline = list(timeline or [])
        self.frames = list(frames or [])
        self.question = question

    def __str__(self) -> str:
        if self.frames:
            return format_table(frame_rows(self.frames))
        if self.rows or self.question or self.timeline:
            return format_table(self.rows)
        if self.response is None:
            return ""
        choice = getattr(self._only(), "choice", None)
        if choice is not None:
            confidence = getattr(self._only(), "confidence", None)
            if confidence is None:
                return str(choice)
            return f"{choice} {float(confidence):.2f}"
        noul = getattr(self._only(), "noul", None)
        if noul is None:
            return ""
        return f"{float(noul):.2f}"

    @property
    def answers(self) -> dict[str, Any]:
        if self.response is None:
            return {}
        return self.response.answers

    @property
    def model(self) -> str:
        if self.response is None:
            return ""
        return self.response.model

    @property
    def usage(self) -> Any:
        if self.response is None:
            return None
        return getattr(self.response, "usage", None)

    def model_dump(self) -> dict[str, Any]:
        if self.frames or self.rows or self.question or self.timeline:
            payload = {
                "question": self.question,
                "frames": self.frames,
                "summary": self.rows,
                "timeline": self.timeline,
            }
            if self.response is not None:
                payload["response"] = self.response.model_dump()
            return payload
        if self.response is None:
            return {}
        return self.response.model_dump()

    @property
    def choice(self) -> str | None:
        answer = self._only()
        return getattr(answer, "choice", None)

    @property
    def confidence(self) -> float | None:
        answer = self._only()
        value = getattr(answer, "confidence", None)
        return None if value is None else float(value)

    @property
    def probabilities(self) -> dict[str, float] | None:
        answer = self._only()
        values = getattr(answer, "probabilities", None)
        if values is None:
            return None
        return {str(key): float(value) for key, value in dict(values).items()}

    @property
    def noul(self) -> float | None:
        answer = self._only()
        value = getattr(answer, "noul", None)
        return None if value is None else float(value)

    def _only(self) -> Any:
        answers = self.answers
        if len(answers) != 1:
            raise ValueError("This call returned multiple answers; read result.answers")
        return next(iter(answers.values()))
