"""Thin view over the official JEV response."""

from __future__ import annotations

from typing import Any


class Result:
    """Official ``SystemOneResponse`` plus shortcuts for a single question."""

    def __init__(self, response: Any):
        self.response = response

    @property
    def answers(self) -> dict[str, Any]:
        return self.response.answers

    @property
    def model(self) -> str:
        return self.response.model

    @property
    def usage(self) -> Any:
        return getattr(self.response, "usage", None)

    def model_dump(self) -> dict[str, Any]:
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
