"""Turn ordinary Python into official JEV question objects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from typesafe_sdk import Choice, Noul, Score


def compile_questions(spec: object, prompt: str) -> dict[str, Choice | Noul | Score]:
    """Compile one question or a mapping of questions.

    A list of labels is a Choice. ``"yes/no"`` and ``{"yes": ..., "no": ...}``
    are Noul questions. ``{"rubric": [...]}`` and a tuple of labels are Score
    questions. A mapping whose values are themselves questions becomes one
    official question per key.
    """

    if _is_question_mapping(spec):
        assert isinstance(spec, Mapping)
        return {str(key): compile_one(value, prompt) for key, value in spec.items()}
    return {"answer": compile_one(spec, prompt)}


def compile_one(spec: object, prompt: str) -> Choice | Noul | Score:
    if spec == "yes/no":
        return Noul(instructions=prompt)
    if isinstance(spec, tuple):
        rubric = [_text(item) for item in spec]
        if not rubric:
            raise ValueError("A score rubric needs at least one level")
        return Score(instructions=prompt, criteria=rubric)
    if isinstance(spec, Mapping) and set(spec) <= {"yes", "no"} and spec:
        criteria = {}
        if "yes" in spec:
            criteria["true"] = spec["yes"]
        if "no" in spec:
            criteria["false"] = spec["no"]
        return Noul(instructions=prompt, criteria=criteria)
    if isinstance(spec, Mapping) and set(spec) == {"rubric"}:
        rubric = [_text(item) for item in spec["rubric"]]
        if not rubric:
            raise ValueError("A score rubric needs at least one level")
        return Score(instructions=prompt, criteria=rubric)
    if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
        labels = [_text(item) for item in spec]
        if not labels:
            raise ValueError("A choice needs at least one option")
        return Choice(instructions=prompt, criteria={label: None for label in labels})
    if isinstance(spec, Mapping):
        if not spec:
            raise ValueError("A choice needs at least one option")
        return Choice(
            instructions=prompt,
            criteria={str(key): value for key, value in spec.items()},
        )
    raise TypeError(f"Unsupported question value {spec!r}")


def _is_question_mapping(spec: object) -> bool:
    if not isinstance(spec, Mapping) or not spec:
        return False
    if set(spec) <= {"yes", "no"} or set(spec) == {"rubric"}:
        return False
    if all(isinstance(value, str) or value is None for value in spec.values()):
        return False
    return True


def _text(value: object) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text:
            return text
    raise ValueError(f"Question labels must be non-empty strings, got {value!r}")
