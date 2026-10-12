"""Public visual session and optional official JEV calls."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .given_that import build_given_that
from .io import load_depth, load_rgb
from .memory import SceneMemory
from .render import DashboardRenderer, write_visual
from .report import JudgmentLog, write_excel, write_json
from .result import Result
from .runtime import DEFAULT_FLORENCE_MODEL
from .tracking import TrackBank

_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
_SCOPE_PREFIX = "__scope__:"


@dataclass(frozen=True)
class _ObjectQuestionPlan:
    question: str
    kind: str
    criteria: Any = None
    semantic_class: str | None = None
    scope_threshold: float = 0.5

    def questions(self, tracks: list[dict[str, Any]]) -> dict[str, Any]:
        from typesafe_sdk import Choice, Noul, Score

        questions: dict[str, Any] = {}
        for track in tracks:
            object_id = str(track.get("object_id") or "")
            if not object_id:
                continue
            if self.semantic_class is not None:
                questions[_scope_key(object_id)] = Noul(
                    instructions={
                        "task": "semantic object class routing",
                        "object_id": object_id,
                        "semantic_class": self.semantic_class,
                        "question": (
                            "Does the referenced object belong to the semantic class "
                            "in this scene? Use its visual label and scene facts."
                        ),
                    }
                )
            instructions = {
                "question": self.question,
                "object_id": object_id,
                "reference": (
                    "Answer about the current_scene.visible_objects entry whose "
                    "object_id matches this value."
                ),
            }
            if self.kind == "probability":
                questions[object_id] = Noul(instructions=instructions)
            elif self.kind == "choice":
                questions[object_id] = Choice(instructions=instructions, criteria=self.criteria)
            elif self.kind == "score":
                questions[object_id] = Score(instructions=instructions, criteria=self.criteria)
            else:  # pragma: no cover - constructed only by ObjectQuestions
                raise ValueError(f"Unknown object question kind: {self.kind}")
        return questions

    def answers(
        self,
        tracks: list[dict[str, Any]],
        answers: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, float]]:
        selected: dict[str, Any] = {}
        scope_probabilities: dict[str, float] = {}
        for track in tracks:
            object_id = str(track.get("object_id") or "")
            if not object_id or object_id not in answers:
                continue
            if self.semantic_class is None:
                selected[object_id] = answers[object_id]
                continue
            scope_answer = answers.get(_scope_key(object_id))
            probability = getattr(scope_answer, "noul", None)
            if probability is None:
                continue
            value = float(probability)
            scope_probabilities[object_id] = value
            if value >= self.scope_threshold:
                selected[object_id] = answers[object_id]
        return selected, scope_probabilities


@dataclass(frozen=True)
class ObjectQuestions:
    """Typed per-object questions, optionally routed through a semantic class."""

    sees: "Sees"
    semantic_class: str | None = None
    scope_threshold: float = 0.5

    def probability(self, source: object, question: str, **kwargs: Any) -> Result:
        return self._run(source, question, kind="probability", **kwargs)

    def choice(
        self,
        source: object,
        question: str,
        *,
        choices: Mapping[str, Any] | Sequence[str],
        **kwargs: Any,
    ) -> Result:
        criteria = _choice_criteria(choices)
        return self._run(source, question, kind="choice", criteria=criteria, **kwargs)

    def score(
        self,
        source: object,
        question: str,
        *,
        rubric: Sequence[Any],
        **kwargs: Any,
    ) -> Result:
        criteria = list(rubric)
        if not criteria:
            raise ValueError("rubric must contain at least one score level")
        return self._run(source, question, kind="score", criteria=criteria, **kwargs)

    def _run(
        self,
        source: object,
        question: str,
        *,
        kind: str,
        criteria: Any = None,
        **kwargs: Any,
    ) -> Result:
        text = " ".join(str(question).split())
        if not text:
            raise ValueError("question must not be empty")
        plan = _ObjectQuestionPlan(
            question=text,
            kind=kind,
            criteria=criteria,
            semantic_class=self.semantic_class,
            scope_threshold=self.scope_threshold,
        )
        return self.sees(source, text, plan, **kwargs)


QuestionFactory = Callable[[list[dict[str, Any]]], Mapping[str, Any]]
QuestionSource = Mapping[str, Any] | QuestionFactory | _ObjectQuestionPlan


class Sees:
    """Discover visible objects, track them, and build official JEV state."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str | None = None,
        base_url: str | None = None,
        florence_model: str = DEFAULT_FLORENCE_MODEL,
        clip_model: str | None = None,
        device: str | None = None,
        perceptor: Any = None,
        client: Any = None,
    ):
        self.api_key = api_key or _api_key_from_env()
        self.model = model
        self.base_url = base_url
        self._client = client
        self._florence_model = florence_model
        self._clip_model = clip_model
        self._device = device
        self._runtime: Any = perceptor
        self.memory = SceneMemory()
        self.tracks = TrackBank("image")
        self.modality = "rgb"
        self._modality_locked = False
        self.image_size: tuple[int, int] | None = None
        self.last_prompt_chars = 0
        self.max_prompt_chars = 0

    @property
    def objects(self) -> ObjectQuestions:
        """Ask one typed question for every object detected in each frame."""

        return ObjectQuestions(self)

    def for_each(
        self,
        semantic_class: str,
        *,
        scope_threshold: float = 0.5,
    ) -> ObjectQuestions:
        """Ask about every object JEV maps to ``semantic_class``.

        Class matching is another typed JEV decision in the same request. The
        SDK never compares labels or maintains a synonym table.
        """

        name = " ".join(str(semantic_class).split())
        if not name:
            raise ValueError("semantic_class must not be empty")
        threshold = float(scope_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("scope_threshold must be between 0 and 1")
        return ObjectQuestions(self, semantic_class=name, scope_threshold=threshold)

    def probability(self, source: object, question: str, **kwargs: Any) -> Result:
        """Return an independent probability for every detected object."""

        return self.objects.probability(source, question, **kwargs)

    def choice(
        self,
        source: object,
        question: str,
        *,
        choices: Mapping[str, Any] | Sequence[str],
        **kwargs: Any,
    ) -> Result:
        """Return a caller-defined choice distribution for every object."""

        return self.objects.choice(source, question, choices=choices, **kwargs)

    def score(
        self,
        source: object,
        question: str,
        *,
        rubric: Sequence[Any],
        **kwargs: Any,
    ) -> Result:
        """Return a caller-defined score for every detected object."""

        return self.objects.score(source, question, rubric=rubric, **kwargs)

    def __call__(
        self,
        source: object,
        question: str,
        questions: QuestionSource,
        depth: object | None = None,
        *,
        intrinsics: dict[str, float] | None = None,
        save: str | Path | None = None,
        json: str | Path | None = None,
        excel: str | Path | None = None,
        every: int | None = None,
        start: int | None = None,
        end: int | None = None,
    ) -> Result:
        """Observe an image/video and ask application-supplied JEV questions.

        The low-level path accepts official JEV question objects or a callable
        that creates them from current tracks. ``probability()``, ``choice()``,
        ``score()``, and ``for_each()`` build the same objects automatically.
        """

        self._begin_call()
        if _is_video(source):
            result = self._call_video(
                Path(source),
                question,
                questions,
                save=save,
                every=every,
                start=start,
                end=end,
            )
        else:
            result = self._call_image(
                source,
                question,
                questions,
                depth=depth,
                intrinsics=intrinsics,
                save=save,
            )
        print(result)
        if json is not None:
            write_json(
                json,
                result.question,
                result.rows,
                result.timeline,
                result.frames,
                result.scope,
            )
        if excel is not None:
            write_excel(excel, result.rows, result.timeline, result.frames)
        return result

    def observe(
        self,
        image: object,
        depth: object | None = None,
        *,
        intrinsics: dict[str, float] | None = None,
        frame_index: int | None = None,
        video_time_s: float | None = None,
    ) -> list[dict[str, Any]]:
        """Discover this frame, keep object ids stable, and update memory."""

        rgb = load_rgb(image)
        depth_map = load_depth(depth)
        mode = "camera" if depth_map is not None else "image"
        if self._modality_locked and mode != self.tracks.mode:
            raise ValueError(f"This Sees stream is already {self.modality}")
        self._modality_locked = True
        if self.tracks.mode != mode:
            self.tracks = TrackBank(mode)
        self.modality = "rgbd" if mode == "camera" else "rgb"
        self.image_size = (int(rgb.shape[1]), int(rgb.shape[0]))
        runtime = self._vision()
        if mode == "camera":
            from .perceive import perceive_rgbd

            observations = perceive_rgbd(runtime, rgb, depth_map, intrinsics)
        else:
            from .perceive import perceive_rgb

            observations = perceive_rgb(runtime, rgb)
        tracked = self.tracks.update(observations)
        self.memory.observe(tracked, frame_index=frame_index, video_time_s=video_time_s)
        return tracked

    def ask(self, prompt: str, questions: Mapping[str, Any]) -> Result:
        """Call JEV with official question objects supplied by the application."""

        official = _official_questions(questions)
        state = self.state(prompt)
        self.last_prompt_chars = int(state["prompt_budget"]["serialized_chars"])
        self.max_prompt_chars = max(self.max_prompt_chars, self.last_prompt_chars)
        response = self._system_one(state, official)
        return Result(response)

    def state(self, prompt: str) -> dict[str, Any]:
        """Return accumulated visual facts as official JEV ``state``."""

        return build_given_that(
            prompt,
            self.memory,
            modality=self.modality,
            image_size=self.image_size,
        )

    def _begin_call(self) -> None:
        self.memory = SceneMemory()
        self.tracks = TrackBank("image")
        self.modality = "rgb"
        self._modality_locked = False
        self.image_size = None
        self.last_prompt_chars = 0
        self.max_prompt_chars = 0

    def _call_image(
        self,
        source: object,
        question: str,
        questions: QuestionSource,
        *,
        depth: object | None,
        intrinsics: dict[str, float] | None,
        save: str | Path | None,
    ) -> Result:
        rgb = load_rgb(source)
        tracks = self.observe(rgb, depth, intrinsics=intrinsics)
        log = JudgmentLog()
        log.see(tracks)
        response = self._ask_if_ready(question, _questions_for(questions, tracks))
        answers, scope_probabilities = _answers_for(questions, tracks, response)
        if response is not None:
            log.add(
                None,
                tracks,
                answers,
                scope=_scope_metadata(questions, scope_probabilities),
            )
        if save is not None:
            dashboard = DashboardRenderer(panel_title="JEV OUTPUT // STRUCTURED ANSWER")
            framed = dashboard.render(
                rgb,
                tracks,
                answers,
                frame_index=0,
                video_time_s=0.0,
                state_chars=self.last_prompt_chars or None,
            )
            write_visual([framed], save, fps=1)
        return _result(question, log, response, answers, questions)

    def _call_video(
        self,
        path: Path,
        question: str,
        questions: QuestionSource,
        *,
        save: str | Path | None,
        every: int | None,
        start: int | None,
        end: int | None,
    ) -> Result:
        import cv2
        import numpy as np

        if start is not None and end is not None and start > end:
            raise ValueError("start must be less than or equal to end")
        if every is not None and every < 1:
            raise ValueError("every must be at least 1")
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Could not open {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 12.0
        stride = every if every is not None else max(1, int(round(fps / 2.0)))
        origin = 0 if start is None else int(start)
        if start is not None:
            capture.set(cv2.CAP_PROP_POS_FRAMES, origin)
        index = origin
        log = JudgmentLog()
        frames: list[Any] = []
        dashboard = DashboardRenderer() if save is not None else None
        last: Result | None = None
        last_answers: dict[str, Any] = {}
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if end is not None and index > end:
                    break
                if (index - origin) % stride == 0:
                    rgb = np.ascontiguousarray(frame[:, :, ::-1])
                    video_time_s = index / fps
                    tracks = self.observe(
                        rgb,
                        frame_index=index,
                        video_time_s=video_time_s,
                    )
                    log.see(tracks)
                    current_questions = _questions_for(questions, tracks)
                    self.last_prompt_chars = 0
                    response = self._ask_if_ready(question, current_questions)
                    if response is not None:
                        last = response
                    answers, scope_probabilities = _answers_for(questions, tracks, response)
                    last_answers = answers
                    log.add(
                        index,
                        tracks,
                        answers,
                        video_time_s=video_time_s,
                        scope=_scope_metadata(questions, scope_probabilities),
                    )
                    if save is not None:
                        assert dashboard is not None
                        frames.append(
                            dashboard.render(
                                rgb,
                                tracks,
                                answers,
                                frame_index=index,
                                video_time_s=video_time_s,
                                state_chars=self.last_prompt_chars or None,
                            )
                        )
                index += 1
        finally:
            capture.release()
        if save is not None:
            if not frames:
                raise ValueError(f"No sampled frame to write to {save}")
            write_visual(frames, save, fps=max(0.1, fps / stride))
        return _result(question, log, last, last_answers, questions)

    def _ask_if_ready(self, question: str, questions: dict[str, Any]) -> Result | None:
        if not questions:
            return None
        return self.ask(question, questions)

    def _vision(self) -> Any:
        if self._runtime is None:
            from .runtime import VisionRuntime

            self._runtime = VisionRuntime(
                florence_model=self._florence_model,
                clip_path=self._clip_model,
                device=self._device,
            )
        return self._runtime

    def _system_one(self, state: dict[str, Any], questions: dict[str, Any]) -> Any:
        if self._client is not None:
            return self._client(state, questions)
        if not self.api_key:
            raise ValueError("Pass api_key or set TYPESAFE_API_KEY")
        from typesafe_sdk import TypeSafeClient

        with TypeSafeClient(api_key=self.api_key, model=self.model, base_url=self.base_url) as client:
            return client.system_one(state=state, questions=questions)


def _questions_for(source: QuestionSource, tracks: list[dict[str, Any]]) -> dict[str, Any]:
    if isinstance(source, _ObjectQuestionPlan):
        return source.questions(tracks)
    questions = source(tracks) if callable(source) else source
    return _official_questions(questions, allow_empty=True)


def _choice_criteria(choices: Mapping[str, Any] | Sequence[str]) -> dict[str, Any]:
    if isinstance(choices, Mapping):
        criteria = {str(name): description for name, description in choices.items()}
    elif isinstance(choices, Sequence) and not isinstance(choices, (str, bytes)):
        criteria = {str(name): None for name in choices}
    else:
        raise TypeError("choices must be a mapping or a sequence of names")
    criteria = {name: description for name, description in criteria.items() if name.strip()}
    if not criteria:
        raise ValueError("choices must contain at least one named option")
    return criteria


def _scope_key(object_id: str) -> str:
    return f"{_SCOPE_PREFIX}{object_id}"


def _scope_metadata(
    source: QuestionSource,
    probabilities: Mapping[str, float],
) -> dict[str, Any] | None:
    if not isinstance(source, _ObjectQuestionPlan) or source.semantic_class is None:
        return None
    scope: dict[str, Any] = {
        "class": source.semantic_class,
        "threshold": source.scope_threshold,
    }
    if probabilities:
        scope["probabilities"] = {
            str(object_id): float(probability)
            for object_id, probability in probabilities.items()
        }
    return scope


def _answers_for(
    source: QuestionSource,
    tracks: list[dict[str, Any]],
    response: Result | None,
) -> tuple[dict[str, Any], dict[str, float]]:
    if response is None:
        return {}, {}
    if isinstance(source, _ObjectQuestionPlan):
        return source.answers(tracks, response.answers)
    return response.answers, {}


def _official_questions(
    questions: Mapping[str, Any],
    *,
    allow_empty: bool = False,
) -> dict[str, Any]:
    if not isinstance(questions, Mapping):
        raise TypeError("questions must map names to official JEV question objects")
    official = {str(key): value for key, value in questions.items()}
    if not official and not allow_empty:
        raise ValueError("questions must contain at least one official JEV question")
    return official


def _result(
    question: str,
    log: JudgmentLog,
    response: Result | None,
    answers: dict[str, Any],
    source: QuestionSource,
) -> Result:
    raw = None if response is None else response.response
    scope = _scope_metadata(source, {})
    return Result(
        raw,
        answers=answers,
        rows=log.rows(),
        timeline=log.export_timeline(),
        frames=log.export_frames(),
        question=question,
        scope=scope,
    )


def _is_video(source: object) -> bool:
    if isinstance(source, (str, Path)):
        return Path(source).suffix.lower() in _VIDEO_SUFFIXES
    return False


def _api_key_from_env() -> str:
    value = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if value:
        return value
    env_path = Path.cwd() / ".env"
    if not env_path.is_file():
        return ""
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        prefix = "TYPESAFE_API_KEY="
        if line.startswith(prefix):
            return line[len(prefix) :].strip().strip('"').strip("'")
    return ""
