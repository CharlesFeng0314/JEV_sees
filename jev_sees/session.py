"""Public visual session and optional official JEV calls."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .given_that import build_given_that
from .io import load_depth, load_rgb
from .memory import SceneMemory
from .render import DashboardRenderer, annotate, write_visual
from .report import JudgmentLog, write_excel, write_json
from .result import Result
from .runtime import DEFAULT_FLORENCE_MODEL
from .tracking import TrackBank

_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

DEFAULT_MODEL = "jev-latest"
DEFAULT_BASE_URL = "https://api.typesafe.ai"

QuestionFactory = Callable[[list[dict[str, Any]]], Mapping[str, Any]]
QuestionSource = Mapping[str, Any] | QuestionFactory


class Sees:
    """Discover visible objects, track them, and build official JEV state."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
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

        ``questions`` must already contain official JEV question objects, or be
        a callable that creates them from the current tracks. Sees never infers
        a question type or constructs Choice/Noul/Score objects.
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
            write_json(json, result.question, result.rows, result.timeline, result.frames)
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
        if response is not None:
            log.add(None, tracks, response.answers)
        if save is not None:
            answers = {} if response is None else response.answers
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
        return _result(question, log, response)

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
                        answers = response.answers
                    else:
                        answers = {}
                    log.add(index, tracks, answers, video_time_s=video_time_s)
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
        return _result(question, log, last)

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
    questions = source(tracks) if callable(source) else source
    return _official_questions(questions, allow_empty=True)


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


def _result(question: str, log: JudgmentLog, response: Result | None) -> Result:
    raw = None if response is None else response.response
    return Result(
        raw,
        rows=log.rows(),
        timeline=log.export_timeline(),
        frames=log.export_frames(),
        question=question,
    )


def _is_video(source: object) -> bool:
    if isinstance(source, (str, Path)):
        return Path(source).suffix.lower() in _VIDEO_SUFFIXES
    return False


def _risks(answers: dict[str, Any]) -> dict[str, float]:
    risks = {}
    for key, answer in answers.items():
        value = getattr(answer, "noul", None)
        if value is not None:
            risks[str(key)] = float(value)
    return risks


def _captions(answers: dict[str, Any]) -> dict[str, str]:
    captions = {}
    for key, answer in answers.items():
        choice = getattr(answer, "choice", None)
        if choice is not None:
            captions[str(key)] = str(choice)
    return captions


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
