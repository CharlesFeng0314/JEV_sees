"""YOLO-style entry point: one object, then call it."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .given_that import build_given_that
from .io import load_depth, load_rgb
from .memory import SceneMemory
from .questions import compile_questions
from .result import Result
from .tracking import TrackBank
from .vocabulary import COCO_CLASSES

DEFAULT_MODEL = "jev-latest"
DEFAULT_BASE_URL = "https://api.typesafe.ai"


class Sees:
    """Watch a camera and ask JEV a closed question about what it sees.

    ``typesafe-sdk`` stays inside this package. Pass an API key, or set
    ``TYPESAFE_API_KEY``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        vocabulary: list[str] | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        yolo_model: str | None = None,
        clip_model: str | None = None,
        client: Any = None,
    ):
        self.api_key = api_key or _api_key_from_env()
        self.model = model
        self.base_url = base_url
        self._client = client
        self.vocabulary = list(vocabulary or COCO_CLASSES)
        self._yolo_model = yolo_model
        self._clip_model = clip_model
        self._runtime: Any = None
        self.memory = SceneMemory()
        self.tracks = TrackBank("image")
        self.modality = "rgb"
        self._modality_locked = False
        self.image_size: tuple[int, int] | None = None

    def __call__(
        self,
        image: object,
        prompt: str,
        questions: object,
        depth: object | None = None,
        *,
        intrinsics: dict[str, float] | None = None,
    ) -> Result:
        self.observe(image, depth, intrinsics=intrinsics)
        return self.ask(prompt, questions)

    def observe(
        self,
        image: object,
        depth: object | None = None,
        *,
        intrinsics: dict[str, float] | None = None,
        vocabulary: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Detect this frame, keep object ids stable, and update scene memory."""

        if vocabulary is not None:
            self.vocabulary = list(vocabulary)
            if self._runtime is not None:
                self._runtime.set_vocabulary(self.vocabulary)
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
        self.memory.observe(tracked)
        return tracked

    def ask(self, prompt: str, questions: object) -> Result:
        """Ask JEV about the memory accumulated so far."""

        state = build_given_that(
            prompt,
            self.memory,
            modality=self.modality,
            image_size=self.image_size,
        )
        compiled = compile_questions(questions, prompt)
        response = self._system_one(state, compiled)
        return Result(response)

    def _vision(self) -> Any:
        if self._runtime is None:
            from .runtime import VisionRuntime

            self._runtime = VisionRuntime(
                yolo_path=self._yolo_model,
                clip_path=self._clip_model,
                vocabulary=self.vocabulary,
            )
        self._runtime.set_vocabulary(self.vocabulary)
        return self._runtime

    def _system_one(self, state: dict[str, Any], questions: dict[str, Any]) -> Any:
        if self._client is not None:
            return self._client(state, questions)
        if not self.api_key:
            raise ValueError("Pass api_key or set TYPESAFE_API_KEY")
        from typesafe_sdk import TypeSafeClient

        with TypeSafeClient(api_key=self.api_key, model=self.model, base_url=self.base_url) as client:
            return client.system_one(state=state, questions=questions)


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
