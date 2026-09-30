"""Locate local model weights without committing them to the repository."""

from __future__ import annotations

import os
from pathlib import Path


def robo_root() -> Path | None:
    env = os.environ.get("JEV_SEES_ROBO_ROOT", "").strip()
    if env:
        path = Path(env)
        return path if path.is_dir() else None
    candidate = Path(r"H:\robo")
    return candidate if candidate.is_dir() else None


def yolo_weight(size: str = "s") -> str:
    root = robo_root()
    if root is not None:
        path = root / "models" / "benchmark" / f"yolov8{size}-worldv2.pt"
        if path.is_file():
            return str(path)
    return f"yolov8{size}-worldv2.pt"


def clip_weight() -> str:
    root = robo_root()
    if root is not None:
        path = root / "weights" / "clip" / "ViT-B-32.pt"
        if path.is_file():
            return str(path)
    return "ViT-B/32"


def mobile_sam_weight() -> str | None:
    root = robo_root()
    if root is None:
        return None
    path = root / "models" / "benchmark" / "mobile_sam.pt"
    return str(path) if path.is_file() else None
