"""Load RGB frames and optional depth maps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(image: object) -> np.ndarray:
    if isinstance(image, np.ndarray):
        array = np.asarray(image)
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        if array.shape[-1] == 4:
            array = array[:, :, :3]
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(array)
    path = Path(str(image))
    return np.asarray(Image.open(path).convert("RGB"))


def load_depth(depth: object | None) -> np.ndarray | None:
    if depth is None:
        return None
    if isinstance(depth, np.ndarray):
        return np.asarray(depth)
    return np.load(Path(str(depth)))
