"""Coarse color names from segmented pixels."""

from __future__ import annotations

import numpy as np


def dominant_color(rgb: np.ndarray, mask: np.ndarray | None = None) -> str:
    pixels = rgb if mask is None else rgb[mask]
    if len(pixels) == 0:
        return "unknown"
    values = pixels.reshape(-1, 3).astype(np.float64) / 255.0
    maximum = values.max(axis=1)
    minimum = values.min(axis=1)
    saturation = (maximum - minimum) / np.maximum(maximum, 1e-6)
    chromatic = values[(saturation > 0.28) & (maximum > 0.16)]
    if len(chromatic) >= max(12, len(values) // 12):
        red, green, blue = np.median(chromatic, axis=0)
        if red > 1.22 * max(green, blue):
            return "red"
        if green > 1.18 * max(red, blue):
            return "green"
        if blue > 1.18 * max(red, green):
            return "blue"
        if red > 0.75 and green > 0.55 and blue < 0.45:
            return "yellow"
        if red > 0.7 and green > 0.35 and blue < 0.35:
            return "orange"
    luminance = float(np.median(maximum))
    if luminance > 0.78:
        return "white"
    if luminance < 0.22:
        return "black"
    return "gray"
