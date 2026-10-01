"""Coarse color names from segmented pixels."""

from __future__ import annotations

import colorsys

import numpy as np


COLOR_NAMES = ("red", "orange", "yellow", "green", "blue", "white", "black", "gray")


def dominant_color(rgb: np.ndarray, mask: np.ndarray | None = None) -> str:
    """Backward-compatible name for the CV estimate."""

    return str(cv_color_evidence(rgb, mask)["label"])


def cv_color_evidence(rgb: np.ndarray, mask: np.ndarray | None = None) -> dict[str, object]:
    """Return a named estimate plus the pixel measurements behind it."""

    pixels = rgb.reshape(-1, 3) if mask is None else rgb[mask]
    if len(pixels) == 0:
        return {
            "label": "unknown",
            "median_rgb": None,
            "pixel_count": 0,
            "chromatic_fraction": 0.0,
            "median_saturation": 0.0,
            "median_luminance": 0.0,
        }
    values = pixels.reshape(-1, 3).astype(np.float64) / 255.0
    maximum = values.max(axis=1)
    minimum = values.min(axis=1)
    saturation = (maximum - minimum) / np.maximum(maximum, 1e-6)
    chromatic_mask = (saturation > 0.28) & (maximum > 0.16)
    chromatic = values[chromatic_mask]
    enough_chromatic = len(chromatic) >= max(12, len(values) // 12)
    representative = chromatic if enough_chromatic else values
    red, green, blue = np.median(representative, axis=0)
    luminance = float(np.median(maximum))
    hue, representative_saturation, _value = colorsys.rgb_to_hsv(red, green, blue)
    hue_degrees = hue * 360.0
    if luminance < 0.22:
        label = "black"
    elif representative_saturation < 0.2 and luminance > 0.78:
        label = "white"
    elif representative_saturation < 0.2:
        label = "gray"
    elif hue_degrees < 15.0 or hue_degrees >= 345.0:
        label = "red"
    elif hue_degrees < 45.0:
        label = "orange"
    elif hue_degrees < 75.0:
        label = "yellow"
    elif hue_degrees < 165.0:
        label = "green"
    else:
        label = "blue"
    return {
        "label": label,
        "median_rgb": [int(round(channel * 255.0)) for channel in (red, green, blue)],
        "pixel_count": int(len(values)),
        "chromatic_fraction": round(float(np.mean(chromatic_mask)), 4),
        "hue_degrees": round(hue_degrees, 2),
        "median_saturation": round(float(np.median(saturation)), 4),
        "median_luminance": round(luminance, 4),
    }
