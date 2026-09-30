"""Keep a stable object id across frames.

RGB tracks match pixel boxes by IoU. RGB-D tracks match camera-frame positions
with the same radius rule as the robot wrist tracker.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

CACHE_MATCH_RADIUS_M = 0.12
MAX_TRACK_MATCH_RADIUS_M = 0.20
MIN_IOU = 0.30


def box_iou(left: list[float], right: list[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    return intersection / max(left_area + right_area - intersection, 1e-9)


def box_gap(left: list[float], right: list[float]) -> float:
    if left[2] < right[0]:
        dx = right[0] - left[2]
    elif right[2] < left[0]:
        dx = left[0] - right[2]
    else:
        dx = 0.0
    if left[3] < right[1]:
        dy = right[1] - left[3]
    elif right[3] < left[1]:
        dy = left[1] - right[3]
    else:
        dy = 0.0
    return math.hypot(dx, dy)


def _position(item: dict[str, Any]) -> list[float] | None:
    position = item.get("position_m")
    if isinstance(position, (list, tuple)) and len(position) == 3:
        return [float(value) for value in position]
    return None


def _track_match_radius(current: dict[str, Any], previous: dict[str, Any]) -> float:
    spans = []
    for item in (current, previous):
        position = _position(item)
        bbox = item.get("bbox3d_m")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 2:
            low, high = bbox
            spans.append(math.dist([float(v) for v in low[:2]], [float(v) for v in high[:2]]))
        elif position is not None:
            spans.append(0.0)
    adaptive = 0.75 * max(spans, default=0.0)
    return min(MAX_TRACK_MATCH_RADIUS_M, max(CACHE_MATCH_RADIUS_M, adaptive))


class TrackBank:
    """Assign ``object_###`` ids. Unmatched history stays available for re-entry."""

    def __init__(self, mode: str = "image"):
        if mode not in {"image", "camera"}:
            raise ValueError("mode must be 'image' or 'camera'")
        self.mode = mode
        self.tracks: dict[str, dict[str, Any]] = {}
        self._next_id = 1

    def update(self, observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        current = [dict(item) for item in observations]
        previous = list(self.tracks.items())
        assignments: dict[int, str] = {}
        if current and previous:
            costs = np.full((len(current), len(previous)), 1e6, dtype=np.float64)
            for row, observation in enumerate(current):
                for col, (_object_id, old) in enumerate(previous):
                    costs[row, col] = self._cost(observation, old)
            rows, cols = linear_sum_assignment(costs)
            for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
                if self._accept(costs[row, col], current[row], previous[col][1]):
                    assignments[row] = previous[col][0]
        tracked = []
        for index, observation in enumerate(current):
            object_id = assignments.get(index)
            if object_id is None:
                object_id = f"object_{self._next_id:03d}"
                self._next_id += 1
            observation["object_id"] = object_id
            self.tracks[object_id] = observation
            tracked.append(observation)
        return tracked

    def _cost(self, current: dict[str, Any], previous: dict[str, Any]) -> float:
        if self.mode == "image":
            left = current.get("bbox_xyxy")
            right = previous.get("bbox_xyxy")
            if not left or not right:
                return 1e6
            return 1.0 - box_iou(left, right)
        current_position = _position(current)
        previous_position = _position(previous)
        if current_position is None or previous_position is None:
            return 1e6
        return math.dist(current_position, previous_position)

    def _accept(self, cost: float, current: dict[str, Any], previous: dict[str, Any]) -> bool:
        if self.mode == "image":
            return (1.0 - cost) >= MIN_IOU
        return cost <= _track_match_radius(current, previous)
