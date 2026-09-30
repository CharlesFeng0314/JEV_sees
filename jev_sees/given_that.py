"""Build the hidden state sent to JEV and keep it inside the context budget."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from typing import Any

from .memory import SceneMemory
from .tracking import box_gap, box_iou

MAX_JEV_INPUT_TOKENS = 31_000
TARGET_PROMPT_CHARS = 24_000
MAX_OBJECTS = 16
MIN_OBJECTS = 4
NEAR_GAP_PX = 40.0

_OBJECT_KEYS = (
    "object_id",
    "label",
    "description",
    "attributes",
    "confidence",
    "bbox_xyxy",
    "centroid_uv",
    "position_m",
    "observation_count",
    "currently_visible",
    "stale",
    "pose_observation_count",
)


def public_object(item: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for key in _OBJECT_KEYS:
        if key not in item or item[key] is None:
            continue
        value = item[key]
        if key == "attributes" and isinstance(value, dict):
            color = value.get("color")
            if color:
                summary["attributes"] = {"color": color}
            continue
        if key == "confidence" and isinstance(value, (int, float)):
            summary[key] = round(float(value), 4)
            continue
        if key == "position_m" and isinstance(value, (list, tuple)):
            summary[key] = [round(float(part), 4) for part in value[:3]]
            continue
        if key in {"bbox_xyxy", "centroid_uv"} and isinstance(value, (list, tuple)):
            summary[key] = [round(float(part), 2) for part in value]
            continue
        if isinstance(value, (str, int, float, bool)):
            summary[key] = value
    return summary


def _centroid(item: dict[str, Any]) -> tuple[float, float] | None:
    centroid = item.get("centroid_uv")
    if isinstance(centroid, (list, tuple)) and len(centroid) >= 2:
        return float(centroid[0]), float(centroid[1])
    return None


def _previous_centroid(record: dict[str, Any]) -> tuple[float, float] | None:
    history = record.get("pose_history") or []
    if len(history) < 2:
        return None
    pose = history[-2].get("pose") or {}
    centroid = pose.get("centroid_uv")
    if isinstance(centroid, (list, tuple)) and len(centroid) >= 2:
        return float(centroid[0]), float(centroid[1])
    return None


def relations_for(objects: list[dict[str, Any]], memory: SceneMemory) -> list[dict[str, Any]]:
    """Geometric facts only. JEV decides whether two objects are in contact."""

    pairs = []
    records = memory.known_objects
    for index, left in enumerate(objects):
        for right in objects[index + 1 :]:
            left_box = left.get("bbox_xyxy")
            right_box = right.get("bbox_xyxy")
            if not left_box or not right_box:
                continue
            iou = box_iou(left_box, right_box)
            gap = box_gap(left_box, right_box)
            if iou <= 0.0 and gap > NEAR_GAP_PX:
                continue
            left_centroid = _centroid(left)
            right_centroid = _centroid(right)
            distance = None
            if left_centroid and right_centroid:
                distance = math.dist(left_centroid, right_centroid)
            approaching = None
            previous_left = _previous_centroid(records.get(str(left.get("object_id")), {}))
            previous_right = _previous_centroid(records.get(str(right.get("object_id")), {}))
            if previous_left and previous_right and left_centroid and right_centroid and distance is not None:
                previous_distance = math.dist(previous_left, previous_right)
                approaching = distance + 2.0 < previous_distance
            fact: dict[str, Any] = {
                "object_a": left.get("object_id"),
                "object_b": right.get("object_id"),
                "bbox_iou": round(iou, 4),
                "box_gap_px": round(gap, 2),
            }
            if distance is not None:
                fact["centroid_distance_px"] = round(distance, 2)
            if approaching is not None:
                fact["approaching"] = approaching
            left_position = left.get("position_m")
            right_position = right.get("position_m")
            if (
                isinstance(left_position, (list, tuple))
                and isinstance(right_position, (list, tuple))
                and len(left_position) == 3
                and len(right_position) == 3
            ):
                fact["gap_m"] = round(math.dist(left_position, right_position), 4)
            pairs.append(fact)
    pairs.sort(key=lambda item: float(item.get("centroid_distance_px") or item.get("box_gap_px") or 0.0))
    return pairs


def uncertainty_for(objects: list[dict[str, Any]]) -> list[str]:
    notes = []
    if not objects:
        notes.append("No object is visible in the current frame.")
    stale = [item["object_id"] for item in objects if item.get("stale")]
    if stale:
        notes.append("Stale remembered objects: " + ", ".join(stale))
    weak = [
        item["object_id"]
        for item in objects
        if item.get("currently_visible") and float(item.get("confidence") or 0.0) < 0.2
    ]
    if weak:
        notes.append("Low-confidence visible objects: " + ", ".join(weak))
    return notes


def _serialized_chars(state: dict[str, Any]) -> int:
    return len(json.dumps(state, ensure_ascii=False, separators=(",", ":"), default=str))


def fit_prompt_budget(state: dict[str, Any]) -> dict[str, Any]:
    """Keep the call under JEV's 31k-token hard limit.

    Serialized characters are the conservative token estimate used by the
    robot project: one character is counted as one token.
    """

    bounded = deepcopy(state)
    known = list((bounded.get("scene_memory") or {}).get("known_objects") or [])
    visible = list((bounded.get("current_scene") or {}).get("visible_objects") or [])
    pairs = list(bounded.get("relations") or [])
    while _serialized_chars(bounded) > TARGET_PROMPT_CHARS and len(pairs) > 0:
        pairs.pop()
        bounded["relations"] = pairs
    while _serialized_chars(bounded) > TARGET_PROMPT_CHARS and len(known) > MIN_OBJECTS:
        dropped = known.pop()
        bounded["scene_memory"]["known_objects"] = known
        visible = [item for item in visible if item.get("object_id") != dropped.get("object_id")]
        bounded["current_scene"]["visible_objects"] = visible
    if _serialized_chars(bounded) > TARGET_PROMPT_CHARS:
        for collection in (known, visible):
            for item in collection:
                item.pop("description", None)
    chars = _serialized_chars(bounded)
    bounded["prompt_budget"] = {
        "hard_limit_tokens": MAX_JEV_INPUT_TOKENS,
        "target_char_budget": TARGET_PROMPT_CHARS,
        "serialized_chars": chars,
        "conservative_token_estimate": chars,
    }
    return bounded


def build_given_that(
    prompt: str,
    memory: SceneMemory,
    *,
    modality: str,
    image_size: tuple[int, int] | None,
) -> dict[str, Any]:
    remembered = [public_object(item) for item in memory.prompt_objects()][:MAX_OBJECTS]
    visible = [item for item in remembered if item.get("currently_visible")][:MAX_OBJECTS]
    width = image_size[0] if image_size else None
    height = image_size[1] if image_size else None
    state = {
        "schema_version": 1,
        "user_goal": prompt,
        "camera": {"modality": modality, "image_width": width, "image_height": height},
        "current_scene": {"visible_objects": visible},
        "scene_memory": {"revision": memory.revision, "known_objects": remembered},
        "relations": relations_for(visible, memory),
        "uncertainty": uncertainty_for(remembered),
        "memory_semantics": (
            "Persistent camera belief. A new observation is merged by object_id. "
            "JEV does not see pixels; these facts are the only scene evidence."
        ),
    }
    return fit_prompt_budget(state)
