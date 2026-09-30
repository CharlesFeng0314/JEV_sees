"""RGB and RGB-D observation builders."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from .color import dominant_color
from .geometry import cluster_depth, default_intrinsics, depth_to_meters, mask_facts
from .runtime import VisionRuntime
from .tracking import box_iou

YOLO_WEIGHT = 2.0


def perceive_rgb(runtime: VisionRuntime, rgb: np.ndarray, *, confidence: float = 0.25) -> list[dict[str, Any]]:
    detections = runtime.detect(rgb, confidence=confidence)
    crops = [_crop(rgb, item["bbox_xyxy"]) for item in detections]
    colors = runtime.color_names(crops) if crops else []
    observations = []
    for detection, color in zip(detections, colors, strict=True):
        x1, y1, x2, y2 = [int(round(value)) for value in detection["bbox_xyxy"]]
        heuristic = dominant_color(rgb[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)])
        # CLIP names a blue bus green on the official crop. A chromatic body color wins.
        named = heuristic if heuristic not in {"unknown", "gray"} else color
        observations.append(
            {
                "label": detection["label"],
                "confidence": detection["score"],
                "bbox_xyxy": [x1, y1, x2, y2],
                "centroid_uv": [round((x1 + x2) / 2.0, 2), round((y1 + y2) / 2.0, 2)],
                "attributes": {"color": named, "dominant_color": heuristic, "clip_color": color},
                "description": f"{named} {detection['label']}",
                "source": "rgb_yolo_world_clip",
            }
        )
    return observations


def perceive_rgbd(
    runtime: VisionRuntime,
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: dict[str, float] | None = None,
    *,
    near_m: float = 0.15,
    far_m: float = 2.0,
    yolo_confidence: float = 0.02,
) -> list[dict[str, Any]]:
    """Depth clusters decide which objects exist. CLIP names them. YOLO is evidence."""

    height, width = rgb.shape[:2]
    calibration = intrinsics or default_intrinsics(width, height)
    depth_m = depth_to_meters(depth)
    if depth_m.shape[:2] != (height, width):
        raise ValueError("Depth shape does not match the RGB frame")
    masks = cluster_depth(depth_m, calibration, near_m=near_m, far_m=far_m)
    prepared = []
    crops = []
    for mask in masks:
        facts = mask_facts(mask, depth_m, calibration)
        if facts is None:
            continue
        prepared.append((mask, facts))
        crops.append(_masked_crop(rgb, mask, facts["bbox_xyxy"]))
    class_scores = runtime.classify_crops(crops) if crops else []
    detections = runtime.detect(rgb, confidence=yolo_confidence)
    _attach_detector(prepared, detections)
    colors = runtime.color_names(crops) if crops else []
    observations = []
    for (mask, facts), scores, color in zip(prepared, class_scores, colors, strict=True):
        if not scores:
            continue
        label = max(scores, key=scores.get)
        total = sum(scores.values()) or 1.0
        normalized = {name: value / total for name, value in scores.items()}
        detector_label = facts.get("detector_label")
        if detector_label in normalized:
            boost = np.zeros(len(normalized), dtype=np.float64)
            names = list(normalized)
            boost[names.index(detector_label)] = YOLO_WEIGHT * float(facts.get("detector_score") or 0.0)
            base = np.asarray([normalized[name] for name in names], dtype=np.float64)
            fused = base + boost
            fused = fused / max(float(fused.sum()), 1e-9)
            label = names[int(np.argmax(fused))]
            normalized = {name: float(fused[index]) for index, name in enumerate(names)}
        edge = _touches_edge(facts["bbox_xyxy"], width, height)
        observations.append(
            {
                "label": label,
                "confidence": float(normalized[label]),
                "semantic_scores": normalized,
                "bbox_xyxy": facts["bbox_xyxy"],
                "centroid_uv": facts["centroid_uv"],
                "position_m": facts["position_m"],
                "attributes": {"color": color},
                "description": f"{color} {label}",
                "partial_view": edge,
                "detector_label": facts.get("detector_label"),
                "detector_score": facts.get("detector_score"),
                "source": "rgbd_cluster_clip_yolo",
                "mask_area_px": facts["area_px"],
            }
        )
    return observations


def _attach_detector(prepared: list[tuple[np.ndarray, dict[str, Any]]], detections: list[dict[str, Any]]) -> None:
    if not prepared or not detections:
        return
    scores = np.zeros((len(prepared), len(detections)), dtype=np.float64)
    for row, (mask, facts) in enumerate(prepared):
        area = max(1, int(np.count_nonzero(mask)))
        for col, detection in enumerate(detections):
            x1, y1, x2, y2 = [int(round(value)) for value in detection["bbox_xyxy"]]
            x1, x2 = sorted((max(0, x1), min(mask.shape[1], x2)))
            y1, y2 = sorted((max(0, y1), min(mask.shape[0], y2)))
            inside = int(np.count_nonzero(mask[y1:y2, x1:x2])) if x2 > x1 and y2 > y1 else 0
            coverage = inside / area
            scores[row, col] = 0.78 * coverage + 0.22 * box_iou(facts["bbox_xyxy"], detection["bbox_xyxy"])
    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(-scores)
    for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
        if scores[row, col] < 0.20:
            continue
        prepared[row][1]["detector_label"] = detections[col]["label"]
        prepared[row][1]["detector_score"] = float(detections[col]["score"] * scores[row, col])


def _crop(rgb: np.ndarray, bbox: list[float]) -> Image.Image:
    height, width = rgb.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, max(x1 + 1, x2)), min(height, max(y1 + 1, y2))
    return Image.fromarray(rgb[y1:y2, x1:x2])


def _masked_crop(rgb: np.ndarray, mask: np.ndarray, bbox: list[int]) -> Image.Image:
    height, width = rgb.shape[:2]
    x1, y1, x2, y2 = [int(value) for value in bbox]
    span = max(x2 - x1, y2 - y1, 1)
    padding = max(6, int(round(span * 0.22)))
    x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
    x2, y2 = min(width, x2 + padding), min(height, y2 + padding)
    crop = rgb[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]
    muted = np.full_like(crop, 116)
    muted[crop_mask] = crop[crop_mask]
    side = max(muted.shape[:2])
    canvas = np.full((side, side, 3), 116, dtype=np.uint8)
    y_offset = (side - muted.shape[0]) // 2
    x_offset = (side - muted.shape[1]) // 2
    canvas[y_offset : y_offset + muted.shape[0], x_offset : x_offset + muted.shape[1]] = muted
    return Image.fromarray(canvas)


def _touches_edge(bbox: list[int], width: int, height: int) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 <= 2 or y1 <= 2 or x2 >= width - 2 or y2 >= height - 2
