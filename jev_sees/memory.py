"""Fuse repeated observations of the same object id.

JEV itself is stateless. This memory is replayed into every given-that.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SceneMemory:
    def __init__(self) -> None:
        self.revision = 0
        self.known_objects: dict[str, dict[str, Any]] = {}

    def observe(
        self,
        observations: list[dict[str, Any]],
        captured_at: str | None = None,
        *,
        frame_index: int | None = None,
        video_time_s: float | None = None,
    ) -> None:
        now = captured_at or utc_now()
        self.revision += 1
        visible: set[str] = set()
        for item in observations:
            key = str(item.get("object_id") or "")
            if not key:
                continue
            visible.add(key)
            previous = self.known_objects.get(key, {})
            observed_label = str(item.get("label") or "unknown")
            evidence = dict(previous.get("label_evidence", {}))
            evidence[observed_label] = int(evidence.get(observed_label, 0)) + 1
            fused_label = max(evidence, key=evidence.get)
            pose = {
                "bbox_xyxy": item.get("bbox_xyxy"),
                "centroid_uv": item.get("centroid_uv"),
                "position_m": item.get("position_m"),
            }
            history = list(previous.get("pose_history", []))
            sample = {"captured_at": now, "pose": pose, "confidence": item.get("confidence")}
            if frame_index is not None:
                sample["frame_index"] = int(frame_index)
            if video_time_s is not None:
                sample["video_time_s"] = float(video_time_s)
            history.append(sample)
            history = history[-20:]
            confidence = item.get("confidence")
            record = {
                **previous,
                "object_id": key,
                "label": fused_label,
                "label_evidence": evidence,
                "description": item.get("description") or fused_label,
                "attributes": dict(item.get("attributes") or previous.get("attributes") or {}),
                "latest_pose": pose,
                "pose_history": history,
                "observation_count": int(previous.get("observation_count", 0)) + 1,
                "first_seen_at": previous.get("first_seen_at") or now,
                "last_seen_at": now,
                "last_seen_revision": self.revision,
                "currently_visible": True,
                "stale": False,
            }
            if confidence is not None:
                record["latest_confidence"] = float(confidence)
                record["best_confidence"] = max(float(previous.get("best_confidence", 0.0)), float(confidence))
            self.known_objects[key] = record
        for key, record in self.known_objects.items():
            if key not in visible:
                record["currently_visible"] = False
                record["stale"] = self.revision - int(record.get("last_seen_revision", self.revision)) >= 3

    def prompt_objects(self) -> list[dict[str, Any]]:
        objects = []
        for record in self.known_objects.values():
            pose = record.get("latest_pose") or {}
            item = {
                "object_id": record["object_id"],
                "label": record["label"],
                "description": record.get("description"),
                "attributes": record.get("attributes") or {},
                "confidence": record.get("latest_confidence"),
                "bbox_xyxy": pose.get("bbox_xyxy"),
                "centroid_uv": pose.get("centroid_uv"),
                "position_m": pose.get("position_m"),
                "observation_count": record["observation_count"],
                "currently_visible": record["currently_visible"],
                "stale": record["stale"],
                "pose_observation_count": len(record.get("pose_history") or []),
            }
            recent_poses = _recent_video_poses(record.get("pose_history") or [])
            if recent_poses:
                item["recent_poses"] = recent_poses
            objects.append(item)
        objects.sort(
            key=lambda item: (
                not bool(item.get("currently_visible")),
                bool(item.get("stale")),
                -int(item.get("observation_count") or 0),
            )
        )
        return objects


def _recent_video_poses(history: list[dict[str, Any]], limit: int = 2) -> list[dict[str, Any]]:
    """Project existing pose history into a small, video-time-aware prompt view."""

    recent = []
    for sample in history[-limit:]:
        if sample.get("video_time_s") is None:
            continue
        pose = sample.get("pose") or {}
        item: dict[str, Any] = {
            "video_time_s": round(float(sample["video_time_s"]), 4),
        }
        if sample.get("frame_index") is not None:
            item["frame_index"] = int(sample["frame_index"])
        for key in ("bbox_xyxy", "centroid_uv", "position_m"):
            value = pose.get(key)
            if isinstance(value, (list, tuple)):
                item[key] = [round(float(part), 4) for part in value]
        if sample.get("confidence") is not None:
            item["confidence"] = round(float(sample["confidence"]), 4)
        recent.append(item)
    return recent
