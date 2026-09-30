"""Lazy YOLO-World and CLIP runtimes."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from .paths import clip_weight, yolo_weight
from .vocabulary import COLOR_LABELS


class VisionRuntime:
    def __init__(
        self,
        *,
        yolo_path: str | None = None,
        clip_path: str | None = None,
        vocabulary: list[str] | None = None,
        device: str | None = None,
    ):
        self.yolo_path = yolo_path or yolo_weight("s")
        self.clip_path = clip_path or clip_weight()
        self.vocabulary = list(vocabulary or [])
        self.device = device
        self._yolo: Any = None
        self._clip: Any = None
        self._preprocess: Any = None
        self._torch: Any = None
        self._text_features: Any = None
        self._text_labels: list[str] = []
        self._color_features: Any = None
        self._yolo_classes: tuple[str, ...] = ()

    def _ensure(self) -> None:
        if self._yolo is not None:
            return
        import clip
        import torch
        from ultralytics import YOLOWorld

        device = self.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._torch = torch
        self._clip, self._preprocess = clip.load(self.clip_path, device=device)
        self._clip.eval()
        self._yolo = YOLOWorld(self.yolo_path)

    def set_vocabulary(self, vocabulary: list[str]) -> None:
        labels = [str(label) for label in vocabulary if str(label).strip()]
        if labels == self.vocabulary and self._text_features is not None:
            return
        self.vocabulary = labels
        self._text_features = None
        self._yolo_classes = ()

    def detect(self, rgb: np.ndarray, *, confidence: float) -> list[dict[str, Any]]:
        self._ensure()
        if not self.vocabulary:
            raise ValueError("Set a vocabulary before detection")
        if tuple(self.vocabulary) != self._yolo_classes:
            self._yolo.set_classes(self.vocabulary)
            self._yolo_classes = tuple(self.vocabulary)
        result = self._yolo.predict(
            rgb[:, :, ::-1],
            device=0 if str(self.device).startswith("cuda") else "cpu",
            imgsz=640,
            conf=confidence,
            iou=0.5,
            max_det=50,
            verbose=False,
        )[0]
        detections = []
        boxes = result.boxes
        if boxes is None:
            return []
        for box, cls, score in zip(boxes.xyxy, boxes.cls, boxes.conf, strict=False):
            detections.append(
                {
                    "label": self.vocabulary[int(cls)],
                    "score": float(score),
                    "bbox_xyxy": [float(value) for value in box.detach().cpu().tolist()],
                }
            )
        return _cross_class_nms(detections)

    def classify_crops(self, crops: list[Image.Image], labels: list[str] | None = None) -> list[dict[str, float]]:
        self._ensure()
        names = list(labels or self.vocabulary)
        if not crops or not names:
            return []
        features = self._encode_labels(names)
        batch = self._torch.stack([self._preprocess(image) for image in crops]).to(self.device)
        with self._torch.inference_mode():
            image_features = self._clip.encode_image(batch)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            probabilities = (100.0 * image_features @ features.T).softmax(dim=-1)
        rows = probabilities.float().cpu().tolist()
        return [{names[index]: float(score) for index, score in enumerate(row)} for row in rows]

    def color_names(self, crops: list[Image.Image]) -> list[str]:
        if not crops:
            return []
        rows = self.classify_crops(crops, [f"a photo of a {color} object" for color in COLOR_LABELS])
        names = []
        for row in rows:
            best = max(row, key=row.get)
            color = best.removeprefix("a photo of a ").removesuffix(" object")
            names.append(color if color in COLOR_LABELS else "unknown")
        return names

    def _encode_labels(self, labels: list[str]) -> Any:
        import clip

        if labels == self._text_labels and self._text_features is not None:
            return self._text_features
        features = []
        with self._torch.inference_mode():
            for label in labels:
                prompt = label if label.startswith("a photo of") else f"a photo of a {label}"
                tokens = clip.tokenize(prompt).to(self.device)
                encoded = self._clip.encode_text(tokens)
                encoded = encoded / encoded.norm(dim=-1, keepdim=True)
                features.append(encoded[0])
        stacked = self._torch.stack(features)
        self._text_labels = list(labels)
        self._text_features = stacked
        return stacked


def _iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1e-9)


def _cross_class_nms(detections: list[dict[str, Any]], threshold: float = 0.65) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for detection in sorted(detections, key=lambda item: float(item["score"]), reverse=True):
        if all(_iou(detection["bbox_xyxy"], other["bbox_xyxy"]) < threshold for other in kept):
            kept.append(detection)
    return kept
