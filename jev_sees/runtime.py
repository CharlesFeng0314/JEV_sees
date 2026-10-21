"""Lazy Florence-2 region captioning and CLIP color runtime."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from .color import COLOR_NAMES
from .paths import clip_weight

DEFAULT_FLORENCE_MODEL = "microsoft/Florence-2-base-ft"
DEFAULT_YOLO_MODEL = "yolo11n.pt"
DENSE_REGION_CAPTION = "<DENSE_REGION_CAPTION>"
DETECTORS = ("florence", "yolo")
VISIONS = ("jev", "jev+florence", "jev+yolo", "jev+yolo+florence")
_MERGE_IOU = 0.5


class VisionRuntime:
    """Discover and name regions without a caller-provided object vocabulary."""

    def __init__(
        self,
        *,
        florence_model: str = DEFAULT_FLORENCE_MODEL,
        clip_path: str | None = None,
        device: str | None = None,
        detector: str = "florence",
        vision: str | None = None,
        yolo_model: str = DEFAULT_YOLO_MODEL,
        yolo_confidence: float = 0.25,
    ):
        selected = _resolve_vision(detector, vision)
        confidence = float(yolo_confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("yolo_confidence must be between 0 and 1")
        self.florence_model = florence_model
        self.clip_path = clip_path or clip_weight()
        self.device = device
        self.vision = selected
        self.detector = _detector_name(selected)
        self.yolo_model = yolo_model
        self.yolo_confidence = confidence
        self._florence: Any = None
        self._processor: Any = None
        self._yolo: Any = None
        self._clip: Any = None
        self._preprocess: Any = None
        self._torch: Any = None
        self._text_features: Any = None
        self._text_labels: list[str] = []

    def _device(self) -> str:
        if self._torch is None:
            import torch

            self._torch = torch
        if self.device is None:
            self.device = "cuda" if self._torch.cuda.is_available() else "cpu"
        return self.device

    def _ensure_florence(self) -> None:
        if self._florence is not None:
            return
        from transformers import AutoModelForCausalLM, AutoProcessor

        device = self._device()
        dtype = self._torch.float16 if device.startswith("cuda") else self._torch.float32
        self._florence = AutoModelForCausalLM.from_pretrained(
            self.florence_model,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation="eager",
        ).to(device)
        self._florence.eval()
        self._processor = AutoProcessor.from_pretrained(
            self.florence_model,
            trust_remote_code=True,
        )

    def _ensure_clip(self) -> None:
        if self._clip is not None:
            return
        import clip

        device = self._device()
        self._clip, self._preprocess = clip.load(self.clip_path, device=device)
        self._clip.eval()

    def _ensure_yolo(self) -> None:
        if self._yolo is not None:
            return
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "YOLO detection needs the optional ultralytics package. "
                "Install it with: python -m pip install ultralytics"
            ) from exc
        self._yolo = YOLO(self.yolo_model)

    def detect(self, rgb: np.ndarray) -> list[dict[str, Any]]:
        """Return boxes and labels for the selected vision stack."""

        if self.vision == "jev":
            return _frame_detection(rgb)
        yolo = self._detect_yolo(rgb) if "yolo" in self.vision else None
        florence = self._detect_florence(rgb) if "florence" in self.vision else None
        if yolo is not None and florence is not None:
            return _merge_yolo_florence(yolo, florence)
        return yolo if yolo is not None else florence or []

    def _detect_florence(self, rgb: np.ndarray) -> list[dict[str, Any]]:
        """Return Florence-2 dense regions and their generated descriptions."""

        self._ensure_florence()
        image = Image.fromarray(np.ascontiguousarray(rgb))
        inputs = self._processor(
            text=DENSE_REGION_CAPTION,
            images=image,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(self.device)
        pixel_values = inputs["pixel_values"].to(
            self.device,
            dtype=next(self._florence.parameters()).dtype,
        )
        with self._torch.inference_mode():
            generated_ids = self._florence.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False,
            )
        generated = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
        )[0]
        parsed = self._processor.post_process_generation(
            generated,
            task=DENSE_REGION_CAPTION,
            image_size=image.size,
        )
        return _dense_regions(parsed, image.size)

    def _detect_yolo(self, rgb: np.ndarray) -> list[dict[str, Any]]:
        """Return COCO-style class names, scores, and boxes from Ultralytics."""

        self._ensure_yolo()
        image = Image.fromarray(np.ascontiguousarray(rgb))
        results = self._yolo.predict(
            source=image,
            conf=self.yolo_confidence,
            verbose=False,
            device=self._device(),
        )
        if not results:
            return []
        return _yolo_regions(results[0], image.size)

    def classify_crops(self, crops: list[Image.Image], labels: list[str]) -> list[dict[str, float]]:
        """Compare crops with explicit attributes such as color names.

        This is intentionally not used for object discovery. Object labels come
        from the selected detector; callers never provide an object vocabulary.
        """

        self._ensure_clip()
        names = [str(label) for label in labels if str(label).strip()]
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
        """Return only the leading name for compatibility with custom runtimes."""

        return [str(item["label"]) for item in self.color_evidence(crops)]

    def color_evidence(self, crops: list[Image.Image]) -> list[dict[str, Any]]:
        """Return CLIP's full color distribution for every crop."""

        if not crops:
            return []
        prompts = [f"a photo of a {color} object" for color in COLOR_NAMES]
        rows = self.classify_crops(crops, prompts)
        evidence = []
        for row in rows:
            probabilities = {
                prompt.removeprefix("a photo of a ").removesuffix(" object"): float(score)
                for prompt, score in row.items()
            }
            best = max(probabilities, key=probabilities.get)
            evidence.append(
                {
                    "label": best if best in COLOR_NAMES else "unknown",
                    "confidence": float(probabilities[best]),
                    "probabilities": probabilities,
                }
            )
        return evidence

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


def _resolve_vision(detector: str, vision: str | None) -> str:
    """Map the public stack names, including the older detector switch."""

    if vision is None:
        name = str(detector).strip().lower()
        if name == "yolo":
            return "jev+yolo"
        if name == "florence":
            return "jev+florence"
        raise ValueError(f"detector must be one of: {', '.join(DETECTORS)}")
    name = "".join(str(vision).split()).lower()
    aliases = {
        "jev": "jev",
        "florence": "jev+florence",
        "jev+florence": "jev+florence",
        "yolo": "jev+yolo",
        "jev+yolo": "jev+yolo",
        "yolo+florence": "jev+yolo+florence",
        "jev+yolo+florence": "jev+yolo+florence",
    }
    if name not in aliases:
        raise ValueError(f"vision must be one of: {', '.join(VISIONS)}")
    return aliases[name]


def _detector_name(vision: str) -> str:
    if vision == "jev+yolo":
        return "yolo"
    if vision == "jev":
        return "jev"
    if vision == "jev+yolo+florence":
        return "yolo+florence"
    return "florence"


def _frame_detection(rgb: np.ndarray) -> list[dict[str, Any]]:
    """One full-frame object so JEV can answer without a local detector."""

    height, width = rgb.shape[:2]
    return [
        {
            "label": "scene",
            "description": "the whole frame",
            "bbox_xyxy": [0.0, 0.0, float(width), float(height)],
        }
    ]


def _merge_yolo_florence(
    yolo_rows: list[dict[str, Any]],
    florence_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep YOLO classes, attach an overlapping Florence caption, and add the rest."""

    from .tracking import box_iou

    merged = [dict(item) for item in yolo_rows]
    for region in florence_rows:
        best_index = None
        best_iou = 0.0
        for index, kept in enumerate(merged):
            overlap = box_iou(region.get("bbox_xyxy") or [], kept.get("bbox_xyxy") or [])
            if overlap >= _MERGE_IOU and overlap > best_iou:
                best_iou = overlap
                best_index = index
        if best_index is None:
            merged.append(dict(region))
            continue
        caption = str(region.get("description") or region.get("label") or "").strip()
        label = str(merged[best_index].get("label") or "").strip()
        if caption and caption != label:
            merged[best_index]["description"] = caption
    return merged


def _dense_regions(parsed: Any, image_size: tuple[int, int]) -> list[dict[str, Any]]:
    """Normalize Florence's post-processed dense-region response."""

    payload = parsed.get(DENSE_REGION_CAPTION, {}) if isinstance(parsed, dict) else {}
    boxes = payload.get("bboxes") or []
    labels = payload.get("labels") or []
    width, height = image_size
    detections = []
    for box, raw_label in zip(boxes, labels, strict=False):
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in box]
        x1, x2 = sorted((max(0.0, x1), min(float(width), x2)))
        y1, y2 = sorted((max(0.0, y1), min(float(height), y2)))
        label = " ".join(str(raw_label).split()).strip(" .")
        if not label or x2 <= x1 or y2 <= y1:
            continue
        detections.append(
            {
                "label": label,
                "description": label,
                "bbox_xyxy": [x1, y1, x2, y2],
            }
        )
    return detections


def _yolo_regions(result: Any, image_size: tuple[int, int]) -> list[dict[str, Any]]:
    """Normalize one Ultralytics result into the shared detection shape."""

    boxes = getattr(result, "boxes", None)
    xyxy = getattr(boxes, "xyxy", None)
    if boxes is None or xyxy is None or len(xyxy) == 0:
        return []
    names = getattr(result, "names", {}) or {}
    coordinates = xyxy.detach().cpu().tolist()
    confidences = boxes.conf.detach().cpu().tolist() if getattr(boxes, "conf", None) is not None else []
    classes = boxes.cls.detach().cpu().tolist() if getattr(boxes, "cls", None) is not None else []
    width, height = image_size
    detections = []
    for index, box in enumerate(coordinates):
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in box]
        x1, x2 = sorted((max(0.0, x1), min(float(width), x2)))
        y1, y2 = sorted((max(0.0, y1), min(float(height), y2)))
        if x2 <= x1 or y2 <= y1:
            continue
        class_id = int(classes[index]) if index < len(classes) else None
        raw_label = names.get(class_id, class_id) if isinstance(names, dict) else class_id
        label = " ".join(str(raw_label).split()).strip() or "object"
        detection: dict[str, Any] = {
            "label": label,
            "description": label,
            "bbox_xyxy": [x1, y1, x2, y2],
        }
        if index < len(confidences):
            detection["score"] = float(confidences[index])
        detections.append(detection)
    return detections
