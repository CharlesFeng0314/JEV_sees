"""Lazy Florence-2 region captioning and CLIP color runtime."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from .color import COLOR_NAMES
from .paths import clip_weight

DEFAULT_FLORENCE_MODEL = "microsoft/Florence-2-base-ft"
DENSE_REGION_CAPTION = "<DENSE_REGION_CAPTION>"


class VisionRuntime:
    """Discover and name regions without a caller-provided object vocabulary."""

    def __init__(
        self,
        *,
        florence_model: str = DEFAULT_FLORENCE_MODEL,
        clip_path: str | None = None,
        device: str | None = None,
    ):
        self.florence_model = florence_model
        self.clip_path = clip_path or clip_weight()
        self.device = device
        self._florence: Any = None
        self._processor: Any = None
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

    def detect(self, rgb: np.ndarray) -> list[dict[str, Any]]:
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

    def classify_crops(self, crops: list[Image.Image], labels: list[str]) -> list[dict[str, float]]:
        """Compare crops with explicit attributes such as color names.

        This is intentionally not used for object discovery. Object labels come
        from Florence-2; callers never provide an object vocabulary.
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
        if not crops:
            return []
        prompts = [f"a photo of a {color} object" for color in COLOR_NAMES]
        rows = self.classify_crops(crops, prompts)
        names = []
        for row in rows:
            best = max(row, key=row.get)
            color = best.removeprefix("a photo of a ").removesuffix(" object")
            names.append(color if color in COLOR_NAMES else "unknown")
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
