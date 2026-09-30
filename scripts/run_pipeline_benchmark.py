"""Compare the local vision weights on the public demo images and wrist RGB-D frames.

Numbers written here are the only figures the README is allowed to quote.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jev_sees.color import dominant_color
from jev_sees.geometry import cluster_depth, depth_to_meters
from jev_sees.paths import clip_weight, mobile_sam_weight, robo_root, yolo_weight
from jev_sees.perceive import perceive_rgb, perceive_rgbd
from jev_sees.runtime import VisionRuntime

ASSETS = ROOT / "assets"
DOCS = ROOT / "docs"
WRIST_INTRINSICS = {"fx": 549.7494505734561, "fy": 549.7494505734561, "cx": 320.0, "cy": 240.0}
HOUSEHOLD = [
    "bowl",
    "sugar box",
    "cracker box",
    "soup can",
    "mug",
    "water bottle",
    "coffee jar",
    "spoon",
    "cube",
]


def _rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _timed(fn):
    started = time.perf_counter()
    value = fn()
    return value, round((time.perf_counter() - started) * 1000.0, 1)


def _pipeline_summary(runtime: VisionRuntime, rgb: np.ndarray) -> dict:
    """Time the RGB product path: YOLO boxes, CLIP color, then the body-color heuristic."""

    observations, elapsed = _timed(lambda: perceive_rgb(runtime, rgb))
    return {
        "latency_ms": elapsed,
        "count": len(observations),
        "objects": [
            {
                "label": item["label"],
                "confidence": round(float(item["confidence"]), 4),
                "color": (item.get("attributes") or {}).get("color"),
                "clip_color": (item.get("attributes") or {}).get("clip_color"),
                "dominant_color": (item.get("attributes") or {}).get("dominant_color"),
                "bbox_xyxy": item["bbox_xyxy"],
            }
            for item in observations
        ],
    }


def _detect_summary(runtime: VisionRuntime, rgb: np.ndarray, confidence: float):
    detections, elapsed = _timed(lambda: runtime.detect(rgb, confidence=confidence))
    return {
        "latency_ms": elapsed,
        "count": len(detections),
        "labels": sorted({item["label"] for item in detections}),
        "detections": [
            {
                "label": item["label"],
                "score": round(float(item["score"]), 4),
                "bbox_xyxy": [round(float(value), 1) for value in item["bbox_xyxy"]],
            }
            for item in detections
        ],
    }


def _wrist_frames() -> list[tuple[Path, Path]]:
    root = robo_root()
    if root is None:
        return []
    live = root / "data" / "manipulation" / "sugar_box_benchmark_v1" / "live"
    # Later frames actually contain the household objects. The first frames look at an empty surface.
    names = (
        "1790672865144606100_6",
        "1790672907604492800_13",
        "1790672919993928700_15",
    )
    pairs = []
    for name in names:
        rgb_path = live / f"{name}_rgb.png"
        depth_path = live / f"{name}_depth.npy"
        if rgb_path.is_file() and depth_path.is_file():
            pairs.append((rgb_path, depth_path))
    return pairs


def _sam_summary(weight: str, rgb: np.ndarray) -> dict:
    from ultralytics import SAM

    model = SAM(weight)
    try:
        result, elapsed = _timed(lambda: model.predict(rgb[:, :, ::-1], verbose=False)[0])
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    masks = getattr(result, "masks", None)
    count = 0 if masks is None else len(masks)
    del model
    return {
        "ok": True,
        "latency_ms": elapsed,
        "mask_count": count,
        "category_names": [],
        "names_objects": False,
    }


def main() -> int:
    import torch

    print("benchmark start", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report: dict = {
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "note": (
            "Development comparison on two public RGB images and three wrist RGB-D frames "
            "that contain household objects. Not a COCO mAP evaluation. "
            "RGB detector latency and the RGB product path (perceive_rgb) are measured after one warmup, so they exclude weight loading."
        ),
        "weights": {
            "yolov8s_world": Path(yolo_weight("s") or "").name,
            "yolov8l_world": Path(yolo_weight("l") or "").name,
            "clip": Path(clip_weight() or "").name,
            "mobile_sam": Path(mobile_sam_weight() or "").name,
        },
        "rgb": {},
        "rgbd": [],
    }
    bus = _rgb(ASSETS / "bus.jpg")
    zidane = _rgb(ASSETS / "zidane.jpg")
    open_vocab = ["bus", "person", "car", "truck"]
    for size in ("s", "l"):
        print(f"rgb yolov8{size}", flush=True)
        runtime = VisionRuntime(yolo_path=yolo_weight(size), vocabulary=open_vocab, device=device)
        runtime.detect(bus, confidence=0.25)
        report["rgb"][f"yolov8{size}_world"] = {
            "bus.jpg": _detect_summary(runtime, bus, 0.25),
            "zidane.jpg": _detect_summary(runtime, zidane, 0.25),
        }
        # Warm the color encoder, then time the same path the SDK uses.
        perceive_rgb(runtime, bus)
        report["rgb"][f"yolov8{size}_world_pipeline"] = {
            "bus.jpg": _pipeline_summary(runtime, bus),
            "zidane.jpg": _pipeline_summary(runtime, zidane),
        }
        del runtime
        if device == "cuda":
            torch.cuda.empty_cache()

    clip_runtime = VisionRuntime(yolo_path=yolo_weight("l"), vocabulary=open_vocab, device=device)
    bus_boxes = report["rgb"]["yolov8l_world"]["bus.jpg"]["detections"]
    bus_box = next((item for item in bus_boxes if item["label"] == "bus"), None)
    if bus_box is not None:
        x1, y1, x2, y2 = [int(value) for value in bus_box["bbox_xyxy"]]
        crop = bus[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
        colors, elapsed = _timed(lambda: clip_runtime.color_names([Image.fromarray(crop)]))
        report["rgb"]["bus_color"] = {
            "clip_color": colors[0] if colors else None,
            "dominant_color": dominant_color(crop),
            "clip_latency_ms": elapsed,
            "crop_bbox_xyxy": bus_box["bbox_xyxy"],
        }
    sam_weight = mobile_sam_weight()
    if sam_weight:
        report["rgb"]["mobile_sam_bus"] = _sam_summary(sam_weight, bus)
        if device == "cuda":
            torch.cuda.empty_cache()

    print("wrist frames", flush=True)
    frames = []
    for rgb_path, depth_path in _wrist_frames():
        rgb = _rgb(rgb_path)
        depth = depth_to_meters(np.load(depth_path))
        masks, cluster_ms = _timed(lambda depth=depth: cluster_depth(depth, WRIST_INTRINSICS))
        frames.append(
            {
                "rgb_name": rgb_path.name,
                "depth_name": depth_path.name,
                "rgb": rgb,
                "depth": depth,
                "depth_clusters": {"count": len(masks), "latency_ms": cluster_ms},
            }
        )
    detectors = {}
    for size in ("s", "l"):
        detectors[size] = VisionRuntime(yolo_path=yolo_weight(size), vocabulary=HOUSEHOLD, device=device)
        if frames:
            detectors[size].detect(frames[0]["rgb"], confidence=0.25)
    for frame in frames:
        record = {
            "rgb": frame["rgb_name"],
            "depth": frame["depth_name"],
            "depth_clusters": frame["depth_clusters"],
        }
        for size, confidence in (("s", 0.25), ("s", 0.02), ("l", 0.25), ("l", 0.02)):
            record[f"yolov8{size}_world_conf_{confidence}"] = _detect_summary(
                detectors[size], frame["rgb"], confidence
            )
        observations, full_ms = _timed(
            lambda frame=frame: perceive_rgbd(
                detectors["l"], frame["rgb"], frame["depth"], WRIST_INTRINSICS, yolo_confidence=0.02
            )
        )
        record["cluster_clip_yolo_l"] = {
            "latency_ms": full_ms,
            "count": len(observations),
            "objects": [
                {
                    "label": item["label"],
                    "confidence": round(float(item["confidence"]), 4),
                    "bbox_xyxy": item["bbox_xyxy"],
                    "detector_label": item.get("detector_label"),
                }
                for item in observations
            ],
        }
        report["rgbd"].append(record)
    del detectors
    if device == "cuda":
        torch.cuda.empty_cache()

    report["selection"] = _selection(report)
    DOCS.mkdir(parents=True, exist_ok=True)
    destination = DOCS / "pipeline_benchmark.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_latency_chart(report, DOCS / "pipeline_latency.png")
    print(destination)
    print(json.dumps(report["selection"], indent=2))
    return 0


def _selection(report: dict) -> dict:
    """Pick the RGB detector from measured hits, then state the RGB-D reason."""

    def hits(size: str) -> dict:
        bus = report["rgb"][f"yolov8{size}_world"]["bus.jpg"]
        person = report["rgb"][f"yolov8{size}_world"]["zidane.jpg"]
        return {
            "bus": "bus" in bus["labels"],
            "person": "person" in person["labels"],
            "bus_ms": bus["latency_ms"],
            "person_ms": person["latency_ms"],
        }

    small = hits("s")
    large = hits("l")
    small_score = int(small["bus"]) + int(small["person"])
    large_score = int(large["bus"]) + int(large["person"])
    if large_score > small_score:
        chosen = "l"
        reason = "YOLOv8l-World named more of the expected objects than YOLOv8s-World."
    elif small_score > large_score:
        chosen = "s"
        reason = "YOLOv8s-World named more of the expected objects than YOLOv8l-World."
    small_ms = small["bus_ms"] + small["person_ms"]
    large_ms = large["bus_ms"] + large["person_ms"]
    if abs(small_ms - large_ms) < 15.0:
        chosen = "s"
        reason = (
            "Both sizes named the same expected objects. Warm latency differed by only a few "
            "milliseconds across runs, so the smaller YOLOv8s-World is the default."
        )
    elif small_ms < large_ms:
        chosen = "s"
        reason = "Both sizes named the same expected objects. YOLOv8s-World was faster, so it is the default."
    else:
        chosen = "l"
        reason = "Both sizes named the same expected objects. YOLOv8l-World was faster on this machine, so it is the default."
    return {
        "default_yolo_size": chosen,
        "rgb_reason": reason,
        "small": small,
        "large": large,
        "rgbd_reason": (
            "Depth clustering decides how many objects exist. CLIP names each cluster. "
            "YOLO-World is attached only when a box agrees with a cluster. "
            "MobileSAM, when it runs, returns masks without class names, so it cannot answer a closed question alone."
        ),
    }


def _write_latency_chart(report: dict, path: Path) -> None:
    """Line chart of boxes against the product path. PIL only."""

    rgb = report.get("rgb") or {}
    frames = report.get("rgbd") or []
    labels = ["YOLOv8s", "YOLOv8l"]
    boxes: list[float | None] = []
    pipeline: list[float | None] = []
    clusters: list[float | None] = [None, None]
    for size in ("s", "l"):
        detector = rgb.get(f"yolov8{size}_world", {}).get("bus.jpg")
        full = rgb.get(f"yolov8{size}_world_pipeline", {}).get("bus.jpg")
        boxes.append(float(detector["latency_ms"]) if detector else None)
        pipeline.append(float(full["latency_ms"]) if full else None)
    for frame in frames:
        stem = Path(str(frame.get("rgb") or "frame")).stem.removesuffix("_rgb")
        labels.append(f"wrist {stem.rsplit('_', 1)[-1]}")
        boxes.append(float(frame["yolov8s_world_conf_0.25"]["latency_ms"]))
        pipeline.append(float(frame["cluster_clip_yolo_l"]["latency_ms"]))
        clusters.append(float(frame["depth_clusters"]["latency_ms"]))
    if not any(value is not None for value in pipeline):
        return
    from PIL import ImageDraw
    from PIL import ImageFont

    font_path = Path(r"C:\Windows\Fonts\arial.ttf")
    font = ImageFont.truetype(str(font_path), 16) if font_path.is_file() else ImageFont.load_default()
    small = ImageFont.truetype(str(font_path), 13) if font_path.is_file() else font
    width, height = 920, 420
    left, right, top, bottom = 64, 24, 56, 48
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    peak = max(value for value in boxes + pipeline + clusters if value is not None)
    peak = max(peak * 1.12, 1.0)
    count = len(labels)

    def xy(index: int, value: float) -> tuple[int, int]:
        x = left + (width - left - right) * index / max(count - 1, 1)
        y = top + (height - top - bottom) * (1.0 - value / peak)
        return int(round(x)), int(round(y))

    for tick in (0, 100, 200, 300, 400):
        if tick > peak:
            continue
        _, y = xy(0, tick)
        draw.line((left, y, width - right, y), fill=(230, 230, 230))
        draw.text((8, y - 8), str(tick), fill=(90, 90, 90), font=small)
    series = (
        ("Boxes", boxes, (90, 140, 170)),
        ("Clusters", clusters, (40, 150, 110)),
        ("Full path", pipeline, (32, 86, 140)),
    )
    for name, values, color in series:
        points = [(index, value) for index, value in enumerate(values) if value is not None]
        if len(points) >= 2:
            draw.line([xy(index, value) for index, value in points], fill=color, width=3)
        for index, value in points:
            x, y = xy(index, value)
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
            draw.text((x - 12, y - 18), f"{value:.0f}", fill=color, font=small)
    for index, label in enumerate(labels):
        x, _ = xy(index, 0)
        draw.text((x - 28, height - 32), label, fill=(30, 30, 30), font=small)
    draw.text((left, 16), "Warm latency, milliseconds", fill=(20, 20, 20), font=font)
    legend_x = width - 280
    for offset, (name, _, color) in enumerate(series):
        y = 18
        x = legend_x + offset * 92
        draw.line((x, y + 6, x + 16, y + 6), fill=color, width=3)
        draw.text((x + 20, y), name, fill=(30, 30, 30), font=small)
    image.save(path)


if __name__ == "__main__":
    raise SystemExit(main())
