# JEV Sees

这个 SDK 从机械臂项目里抽出公共视觉能力，把相机画面加工成 JEV 能用的场景事实。完整的机械臂闭环在 [JEV_control_your_roboarm](https://github.com/CharlesFeng0314/JEV_control_your_roboarm)。

JEV Sees does one thing. It turns an RGB or RGB-D frame into a hidden scene description, asks JEV a closed question, and returns JEV's answer. Robot actions, simulators, and grippers are not part of this package.

```python
from jev_sees import Sees

sees = Sees()  # reads TYPESAFE_API_KEY
result = sees("bus.jpg", "What color is the bus?", ["yellow", "red", "white", "blue", "uncertain"])
print(result.choice, result.confidence)
```

`result` is the official `SystemOneResponse` (`choice`, `confidence`, `probabilities`) with shortcuts for a single question. Several questions are returned on `result.answers`. The hidden scene description is not part of the response.

Questions are ordinary Python. The package turns them into JEV's `Choice`, `Noul`, and `Score` types:

- a list, or `{label: description}`, is a choice
- `"yes/no"` or `{"yes": "...", "no": "..."}` is a yes/no question
- `{"rubric": ["...", "..."]}` or a tuple is an ordered score
- a dict of those values is several questions

For a video, call `sees.observe(frame)` on each frame, then `sees.ask(...)` once. Every frame is detected and tracked. JEV is called only when you ask.

## What JEV actually receives

JEV does not receive pixels. The package builds an internal `given_that` from the user sentence, the objects visible now, the objects remembered from earlier frames, and the uncertainties. Each object fact keeps `object_id`, label, confidence, `bbox_xyxy`, and a pixel centroid. A depth frame also adds a camera-frame `position_m`. Masks, full score vectors, and the pose trail stay out of the prompt.

JEV's input limit is 31,000 tokens. The same budget used by the robot project applies here: the target is 24,000 characters, and the character count is the conservative token estimate. At most 16 objects are kept from memory and 16 from the current frame, preferring objects that are visible, not stale, and seen more often. If that is still too long, lower-priority objects are dropped down to 4, and distant contact pairs are dropped first. Contact is recorded only when boxes overlap or centroids are within 40 pixels. The budget numbers stay inside `given_that`. They are not copied onto the JEV response.

Object identity is stable across frames. The same `object_id` is merged rather than replaced. Labels are a majority vote, the latest box is what the prompt sees, and an object goes stale after three missed frames.

## Install

```bash
pip install -e .
```

Set `TYPESAFE_API_KEY`. Weights are not in this repository. If `JEV_SEES_ROBO_ROOT` points at a checkout that already has `models/benchmark/yolov8s-worldv2.pt` and `weights/clip/ViT-B-32.pt`, those files are used. Otherwise Ultralytics and CLIP download their own copies on first use. The default detector is `yolov8s-worldv2`. Pass `yolo_model=` to use another file.

`observe` works without an API key and returns `object_id` plus `bbox_xyxy`. `ask` requires the key.

## Why this pipeline

These numbers are one development comparison on two public images and three wrist RGB-D frames. They are not a COCO mAP. The machine was an NVIDIA GeForce RTX 4070 Ti SUPER. Detector latency is after one warmup prediction. The full record is [`docs/pipeline_benchmark.json`](docs/pipeline_benchmark.json).

### RGB

Vocabulary for this run: `bus`, `person`, `car`, `truck`. Confidence 0.25.

| Model | bus.jpg | zidane.jpg |
| --- | --- | --- |
| YOLOv8s-World | bus 0.87 and 4 people, 21.5 ms | 2 people, 23.5 ms |
| YOLOv8l-World | bus 0.94 and 4 people, 17.7 ms | 2 people, 20.8 ms |
| MobileSAM | 83 masks, no category names, 1652.9 ms | not used |

Both World models named the bus and the people. The warm-time gap was a few milliseconds and flipped between runs, so the smaller model is the default. MobileSAM returned masks and no class names, so it cannot answer a closed question by itself.

Color is a separate fact. On the bus crop, CLIP ViT-B/32 said `green` (1735.0 ms) and the body-color heuristic said `blue`. The bus in `assets/bus.jpg` is blue, so the color written into `given_that` is the heuristic when that heuristic is chromatic. CLIP remains the name source for depth clusters.

### RGB-D

Three wrist frames from a tabletop run, with the camera intrinsics of that camera. Household names only. Depth clustering removes the dominant plane, then groups the remaining points. YOLO-World at the detector's usual 0.25 confidence, and at 0.02, is the "boxes only" column.

| Frame | Depth clusters | YOLOv8s @ 0.25 | YOLOv8s @ 0.02 | YOLOv8l @ 0.25 | YOLOv8l @ 0.02 | Clusters + CLIP + YOLO-l |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | 5 in 223.2 ms | 1 soup can, 10.3 ms | 2 soup cans, 12.7 ms | 1 soup can, 19.7 ms | 2 soup cans, 18.7 ms | 5 objects in 386.0 ms |
| 13 | 4 in 243.4 ms | 0 in 11.7 ms | 0 in 11.0 ms | 0 in 15.2 ms | 0 in 15.0 ms | 4 objects in 434.6 ms |
| 15 | 5 in 217.6 ms | 1 soup can, 11.7 ms | 2 soup cans, 13.5 ms | 1 soup can, 15.6 ms | 1 soup can, 18.5 ms | 5 objects in 389.6 ms |

On frame 13 the detector named nothing at either confidence, while depth still found 4 objects and CLIP named them `spoon`, `cube`, `water bottle`, and `cracker box`. On frames 6 and 15 the detector's only class was `soup can`, including a second box at 0.02 that the depth path did not treat as a second can. Depth decides how many objects exist. CLIP names each cluster. YOLO is attached only when a box agrees with a cluster.

## Demos

Picture: [`examples/bus_color.py`](examples/bus_color.py) on [`assets/bus.jpg`](assets/bus.jpg).

```text
python examples/bus_color.py
```

Video: [`examples/docking_contact.py`](examples/docking_contact.py) reads [`assets/docking.mp4`](assets/docking.mp4), a five-second NASA public-domain excerpt of Progress 92 beside the station. See [`assets/CREDITS.md`](assets/CREDITS.md). The script prints `object_id` and `bbox_xyxy` for sampled frames, then asks once whether the vehicles are `in_contact`, `approaching`, `separated`, or `uncertain`. The vocabulary is set by the example (`spacecraft`, `space station`, `cargo ship`, `module`), not by the library.

Observing `assets/bus.jpg` twice kept the same ids. Color is taken from the whole box, so a person standing in front of the bus can inherit the bus color:

```text
object_001 person 0.9161 [51, 398, 247, 903] white
object_002 person 0.9054 [668, 386, 810, 879] blue
object_003 person 0.882 [223, 406, 345, 861] blue
object_004 bus 0.8703 [3, 231, 807, 745] blue
object_005 person 0.6351 [0, 484, 79, 886] blue
```

`examples/bus_color.py` then asked JEV. The response was:

```text
blue 1.0
{'white': 0.0, 'black': 0.0, 'blue': 1.0, 'uncertain': 0.0, 'red': 0.0, 'yellow': 0.0}
```

`examples/docking_contact.py` sampled 25 frames. At the default confidence the detector named nothing on this 480×270 excerpt, so no boxes were printed. JEV still answered from that empty memory:

```text
uncertain 0.98
{'uncertain': 0.99, 'in_contact': 0.0, 'approaching': 0.0, 'separated': 0.01}
```

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover track identity, memory merge, a `given_that` that keeps pixel boxes and omits image bytes, and the three question forms. The JEV client is faked. Tests do not call the network.
