**English** | [中文](README.zh.md)

# JEV Sees

### Give JEV eyes.

**JEV is fast, structured, and built to make decisions. It just cannot see — officially. JEV Sees changes that.**

Connect an image, a video stream, or an RGB-D camera to JEV, and suddenly the same judgment engine can work on the visual world: identify what matters, estimate risk, score a situation, or judge many visible objects at once.

**One frame. Many objects. One JEV call.**

![JEV Sees traffic demo](assets/demo.gif)

> In the demo above, every visible pedestrian gets an accident-risk probability from the same JEV call.

**JEV judges. JEV Sees lets it see. [JEV Control](https://github.com/CharlesFeng0314/JEV_control_your_roboarm) lets it act.**

---

## Why give JEV eyes?

JEV is already good at fast, closed, structured judgment: **choice, probability, score**.

But without vision, a huge class of useful questions is simply unavailable:

- Is this pedestrian in danger?
- Which object should the robot pay attention to?
- Has the target changed state?
- Which visible item best matches the condition?
- Can I ask the same question about every person in this frame?

JEV Sees opens that door.

It adds a visual front end to JEV so a camera can become another source of decisions — not just something that produces a description.

### Why this is interesting next to a VLM

VLMs are great when you want open-ended visual understanding: describe a scene, answer a broad question, explain what is happening.

JEV is interesting when the question is already known and you want the answer **fast, structured, repeatable, and easy to run many times**.

| Open-ended VLM workflow | JEV Sees + JEV |
| --- | --- |
| “Describe what is happening.” | “Is each pedestrian at risk?” |
| Free-form generation | Choice / probability / score |
| One broad visual prompt | Many closed judgments in one call |
| Human-readable response | Program-ready result |
| Great for exploration | Great for repeated decisions |

That difference becomes especially useful in live video, robotics, monitoring, testing, and other systems where the same kind of decision may need to be made again and again.

---

## What can I ask?

Anything that can be expressed as a closed judgment over the scene.

### Identify

```python
result = sees(
    "assets/bus.jpg",
    "What color is the bus?",
    ["yellow", "red", "white", "blue", "black", "uncertain"],
)

print(result.choice)
# blue
```

### Estimate a probability

```python
result = sees.ask(
    "Is object_003 in immediate danger from a vehicle?",
    "yes/no",
)

print(result.noul)
# probability of "yes"
```

### Judge many objects at once

```python
questions = {
    "object_001": {
        "yes": "object_001 is in a car accident or a car is about to hit them",
        "no": "object_001 is clear of every car",
    },
    "object_004": {
        "yes": "object_004 is in a car accident or a car is about to hit them",
        "no": "object_004 is clear of every car",
    },
}

result = sees.ask("Each question is one pedestrian.", questions)

for object_id, answer in result.answers.items():
    print(object_id, answer.noul)
```

One call can therefore return one structured answer per object instead of one free-form paragraph about the entire frame.

---

## Quick start

### 1. Install

```bash
git clone https://github.com/CharlesFeng0314/JEV_sees.git
cd JEV_sees
python -m pip install -e .
```

JEV Sees requires Python 3.10+.

### 2. Add a JEV API key

Create a key at [console.typesafe.ai/keys](https://console.typesafe.ai/keys).

macOS / Linux:

```bash
export TYPESAFE_API_KEY="your-key"
```

PowerShell:

```powershell
$env:TYPESAFE_API_KEY = "your-key"
```

Or create a `.env` file in the directory you run from:

```text
TYPESAFE_API_KEY=your-key
```

You can also pass `Sees(api_key="...")` directly.

> No key yet? `observe()` still works locally. The API key is only required when JEV is asked to make a judgment.

### 3. Run the smallest example

```python
from jev_sees import Sees

sees = Sees()

result = sees(
    "assets/bus.jpg",
    "What color is the bus?",
    ["yellow", "red", "white", "blue", "black", "uncertain"],
)

print(result.choice, result.confidence)
```

Expected result on the included image:

```text
blue 1.0
```

Full example: [examples/bus_color.py](examples/bus_color.py)

---

## Perception, tracking, and scene memory

`observe()` is the local visual layer behind JEV Sees:

```python
tracks = sees.observe("assets/bus.jpg")

for obj in tracks:
    print(obj["object_id"], obj["label"], obj["bbox_xyxy"])
```

A typical tracked object contains:

```text
object_id
label
confidence
bbox_xyxy
centroid_uv
attributes.color
```

Across video frames, JEV Sees keeps object IDs stable when possible and maintains scene memory. That lets visual questions stay attached to the same object over time instead of treating every frame as a completely new world.

The scene can also carry useful spatial context such as:

- what is visible now
- what was seen earlier
- whether a remembered object is stale
- bounding-box overlap and gap
- centroid distance
- whether nearby objects appear to be approaching
- metric 3D positions and gaps when RGB-D is available

These are implementation details, but they unlock the product behavior that matters: **JEV can keep making structured judgments about a changing visual scene.**

---

## Question types

You write ordinary Python values; JEV Sees turns them into JEV question objects.

| What you want | Python input | Result |
| --- | --- | --- |
| Pick one answer | `["red", "blue", "green"]` | choice + probabilities |
| Pick one answer with descriptions | `{"safe": "...", "unsafe": "..."}` | choice + probabilities |
| Yes / no probability | `"yes/no"` | probability of yes |
| Yes / no with explicit criteria | `{"yes": "...", "no": "..."}` | probability of yes |
| Score against a rubric | tuple or `{"rubric": [...]}` | structured score |
| Ask several questions together | `{question_id: question_spec}` | one answer per key |

For a single question, shortcuts such as `result.choice`, `result.confidence`, `result.probabilities`, and `result.noul` are available.

For multiple questions, use `result.answers`.

---

## Live video: one probability per person

[examples/traffic_relations.py](examples/traffic_relations.py) reads the included street clip, tracks road users, and periodically asks JEV about every visible pedestrian in one call.

The loop is conceptually simple:

```text
video frame
   ↓
observe()
   ↓
person_1, person_2, car_1, ...
   ↓
one yes/no question per pedestrian
   ↓
one JEV call
   ↓
risk(person_1), risk(person_2), ...
```

Run it with:

```bash
python examples/traffic_relations.py
```

Without an API key, the script still runs the local perception/tracking path and writes the GIF; it simply skips the JEV judgment.

Source video credits: [assets/CREDITS.md](assets/CREDITS.md)

---

## RGB-D: let depth decide what exists

RGB-only mode starts from detector boxes.

RGB-D mode takes a different path: **depth clusters decide which physical objects exist**, then CLIP names the clusters and YOLO contributes additional semantic evidence.

That matters when a 2D detector misses something that is still physically present.

In the included RGB-D sample, frame 13 has no YOLO detection at the configured threshold, while the RGB-D pipeline still returns four objects.

| | | |
| --- | --- | --- |
| ![frame 6](assets/frame_06.png) | ![frame 13](assets/frame_13.png) | ![frame 15](assets/frame_15.png) |

| Frame | Depth clusters | YOLOv8s @ 0.25 | RGB-D pipeline |
| --- | ---: | --- | --- |
| 6 | 5 | 1 soup can | 5 objects |
| 13 | 4 | nothing | spoon, cube, water bottle, cracker box |
| 15 | 5 | 1 soup can | 5 objects |

With camera intrinsics, RGB-D observations can also include `position_m`, which lets the scene state carry metric 3D positions and object-to-object gaps.

---

## Performance

Measured on an RTX 4070 Ti SUPER after one warmup. Raw record: [assets/pipeline_benchmark.json](assets/pipeline_benchmark.json).

![Pipeline latency](assets/pipeline_latency.png)

| Input | YOLOv8s boxes | YOLOv8s full pipeline | YOLOv8l boxes | YOLOv8l full pipeline |
| --- | --- | --- | --- | --- |
| bus.jpg | bus + 4 people, 19.5 ms | bus labeled blue, 92.4 ms | bus + 4 people, 18.8 ms | 98.7 ms |
| zidane.jpg | 2 people, 20.8 ms | 102.9 ms | 2 people, 22.0 ms | 98.2 ms |

YOLOv8s is the default because the larger YOLOv8l did not improve latency in this benchmark.

The current RGB-D path is heavier: the included sample frames are roughly 379–408 ms end to end.

---

## Models and weights

Weights are intentionally not stored in this repository.

If `JEV_SEES_ROBO_ROOT` points to a directory containing:

```text
models/benchmark/yolov8s-worldv2.pt
weights/clip/ViT-B-32.pt
```

JEV Sees uses those files.

Otherwise Ultralytics and CLIP download their own weights on first use.

You can also pass another detector with `yolo_model=...`.

---

## Troubleshooting

If this fails:

```bash
python -c "import jev_sees"
```

the most common cause is that `pip` installed the package into a different Python environment from the `python` command you are using.

Use:

```bash
python -m pip install -e .
python -c "import jev_sees; print(jev_sees.__version__)"
```

Using `python -m pip` keeps installation and execution on the same interpreter.

Run the test suite with:

```bash
python -m unittest discover -s tests -v
```

---

## JEV family

JEV Sees is part of a simple idea: **JEV should not stop at text.**

Give it eyes, and it can judge the visual world.
Give it hands, and those judgments can become actions.

```text
              JEV
       structured judgment
          /           \
         /             \
   JEV Sees        JEV Control
      eyes             hands
     vision          robot action
```

- **JEV Sees** — give JEV visual input from images, video, and RGB-D cameras
- **[JEV Control Your Roboarm](https://github.com/CharlesFeng0314/JEV_control_your_roboarm)** — use JEV judgments to choose robot-arm actions

The long-term idea is straightforward: **see → judge → act.**

---

## Status

JEV Sees is currently **v0.1.0** and intentionally experimental.

The public surface is already small — `Sees()`, `observe()`, and `ask()` — but the perception stack, scene representation, examples, and evaluation are still evolving.

If you try it on another camera, another robot, a weird scene, or a use case the examples did not anticipate, open an issue or start a discussion. Those experiments are exactly what this repo is for.

---

## License

MIT
