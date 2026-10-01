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
from jev_sees import Choice, Sees, TypeSafeClient

question = "What color is the bus?"
sees = Sees()
sees.observe("assets/bus.jpg")

with TypeSafeClient() as client:
    response = client.system_one(
        state=sees.state(question),
        questions={
            "bus_color": Choice(
                instructions=question,
                criteria={"yellow": None, "red": None, "blue": None, "uncertain": None},
            )
        },
    )

print(response.choices["bus_color"].choice)
# blue
```

`JEV Sees` creates `state`. The official `typesafe-sdk` still owns the `Choice`, the `system_one` call, and the response.

### Estimate a probability

```python
from jev_sees import Noul

question = "Is object_003 in immediate danger from a vehicle?"
state = sees.state(question)

with TypeSafeClient() as client:
    response = client.system_one(
        state=state,
        questions={"in_danger": Noul(instructions=question)},
    )

print(response.nouls["in_danger"].noul)
# probability of "yes"
```

### Judge many objects at once

```python
questions = {
    obj["object_id"]: Noul(
        instructions=f"Is {obj['object_id']} in immediate danger from a car?"
    )
    for obj in tracks
    if obj["label"] == "person"
}

with TypeSafeClient() as client:
    response = client.system_one(state=sees.state("Pedestrian risk"), questions=questions)
```

Each pedestrian remains a named official `Noul` question, and its official answer is available from `response.nouls[object_id]`.

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

Both the official `TypeSafeClient` and the high-level video entry point can read this key. `observe()` and `state()` themselves remain local.

### 3. Run the smallest example

```python
from jev_sees import Choice, Sees, TypeSafeClient

question = "What color is the bus?"
sees = Sees()
sees.observe("assets/bus.jpg")
with TypeSafeClient() as client:
    response = client.system_one(
        state=sees.state(question),
        questions={
            "bus_color": Choice(
                instructions=question,
                criteria={"yellow": None, "red": None, "blue": None, "uncertain": None},
            )
        },
    )
print(response.choices["bus_color"].choice)
```

Expected result on the included image:

```text
blue
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
attributes.color_evidence.cv
attributes.color_evidence.clip
attributes.color_evidence.caption
```

Color is evidence, not an SDK verdict. The CV branch preserves pixel measurements, CLIP preserves its complete color probability distribution, and Florence's original region caption remains alongside both. JEV can therefore judge agreement or disagreement instead of receiving one preselected color string.

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

`Choice`, `Noul`, `Score`, and `TypeSafeClient` are re-exported by `jev_sees` for a single import line. They are the official `typesafe-sdk` classes, so direct calls still use `client.system_one(...)` and official typed collections such as `response.choices`, `response.nouls`, and `response.scores`.

JEV Sees does not infer a question type from natural language and does not turn lists or dictionaries into JEV questions. Applications construct the official question objects themselves. For video, `questions=` may be a callable that receives the current tracked objects and returns a mapping of official questions.

---

## Video: one risk per pedestrian

[examples/traffic_relations.py](examples/traffic_relations.py) keeps sampling, tracking, rendering, and output formatting inside the SDK. Its small `questions()` function is application code: it creates official `Noul` objects for the current tracked objects. JEV Sees does not contain a traffic-specific plan or inspect the prompt to invent those questions.

Run it with:

```bash
python examples/traffic_relations.py
```

This example calls JEV and therefore requires `TYPESAFE_API_KEY`.

Source video credits: [assets/CREDITS.md](assets/CREDITS.md)

---

## RGB-D: let depth decide what exists

RGB-only mode starts from Florence-2 dense-region captions. Florence discovers boxes and generates their labels; callers do not pass an object vocabulary.

RGB-D mode takes a different path: **depth clusters decide which physical objects exist**, and overlapping Florence regions provide free-text names when available.

That matters when a 2D detector misses something that is still physically present.

| | | |
| --- | --- | --- |
| ![frame 6](assets/frame_06.png) | ![frame 13](assets/frame_13.png) | ![frame 15](assets/frame_15.png) |

With camera intrinsics, RGB-D observations can also include `position_m`, which lets the scene state carry metric 3D positions and object-to-object gaps.

---

## Performance

The earlier YOLO benchmark does not describe the Florence-2 pipeline and has intentionally been removed from the current documentation. A new image and video benchmark is required before publishing latency claims for this backend.

---

## Models and weights

Weights are intentionally not stored in this repository.

Florence-2 is loaded from Hugging Face on first use. The default is `microsoft/Florence-2-base-ft`; choose another compatible checkpoint with:

```python
Sees(florence_model="microsoft/Florence-2-large-ft")
```

If `JEV_SEES_ROBO_ROOT` points to a directory containing:

```text
weights/clip/ViT-B-32.pt
```

JEV Sees uses those files.

JEV Sees uses that local CLIP weight for color evidence. Otherwise CLIP downloads its weight on first use.

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

The public surface stays small: `Sees(...)` is the high-level image/video product entry point, while `observe()` and `state()` expose the visual layer for direct official JEV calls. Official JEV classes are re-exported unchanged. The perception stack, scene representation, examples, and evaluation are still evolving.

If you try it on another camera, another robot, a weird scene, or a use case the examples did not anticipate, open an issue or start a discussion. Those experiments are exactly what this repo is for.

---

## License

MIT
