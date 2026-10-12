**English** | [中文](README.zh.md)

# JEV Sees

### Let JEV understand images and video—and turn visual judgments into program-ready output.

**JEV is fast, structured, and built to make decisions.**

[JEV](https://typesafe.ai/blog/introducing-system-one-models-and-jev) is the first **System One Model** released by [TypeSafe AI](https://typesafe.ai/), built for fast, structured decisions inside software. JEV, the System One API, and `typesafe-sdk` are all provided by TypeSafe AI. **JEV Sees is my independent open-source project built on those capabilities; it is not an official TypeSafe AI vision product.**

JEV Sees grew out of my other project, [JEV Control Your Roboarm](https://github.com/CharlesFeng0314/JEV_control_your_roboarm). I first used visual context there to let JEV control a robot arm. I then pulled that vision layer out of one robot-specific workflow and turned it into a reusable product for images, video, monitoring, and robotics.

Vision runs locally: raw pixels become scene state with labeled boxes, stable object IDs, position history, and spatial relations before JEV makes a judgment. You define the questions and options; JEV Sees makes each visual decision object-aware, stateful, repeatable, and ready to use in code.

**One frame. Many objects. One JEV call.**

![JEV Sees traffic demo](assets/demo.gif)

> In the demo above, every visible pedestrian gets an accident-risk probability from the same JEV call.

- One entry point for images, video, and RGB-D
- A label, bounding box, and stable cross-frame object ID for every tracked object
- Many selected objects combined into one JEV call per sampled frame
- Results ready for the terminal, JSON, Excel, or a live visual dashboard

---

## From one image to a visual decision loop

JEV Sees attaches a question to a specific object and carries that question with the object's ID across frames. Every sampled moment produces an explicit choice, probability, or score that an application can display, record, threshold, or use to trigger the next action.

Ask continuously:

- What is each pedestrian's probability of currently being in a car accident?
- Which object should the robot pay attention to?
- Has the target changed state?
- Which visible item best matches the condition?
- Can every selected object in this frame be judged at once?

The same flow works on one image or continuously in video, monitoring, automated testing, and robotics. You control the questions and options; JEV Sees handles object tracking, scene context, batched calls, and result organization.

---

## Run the demos

The repository includes one bus image and one traffic video. The examples stay intentionally small: they call the real JEV API and print structured results in the terminal. Together they show a direct official `Choice`, automatic per-object probabilities over video, and semantic class routing for an animal motion question.

### Bus: choose from user-provided colors

[examples/bus_color.py](examples/bus_color.py) detects the bus, then creates an official `Choice` keyed by that bus's object ID. The application explicitly supplies yellow, red, white, blue, black, and uncertain as the available choices.

```bash
python examples/bus_color.py
```

Expected terminal output:

```text
object_id   label  choice  confidence
object_001  bus    blue    1.00
```

### Traffic: one live accident probability per pedestrian per sampled frame

[examples/traffic_relations.py](examples/traffic_relations.py) supplies one English question and the semantic class `pedestrian`. JEV maps visual labels such as person or man to that class, while JEV Sees automatically binds the question to every current object ID. Scope decisions and accident probabilities share one JEV call per sampled frame.

```bash
python examples/traffic_relations.py
```

A real JEV run prints rows like these:

```text
frame_index  video_time_s  object_id   label   scope_probability  probability
504          42.00         object_001  person  0.98               0.04
510          42.50         object_001  person  0.99               0.05
```

The probability answers “Is this pedestrian currently involved in a car accident in this frame?” It does not add a two-second prediction horizon. Object IDs stay stable across consecutive samples; a new person receives a new ID.

The application always keeps control of the question contract: callers explicitly choose `Choice`, `Noul`, or `Score`, and a `Choice` must include caller-provided options. JEV Sees does not infer question types from natural language or invent options for the user.

### Animal motion: one Choice distribution per animal

[examples/animal_motion.py](examples/animal_motion.py) uses an animal image or video selected in its `MEDIA` constant. The visual model may label an object `rabbit`, `dog`, or `horse`; JEV decides whether that label belongs to the requested `animal` class and returns the caller-provided motion-state distribution for every match.

```bash
python examples/animal_motion.py
```

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

```bash
python examples/bus_color.py
```

Expected terminal output for the included image:

```text
object_id   label  choice  confidence
object_001  bus    blue    1.00
```

The example uses `Sees()(image, question, questions)` for local perception, state construction, and the official JEV call. Full source: [examples/bus_color.py](examples/bus_color.py)

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
```

Color is evidence, not an SDK verdict. The CV branch preserves pixel measurements and CLIP preserves its complete color probability distribution. Florence region captions are not sent to JEV because they can repeat labels or inject unverified relationships.

Across video frames, JEV Sees keeps object IDs stable in private tracking memory. The prompt contains only objects detected in the current sampled frame. Their two most recent time-aware poses include `frame_index` and media time, so JEV can reason from box movement and the actual interval without receiving stale objects.

The scene can also carry useful spatial context such as:

- what is visible now
- bounding-box overlap and gap
- centroid distance
- whether nearby objects appear to be approaching
- metric 3D positions and gaps when RGB-D is available

These are implementation details, but they unlock the product behavior that matters: **JEV can keep making structured judgments about a changing visual scene.**

### Keeping `given_that` / state within the input limit

JEV Sees sends structured visual context produced by `state()`, not video frames themselves. The implementation respects JEV's 31,000-token hard input limit and targets a more conservative maximum of **24,000 serialized characters**. It deliberately estimates one character as one token, leaving headroom for questions and protocol overhead.

Compaction preserves every current object ID. It reduces relations, pose history, and optional attributes in that order. If the essential current-frame state still cannot fit, it raises an explicit error instead of silently dropping objects that have questions attached. The actual serialized length is available in `state["prompt_budget"]`, `sees.last_prompt_chars`, and the lower-right corner of rendered dashboards.

---

## Question types

`Choice`, `Noul`, `Score`, and `TypeSafeClient` are re-exported by `jev_sees` for a single import line. They are the official `typesafe-sdk` classes, so direct calls still use `client.system_one(...)` and official typed collections such as `response.choices`, `response.nouls`, and `response.scores`.

The high-level object API makes the type explicit in the method name and automatically creates the official questions:

```python
Sees().probability(video, "Is this object blocking the road?")

Sees().for_each("animal").choice(
    video,
    "What is this object's current motion state?",
    choices=["stationary", "walking", "running", "uncertain"],
)
```

Without `for_each(...)`, every detected object is returned. With it, JEV returns an additional class-membership probability per object; the SDK includes objects at or above `scope_threshold` and preserves the scope probabilities in structured results. The raw `Sees()(source, question, questions)` entry point remains available when an application wants to construct official `Choice`, `Noul`, or `Score` objects itself.

---

## Video: one live probability per pedestrian per sampled frame

[examples/traffic_relations.py](examples/traffic_relations.py) is intentionally a minimal terminal example. The application supplies `pedestrian` as its semantic class and an English accident question. The SDK creates the per-ID questions; no traffic plan or accident definition is built into JEV Sees.

Video results are frame-primary in `result.frames`: every evaluated sample contains its `frame_index`, `video_time_s`, and the probability for every selected object in that frame. `result.rows` remains a per-object peak summary for compatibility; JSON and Excel exports also put frame data first.

Video defaults to roughly two evaluated samples per second. Pass `every=1` to evaluate every frame or `every=N` to evaluate every Nth frame. Perception and JEV run only on sampled frames, and all selected objects in one sample are combined into one JEV request.

With `save=`, the output includes bounding boxes labeled with object type, object ID, and the live answer. A selected object's right-panel row stays pinned after its first appearance. Probability rows show current value, peak, and sample count; Choice rows show the selected option and confidence. The lower-right corner shows the serialized state length.

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

## Where I started

I first built [JEV Control Your Roboarm](https://github.com/CharlesFeng0314/JEV_control_your_roboarm), connecting TypeSafe AI's JEV to robot-arm action selection. While building its vision path, I realized that capability should not belong to one arm, so I extracted and generalized it into JEV Sees.

```text
TypeSafe AI / JEV
        │
        ├── JEV Control Your Roboarm: JEV judgment → robot-arm action
        │          │
        │          └── extract and generalize the visual layer
        │                              ↓
        └────────────────────────── JEV Sees: image / video / RGB-D → structured visual state
```

- **[JEV Control Your Roboarm](https://github.com/CharlesFeng0314/JEV_control_your_roboarm)** — my original project, connecting JEV judgments to robot-arm action selection
- **JEV Sees** — the standalone vision project I derived from it, converting images, video, and RGB-D into structured context that JEV can judge

Together they follow the same direction: **see → judge → act.**

---

## Status

JEV Sees is currently **v0.1.0** and intentionally experimental.

The public surface stays small: `Sees(...)` is the high-level image/video product entry point, while `observe()` and `state()` expose the visual layer for direct official JEV calls. Official JEV classes are re-exported unchanged. The perception stack, scene representation, examples, and evaluation are still evolving.

If you try it on another camera, another robot, a weird scene, or a use case the examples did not anticipate, open an issue or start a discussion. Those experiments are exactly what this repo is for.

---

## License

MIT
