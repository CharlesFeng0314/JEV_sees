**English** | [中文](README.zh.md)

# JEV Sees

**Ask JEV about a picture. Or about every person in a live video.**

What color is the bus. How likely is a car accident for each pedestrian, right now. One picture, one call. One video frame, every person scored together.

![Chance of a car accident for each person](docs/demo.gif)

The camera half of [JEV Control Your Roboarm](https://github.com/CharlesFeng0314/JEV_control_your_roboarm). The eyes live here. The hands stay there.

- A picture, one closed question, one answer
- A video, one probability per person, one call
- Color, or color plus depth

## Run it

```bash
pip install -e .
```

Create a key at [console.typesafe.ai/keys](https://console.typesafe.ai/keys). Hit **Create key**, name it, and copy it immediately. It is shown once. Do not put it in source, and do not commit it.

This PowerShell window only:

```powershell
$env:TYPESAFE_API_KEY = "paste your key"
```

That disappears when the window closes. To keep it, put a `.env` in the directory you run from. This repository already ignores that file.

```text
TYPESAFE_API_KEY=paste your key
```

Or pass it in code: `Sees(api_key="paste your key")`. Any one of the three is enough. An `api_key` argument wins. Otherwise the environment variable is used, then `.env` in the current directory.

Install and run with the same Python. Check first:

```bash
python -c "import jev_sees"
```

If that fails, the `python` you typed is not the one `pip` just used. Call the examples with that interpreter's full path.

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

```text
blue 1.0
```

A dict is one call that scores every person. `{"yes": "...", "no": "..."}` comes back as a probability:

```python
risks = sees.ask(
    "Each question is one pedestrian.",
    {
        "object_001": {
            "yes": "object_001 is in a car accident or a car is about to hit them",
            "no": "object_001 is clear of every car",
        },
    },
)
print(risks.answers["object_001"].noul)
```

- a list, or `{label: text}`, is a choice
- `"yes/no"` or `{"yes": "...", "no": "..."}` is yes/no
- a tuple, or `{"rubric": ["...", "..."]}`, is a score
- a dict of those values is several questions in one call

One question: `result.choice`. Several: `result.answers`. The video: [`examples/traffic_relations.py`](examples/traffic_relations.py).

## See it

**The bus is blue.** [`examples/bus_color.py`](examples/bus_color.py) on [`assets/bus.jpg`](assets/bus.jpg). Without a key you still get the boxes. With a key, JEV answers `blue` at 1.0.

![Bus](assets/bus.jpg)

**Every pedestrian, a live probability.** [`examples/traffic_relations.py`](examples/traffic_relations.py) reads [`assets/traffic.mp4`](assets/traffic.mp4). The GIF above is that run. Credits: [`assets/CREDITS.md`](assets/CREDITS.md).

## Fast enough to stay on the stream

RTX 4070 Ti SUPER, after one warmup. Full record: [`docs/pipeline_benchmark.json`](docs/pipeline_benchmark.json).

![Latency](docs/pipeline_latency.png)

Boxes land near 20 ms. The full color path on `bus.jpg` is 92 ms, and the bus is written down as blue. The larger YOLOv8l is not faster here, so YOLOv8s is the default.

| | YOLOv8s boxes | YOLOv8s pipeline | YOLOv8l boxes | YOLOv8l pipeline |
| --- | --- | --- | --- | --- |
| bus.jpg | bus + 4 people, 19.5 ms | bus written blue, 92.4 ms | bus + 4 people, 18.8 ms | 98.7 ms |
| zidane.jpg | 2 people, 20.8 ms | 102.9 ms | 2 people, 22.0 ms | 98.2 ms |

On a depth camera, depth decides how many objects exist and CLIP names them. On frame 13 the detector names nothing. The pipeline still returns four objects.

| | | |
| --- | --- | --- |
| ![frame 6](assets/rgbd/frame_06.png) | ![frame 13](assets/rgbd/frame_13.png) | ![frame 15](assets/rgbd/frame_15.png) |

| Frame | Depth clusters | YOLOv8s at 0.25 | Clusters + CLIP + YOLO-l |
| --- | --- | --- | --- |
| 6 | 5 in 230.4 ms | 1 soup can, 11.4 ms | 5 objects, 378.9 ms |
| 13 | 4 in 246.5 ms | nothing, 12.6 ms | spoon, cube, water bottle, cracker box, 408.4 ms |
| 15 | 5 in 209.4 ms | 1 soup can, 10.9 ms | 5 objects, 388.2 ms |

## Weights

Weights are not in this repository. Point `JEV_SEES_ROBO_ROOT` at a directory that already has `models/benchmark/yolov8s-worldv2.pt` and `weights/clip/ViT-B-32.pt`, and those files are used. Otherwise Ultralytics and CLIP download their own copies on first use. Pass `yolo_model=` for another file.

`observe` works without a key and returns `object_id` plus a box. `ask` needs the key.

```bash
python -m unittest discover -s tests -v
```
