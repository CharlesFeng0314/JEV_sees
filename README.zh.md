[English](README.md) | **中文**

# JEV Sees

### 我们让 JEV 拥有了视觉能力。

**JEV 很会判断：快、结构化，而且天然适合做封闭式决策。**

[JEV](https://typesafe.ai/blog/introducing-system-one-models-and-jev) 是 [TypeSafe AI](https://typesafe.ai/) 发布的首个 **System One Model**，面向软件中的快速、结构化决策。JEV Sees 是基于 JEV 和 `typesafe-sdk` 开发的独立开源项目，并非 TypeSafe AI 的官方视觉产品。

JEV Sees 支持图片、视频和 RGB-D 相机。它会找到画面中的对象，在视频中持续识别同一个对象，并让 JEV 回答你对每个目标提出的问题。

<table>
  <tr>
    <th width="33%">图片 · 公交车颜色</th>
    <th width="33%">视频 · 行人事故概率</th>
    <th width="33%">视频 · 动物运动状态</th>
  </tr>
  <tr>
    <td><img src="assets/bus_demo.png" alt="JEV 判断公交车是蓝色"></td>
    <td><img src="assets/demo.gif" alt="JEV 判断交通视频中每个行人的事故概率"></td>
    <td><img src="assets/animal_demo.gif" alt="JEV 判断黑猫的运动状态"></td>
  </tr>
  <tr>
    <td><strong>问题：</strong>这辆公交车是什么颜色？<br><strong>回答：</strong>blue</td>
    <td><strong>问题：</strong>这个行人当前是否正在经历交通事故？<br><strong>回答：</strong>每个行人一个概率</td>
    <td><strong>问题：</strong>这只动物当前是什么运动状态？<br><strong>回答：</strong>walking</td>
  </tr>
</table>

---

## 快速开始

### 1. 安装

```bash
git clone https://github.com/CharlesFeng0314/JEV_sees.git
cd JEV_sees
python -m pip install -e .
```

需要 Python 3.10+。第一次运行视觉任务时，如果本地还没有相应权重，会自动下载默认的 Florence-2 和 CLIP 模型。

### 2. 配置 JEV API Key

前往 [console.typesafe.ai/keys](https://console.typesafe.ai/keys) 创建 key。

macOS / Linux：

```bash
export TYPESAFE_API_KEY="your-key"
```

PowerShell：

```powershell
$env:TYPESAFE_API_KEY = "your-key"
```

### 3. 写下第一个视觉问题

下面是一段完整程序，不是对仓库 example 的二次封装：

```python
from jev_sees import Sees

result = Sees().for_each("bus").choice(
    "assets/bus.jpg",
    "What color is this bus?",
    choices=["yellow", "red", "white", "blue", "black", "uncertain"],
    save="bus_answer.png",
)

for row in result.rows:
    print(row["object_id"], row["choice"], row["confidence"])
```

典型输出：

```text
object_001 blue 1.0
```

保存的图片会标出检测到的公交车、稳定的 object ID 和 JEV 的回答。

---

## 仓库内置示例

三个示例使用完全一致的结构：输入媒体、提出问题、明确指定回答类型。每个脚本都会调用 JEV，并在终端打印结构化结果。

### 公交车颜色

- **输入：**[`assets/bus.jpg`](assets/bus.jpg)
- **问题：**“What color is the bus?”
- **回答类型：**从应用给出的颜色选项中返回 `Choice`

```bash
python examples/bus_color.py
```

```text
object_id   label  choice  confidence
object_001  bus    blue    1.00
```

源码：[`examples/bus_color.py`](examples/bus_color.py)

### 行人事故概率

- **输入：**[`assets/traffic.mp4`](assets/traffic.mp4) 的第 42–50 秒
- **问题：**“Is this object currently involved in a car accident?”
- **回答类型：**每个采样帧中，每个行人一个 probability

```bash
python examples/traffic_relations.py
```

```text
frame_index  video_time_s  object_id   label   scope_probability  probability
504          42.00         object_001  person  0.88               0.04
510          42.50         object_001  person  0.92               0.04
```

这个问题只判断当前画面正在发生什么，不预测未来事故。源码：[`examples/traffic_relations.py`](examples/traffic_relations.py)。视频来源：[`assets/CREDITS.md`](assets/CREDITS.md)。

### 动物运动状态

- **输入：**[`assets/black_cat_walking.mp4`](assets/black_cat_walking.mp4)
- **问题：**“What is this animal's current motion state?”
- **回答类型：**从 `stationary`、`walking`、`running` 和 `uncertain` 中返回 `Choice`

```bash
python examples/animal_motion.py
```

```text
frame_index  video_time_s  object_id   label      choice    confidence
0            0.00          object_001  black cat  uncertain 0.60
120          5.00          object_001  black cat  walking   0.23
```

源码：[`examples/animal_motion.py`](examples/animal_motion.py)

---

## 选择你需要的回答

JEV Sees 不会根据问题的措辞猜测回答格式，而是由代码明确选择：

- `probability(...)` 为每个目标返回一个 0 到 1 的独立概率，对应 JEV 官方的 `Noul` 问题类型。
- `choice(...)` 从应用给出的选项中选择，并返回选项、confidence 和完整概率分布。
- `score(...)` 按应用给出的 rubric 为每个目标评分。

```python
from jev_sees import Sees

sees = Sees()

risk = sees.for_each("pedestrian").probability(
    "assets/traffic.mp4",
    "Is this pedestrian currently involved in a car accident?",
    start=504,
    end=600,
)

motion = sees.for_each("animal").choice(
    "assets/black_cat_walking.mp4",
    "What is this animal's current motion state?",
    choices=["stationary", "walking", "running", "uncertain"],
)

occlusion = sees.for_each("package").score(
    "warehouse.jpg",
    "How obstructed is this package?",
    rubric=["0: clear", "1: partly obstructed", "2: heavily obstructed"],
)
```

不调用 `for_each(...)`，直接使用 `Sees().probability(...)`、`.choice(...)` 或 `.score(...)`，就会询问画面里检测到的全部对象。需要直接构造 JEV 官方问题时，也可以从 `jev_sees` 导入原始的 `Choice`、`Noul`、`Score` 和 `TypeSafeClient`。

---

## 输入与结果

### 图片

可以传入图片路径、PIL 图片或 RGB NumPy 数组。每一行结果都会指出这次判断对应哪个对象。

### 视频

object ID 会在连续采样帧间保持稳定，因此结果可以直接画图、设置阈值或触发动作。默认每秒大约判断两次；`every=1` 表示逐帧判断，`every=N` 表示每 N 帧判断一次。

### RGB-D 相机

实时相机循环可以通过 `observe(rgb, depth, intrinsics=...)` 传入对齐的 RGB 和深度帧。提供相机内参后，对象还可以带有米制三维位置和距离，再通过 `ask(...)` 交给 JEV 判断。

### 程序与文件输出

- `result.answers`：以 object ID 为 key 的 JEV 回答
- `result.rows`：每个对象一行摘要
- `result.frames`：按视频采样帧组织的回答
- `save="output.gif"`：带检测框和实时回答的图片、GIF 或视频
- `json="output.json"` 和 `excel="output.xlsx"`：结构化导出

同一个采样帧中所有被选择的对象会放进同一次 JEV 请求。

---

## 模型与本地视觉

图片和视频的视觉感知在本地运行。默认检测与描述模型是 `microsoft/Florence-2-base-ft`；也可以使用兼容的 checkpoint：

```python
Sees(florence_model="microsoft/Florence-2-large-ft")
```

如果 `JEV_SEES_ROBO_ROOT` 指向的项目中存在 `weights/clip/ViT-B-32.pt`，JEV Sees 会使用这份本地 CLIP 权重；否则会在首次使用时下载。模型权重不会提交到本仓库。

---

## 开发

运行测试：

```bash
python -m unittest discover -s tests -v
```

如果安装后 `import jev_sees` 失败，请确认 `python -m pip install -e .` 和运行程序时使用的是同一个 Python 解释器。

JEV Sees 最初是 [JEV Control Your Roboarm](https://github.com/CharlesFeng0314/JEV_control_your_roboarm) 中的视觉部分，现在已经成为一个可独立用于图片、视频、相机和机器人的项目。

项目当前版本为 **v0.1.0**，仍处于实验阶段；公开 API、视觉方案和示例还会继续迭代。

## License

MIT
