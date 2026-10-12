[English](README.md) | **中文**

# JEV Sees

### 让 JEV 看懂图片和视频，把视觉判断直接交给程序。

**JEV 很会判断：快、结构化，而且天然适合做封闭式决策。**

[JEV](https://typesafe.ai/blog/introducing-system-one-models-and-jev) 是 [TypeSafe AI](https://typesafe.ai/) 发布的首个 **System One Model**，专门面向软件中的快速、结构化决策。JEV、System One API 和 `typesafe-sdk` 都由 TypeSafe AI 提供；**JEV Sees 是我基于这些能力开发的独立开源项目，并非 TypeSafe AI 的官方视觉产品。**

JEV Sees 衍生自我的另一个项目：[JEV Control Your Roboarm](https://github.com/CharlesFeng0314/JEV_control_your_roboarm)。我最初在那个项目里让 JEV 根据视觉信息控制机械臂；现在，我把视觉能力从单一机器人场景中抽离出来，做成一个可以复用在图片、视频、监控和机器人上的产品。

视觉感知在本地完成：原始像素会被转换成带 bbox、标签、稳定 object ID、位置历史和空间关系的场景状态，再交给 JEV 判断。你定义问题和选项，JEV Sees 负责让每一次视觉判断都有对象、有上下文、能持续、可直接写进程序。

**一帧画面，多个对象，一次 JEV 调用。**

![JEV Sees 行人风险 Demo](assets/demo.gif)

> 上面的 Demo 里，同一个采样帧中所有可见行人的事故风险，都由一次 JEV 调用一起返回。

- 图片、视频和 RGB-D 使用同一个入口
- 每个对象都有 bbox、标签和跨帧稳定的 object ID
- 每个采样帧的多个对象合并进一次 JEV 调用
- 结果可以直接输出到终端、JSON、Excel 或实时可视化面板

---

## 从一帧图片，到持续运行的视觉判断

JEV Sees 把问题绑定到画面里的具体对象，并让这个问题跟着 object ID 穿过连续帧。每个采样时刻都会得到明确的 choice、probability 或 score，应用可以立刻显示、记录、设阈值或触发下一步动作。

你可以持续问：

- 这一帧里每个行人正在发生车祸的概率是多少？
- 机器人应该关注哪个物体？
- 目标的状态有没有发生变化？
- 画面里哪个对象最符合这个条件？
- 这一帧里所有目标能不能一次完成判断？

同一个流程可以用于单张图片，也可以在视频、监控、自动化测试和机器人中持续运行。问题和选项由你控制，对象跟踪、场景上下文、批量调用和结果整理交给 JEV Sees。

---

## 运行示例

仓库自带一张 bus 图片和一段 traffic 视频。示例都保持得很小：调用真实 JEV，并把结构化结果打印在终端。它们分别展示直接使用官方 `Choice`、视频中自动生成的逐对象概率，以及 animal motion 问题中的语义类别路由。

### Bus：在用户给定的颜色选项中判断

[examples/bus_color.py](examples/bus_color.py) 先检测 bus，再以该 bus 的 object ID 创建官方 `Choice`。颜色选项由示例代码明确给出：yellow、red、white、blue、black、uncertain。

```bash
python examples/bus_color.py
```

预期终端输出：

```text
object_id   label  choice  confidence
object_001  bus    blue    1.00
```

### Traffic：每个采样帧中，每个行人的实时事故概率

[examples/traffic_relations.py](examples/traffic_relations.py) 只提供一个英文问题和语义类别 `pedestrian`。JEV 负责把 person、man 等视觉标签映射到这个类别；JEV Sees 自动把问题绑定到当前帧的每个 object ID。类别判断和事故概率在每个采样帧共用一次 JEV 调用。

```bash
python examples/traffic_relations.py
```

真实 JEV 调用的输出形如：

```text
frame_index  video_time_s  object_id   label   scope_probability  probability
504          42.00         object_001  person  0.98               0.04
510          42.50         object_001  person  0.99               0.05
```

这个概率回答的是“当前帧中，该行人正在发生 car accident 吗？”，不额外预测未来两秒。object ID 会在连续采样帧间保持稳定；新的行人才会得到新的 ID。

问题的控制权始终在应用侧：调用者明确选择 `Choice`、`Noul` 或 `Score`；使用 `Choice` 时，候选项也必须由调用者明确给出。JEV Sees 不会从自然语言里猜问题类型，也不会替用户发明选项。

### Animal motion：每个 animal 一份 Choice 概率分布

[examples/animal_motion.py](examples/animal_motion.py) 通过 `MEDIA` 常量指定动物图片或视频。视觉模型可以把对象标成 `rabbit`、`dog` 或 `horse`；JEV 判断它是否属于用户指定的 `animal` 类别，再为每个匹配对象返回用户定义的运动状态概率。

```bash
python examples/animal_motion.py
```

---

## Quick Start

### 1. 安装

```bash
git clone https://github.com/CharlesFeng0314/JEV_sees.git
cd JEV_sees
python -m pip install -e .
```

需要 Python 3.10+。

### 2. 配置 JEV API Key

到 [console.typesafe.ai/keys](https://console.typesafe.ai/keys) 创建一个 key。

macOS / Linux：

```bash
export TYPESAFE_API_KEY="your-key"
```

PowerShell：

```powershell
$env:TYPESAFE_API_KEY = "your-key"
```

官方 `TypeSafeClient` 和视频高级入口都可以读取这个 key；`observe()` 与 `state()` 本身仍然只是本地视觉流程。

### 3. 跑最小示例

```bash
python examples/bus_color.py
```

仓库自带图片上的预期终端结果：

```text
object_id   label  choice  confidence
object_001  bus    blue    1.00
```

示例内部使用 `Sees()(image, question, questions)` 完成视觉感知、state 构建和官方 JEV 调用。完整代码：[examples/bus_color.py](examples/bus_color.py)

---

## 感知、跟踪和场景记忆

`observe()` 是 JEV Sees 的本地视觉层：

```python
tracks = sees.observe("assets/bus.jpg")

for obj in tracks:
    print(obj["object_id"], obj["label"], obj["bbox_xyxy"])
```

每个被跟踪的对象会带上一组可以继续被程序使用的信息，例如：

```text
object_id
label
confidence
bbox_xyxy
centroid_uv
attributes.color_evidence.cv
attributes.color_evidence.clip
```

颜色在 state 中是证据，不是 SDK 提前拍板的答案。CV 分支保留像素测量，CLIP 保留完整颜色概率分布。Florence region caption 不再发送给 JEV，避免重复标签或注入未经验证的物体关系。

连续处理视频帧时，JEV Sees 会在内部跟踪记忆中尽量维持稳定的 object ID。发送给 JEV 的 state 只包含当前采样帧实际检测到的对象；当前对象最近两次带 `frame_index` 和视频时间的 pose 会进入 state，因此 JEV 可以根据框的位置变化和真实帧间隔判断运动，同时不会看到已经消失的旧对象。

场景里还可以继续带上：

- 当前哪些对象可见
- 两个框是否重叠、间距多大
- 两个对象中心点距离
- 相邻对象是否正在靠近
- RGB-D 下的三维位置与物体间实际距离

这些属于实现层细节。真正重要的是它带来的产品效果：**JEV 可以持续对一个变化中的视觉场景做结构化判断。**

### `given_that` / state 的长度控制

JEV Sees 发送给 JEV 的不是视频帧本身，而是由 `state()` 生成的结构化视觉上下文。当前实现遵守 JEV 的 31,000 token 输入硬限制，并把序列化后的 state 控制在更保守的 **24,000 字符目标预算**内；这里按“一个字符等于一个 token”估算，主动为问题和协议开销留出空间。

长度收缩会保留当前帧的每个 object ID，再依次压缩关系、pose history 和可选属性。如果当前帧最小必要状态仍然超限，会明确报错，而不是静默删除已经绑定问题的对象。`state["prompt_budget"]`、`sees.last_prompt_chars` 和展示面板右下角都可以看到实际序列化长度。

---

## 支持哪些问题类型？

`jev_sees` 会直接重新导出官方的 `Choice`、`Noul`、`Score` 和 `TypeSafeClient`，因此只需要一行 import。它们仍然是 `typesafe-sdk` 的原始类；直接调用仍写 `client.system_one(...)`，输出仍从 `response.choices`、`response.nouls` 和 `response.scores` 读取。

高级逐对象 API 会通过方法名明确选择输出类型，并自动创建官方问题：

```python
Sees().probability(video, "Is this object blocking the road?")

Sees().for_each("animal").choice(
    video,
    "What is this object's current motion state?",
    choices=["stationary", "walking", "running", "uncertain"],
)
```

不调用 `for_each(...)` 时，输出当前帧检测到的全部对象；调用后，JEV 会额外返回每个对象的类别匹配概率，SDK 按 `scope_threshold` 组织主结果，并在结构化结果中保留原始 scope probability。应用如果需要自己构造官方 `Choice`、`Noul` 或 `Score`，仍然可以继续使用底层的 `Sees()(source, question, questions)` 入口。

---

## 视频：每个采样帧、每个行人一个实时概率

[examples/traffic_relations.py](examples/traffic_relations.py) 保持为只向终端输出的最小示例。应用只提供 `pedestrian` 语义类别和一个英文事故问题，SDK 自动构造每个 ID 的问题。JEV Sees 内部没有 traffic plan，也没有内置 car accident 的定义。

视频结果以 `result.frames` 为主：每个被实际判断的采样帧带有 `frame_index`、`video_time_s`，以及该帧所有被选对象的概率。`result.rows` 仍保留每个对象的峰值摘要，方便兼容已有代码；JSON 和 Excel 导出同样把帧数据放在首要位置。

默认视频采样频率约为每秒两次。传入 `every=1` 可以逐帧判断，传入 `every=N` 则每 N 帧判断一次。只有采样帧才会运行感知和 JEV 调用。同一采样帧中所有被选择的对象合并成一次 JEV 请求。

使用 `save=` 时，输出会绘制带物体标签、object ID 和实时答案的 bbox，并加入右侧统计面板。某个被选择对象首次出现后，它的行会固定保留；概率任务显示当前值、峰值和样本数，Choice 任务显示选择与 confidence，右下角同时显示 state 长度。

运行：

```bash
python examples/traffic_relations.py
```

这个示例会调用 JEV，因此需要 `TYPESAFE_API_KEY`。

视频来源与授权：[assets/CREDITS.md](assets/CREDITS.md)

---

## RGB-D：不只相信 2D 检测框

RGB 模式从 Florence-2 dense-region caption 开始。Florence 自己发现区域、拉框并生成自由文本标签，调用者不传物体 vocabulary。

RGB-D 模式则换了一条路线：

**先用深度聚类决定“这里到底有没有一个物体”，再用与它重叠的 Florence region 提供自由文本名称。**

这件事的价值很直接：即使 2D detector 漏掉一个真实存在的物体，深度仍然可能把它从空间里“捞出来”。

| | | |
| --- | --- | --- |
| ![第 6 帧](assets/frame_06.png) | ![第 13 帧](assets/frame_13.png) | ![第 15 帧](assets/frame_15.png) |

如果同时传入相机内参，RGB-D 观测还可以得到 `position_m`，于是场景里不只有“它在图片哪里”，还可以有**米制三维位置和物体间距离**。

---

## 性能

以前的 YOLO benchmark 已经不能代表 Florence-2 pipeline，因此当前文档不再引用那组延迟。这个后端需要重新做图片与视频 benchmark 后才能发布性能结论。

---

## 模型与权重

权重没有直接放进仓库。

Florence-2 会在第一次使用时从 Hugging Face 加载。默认模型是 `microsoft/Florence-2-base-ft`；也可以替换兼容 checkpoint：

```python
Sees(florence_model="microsoft/Florence-2-large-ft")
```

如果 `JEV_SEES_ROBO_ROOT` 指向的目录里已经有：

```text
weights/clip/ViT-B-32.pt
```

JEV Sees 会把这份本地 CLIP 权重用于颜色证据；否则 CLIP 会在第一次使用时下载权重。

---

## 常见问题

如果：

```bash
python -c "import jev_sees"
```

报错，最常见的原因不是 JEV Sees 本身，而是你刚才运行 `pip` 的 Python 环境和现在这个 `python` 不是同一个。

建议始终用：

```bash
python -m pip install -e .
python -c "import jev_sees; print(jev_sees.__version__)"
```

这样安装和运行一定走同一个解释器。

测试：

```bash
python -m unittest discover -s tests -v
```

---

## 我从哪里开始

我先做了 [JEV Control Your Roboarm](https://github.com/CharlesFeng0314/JEV_control_your_roboarm)，把 TypeSafe AI 的 JEV 接到机械臂动作选择上。做到视觉部分时，我意识到这套能力不应该只属于一台机械臂，于是把它抽离、泛化，做成了 JEV Sees。

```text
TypeSafe AI / JEV
        │
        ├── JEV Control Your Roboarm：JEV 判断 → 机械臂动作
        │          │
        │          └── 抽离并泛化视觉能力
        │                         ↓
        └──────────────────── JEV Sees：图片 / 视频 / RGB-D → 结构化视觉状态
```

- **[JEV Control Your Roboarm](https://github.com/CharlesFeng0314/JEV_control_your_roboarm)**：我的原始项目，把 JEV 的判断接到机械臂动作选择上
- **JEV Sees**：我由此衍生出的独立视觉项目，把图片、视频和 RGB-D 转成 JEV 可判断的结构化上下文

它们组合起来仍然是同一个方向：**see → judge → act**。

---

## 当前状态

JEV Sees 目前是 **v0.1.0**，仍然是一个实验性项目。

对外接口仍然很小：`Sees(...)` 是图片/视频的产品级入口，`observe()` 和 `state()` 则把视觉层暴露给官方 JEV 调用。JEV 的官方类只做原样重新导出。底层视觉方案、scene representation、示例和 benchmark 都还会继续迭代。

如果你把它接到了另一台相机、另一种机器人，拿去看一个很奇怪的场景，或者想出了当前示例完全没覆盖的用法，欢迎直接开 Issue / Discussion。

**这个项目现在最需要的，不是更多“看起来完整”的文档，而是更多人真的拿去玩。**

---

## License

MIT
