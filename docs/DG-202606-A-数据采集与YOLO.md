# A 线：仿真采图与 YOLO 训练

负责人：数据采集 / 训练。代码与权重最终都进零售仓库（当前是 `D:\DG\supermarket_sorting`）。不要在 `vlm_pipeline` 里训零售模型。

更新：2026-09-01。

新 Agent 先读 [AGENTS.md](./AGENTS.md)。和 B 的边界、9 类名字、合并节点见 [DG-202606-分工总览.md](./DG-202606-分工总览.md)。

**写采图脚本前先读下面「4060 与 3DGS 分批」。** 评测头相机是 3DGS 贴图，不是红几何体；本机 8GB 显存一次装不下全部 9 类 ply。

## 当前情况

| 项 | 状态 |
| --- | --- |
| 官方权重 | 仅 `weights/kele.pt`，一类 `kele` |
| 检测后端 | `perception/yolo_backend.py` 写死 `CLASS_NAMES = ["kele"]`，等你的 9 类权重再改（**改这一行属交接，A 先不要改评分路径**） |
| 采图脚本 | **还没有** |
| 训练集 | **还没有** |
| P3 预览 | 车在货架前、`GS=1` slim 可乐时：ArUco 能解、`kele.pt` 约 0.94～0.95。P3 必须 **CPU** 推理 |
| 训练策略 | 不要从零训、不要 Genesis、不要评测时联网。用 COCO 预训练 **YOLOv8s**（或 `n` 做本机试跑）在本仿真 **GS=1 外观** 上微调 |

本机 RTX 4060 笔记本：适合 **分批 GS 采图**、试跑 `yolov8n`。主训建议租 **4090（约 20GB）**：有效训练一般 **1～3 小时/轮**。不要在 4060 上对 `SUPERMARKET_GS_KINDS=all` 开 Server。

## 4060 与 3DGS 分批（采图硬约束）

### 结论

1. YOLO 必须用 **`SUPERMARKET_USE_GS=1` 的头相机图** 训练。`GS=0` 是无贴图红圆柱/方块，和评测外观不是同一域；现有 `kele.pt` 在红块上对不上，在可乐 GS 上约 0.95。
2. 本机 **RTX 4060 Laptop 8.6GB 不能一次加载 45 件商品 ply + 背景 + 机器人**。全量会在 `GSRendererMuJoCo` 构造时被内核 **SIGKILL**（无 Python traceback）。
3. 换要拍摄的类别 = 改 `SUPERMARKET_GS_KINDS` + **重启 Server**。不能热更新。

### 原因

官方 `GS=1` 把高斯点云一次性塞进渲染器：机器人（mmk2 + 双臂）、超市背景 `retail_background_fit.ply`（**货架和 ArUco 在背景 ply 里**）、9 类 × 5 = 45 件商品 ply、可选障碍箱。

补丁 `patches/simulator.py` 在显存 &lt; 12GB 时会 slim：默认只留机器人 + 背景 + `SUPERMARKET_GS_KINDS`（默认 `kele`）。实测加载后约 0.4～0.9GB，但每帧渲染还会再涨。以前对头/双腕/窗口四路一起做 GS，更容易爆。

还会把 Server 挤死的情况：

- Client 把 YOLO 放到 `cuda:0`（和 Server 抢同一张卡）
- 即使 YOLO 设成 CPU，采图进程里 `import torch` 仍可能建 CUDA 上下文 → 采图脚本设 `CUDA_VISIBLE_DEVICES=`
- gsplat 首次编译 CUDA 时镜像默认 `MAX_JOBS=10` 会打满内存；补丁已压到 2。看到 `Setting up CUDA` 要等到 `[gs] frame0`

**不要用 Server 可视化窗口判断贴图是否加载。** 4060 上窗口被改回 MuJoCo 以省显存，窗口里仍是红块是正常的。训练只信 `/head_camera/color/image_raw`。

### 采图时怎么开 Server

```text
SUPERMARKET_USE_GS=1
SUPERMARKET_GS_KINDS=kele          # 一次一类，或逗号分隔极少几类
# 不要 SUPERMARKET_GS_NO_BACKGROUND=1   # 关掉则没有货架/ArUco，画面近乎全黑
# 不要 SUPERMARKET_GS_KINDS=all         # 8GB 必 OOM
```

一次只保证 **当前 KINDS 里的类别是真贴图**。其余类别在同一帧里会消失或仍是红块。**不要把红块标成其它 `kind`。**

建议 9 类分 9 次（一批 2～3 类若又 OOM 就减回 1 类）：

```text
kele → maidong → kouxiangtang → shupian → zhijin
     → sanmingzhi → heweidao → pingguo → chengzi
每批：
  1. 停 Client（尤其不要 GPU 上的 YOLO/训练）
  2. 重启 Server（新的 SUPERMARKET_GS_KINDS=该类）
  3. 等 [gs] product batch / renderer ready / frame0
  4. 车开到货架前（拣货区 y ∈ [1.70, 3.25]），再采头相机
  5. 只标注这一批已加载贴图的商品
```

确认贴图在头相机里：能看到罐装可乐/对应商品，而不是光滑红圆柱。`rgb_mean` 若只有 1～2（几乎全黑），多半开了 `SUPERMARKET_GS_NO_BACKGROUND=1`。

评测机或 4090 才可能尝试一次加载更多类别。本机 4060 不要赌 `all`。

## 任务目标

做出能在官方 Client 里离线加载的 9 类检测器，类别字符串与任务 JSON 的 `kind` 完全一致。

交付物（放到零售仓库）：

```text
supermarket_sorting/
  datasets/supermarket9/
    images/{train,val}/
    labels/{train,val}/     # YOLO txt：class cx cy w h（归一化）
    data.yaml
  weights/supermarket9.pt   # 交接文件名；若改名请写在总览里
  docs 或本文件更新：训练命令、epoch、val mAP、已知混淆类
```

`data.yaml` 示例（names 顺序 = class id）：

```yaml
path: /workspace/baseline/datasets/supermarket9
train: images/train
val: images/val
names:
  0: sanmingzhi
  1: heweidao
  2: shupian
  3: zhijin
  4: maidong
  5: kouxiangtang
  6: pingguo
  7: chengzi
  8: kele
```

## 建议做法（按顺序）

### 1. 写 Client 侧采图脚本（必做）

在零售仓库新增例如 `scripts/capture_yolo_frames.py` + `scripts/run_capture.sh`（LF）。

每帧保存：

- 头部 RGB（话题 `/head_camera/color/image_raw`，640×480）——**只保存这一路当训练图**
- 可选深度、`camera_info`、odom、头部关节、当时 `/supermarket_sorting/task`、当时 `SUPERMARKET_GS_KINDS`
- 文件名带时间戳，方便和标注对齐
- 进程开头：`export CUDA_VISIBLE_DEVICES=`，避免和 Server GS 抢卡

脚本通路可用 `GS=0` 测「能否写盘」。**正式训练集只用 GS=1 分批图。** 固定布局打通：`FIXED_BASELINE=1`、`RANDOMIZE=0`、`TASKS=product_032`。9 类再 `RANDOMIZE=1` 多种子。

B 已能把车开到货架（`scripts/run_drive_to_shelf.sh`）。对着送货台录的图对货架检测几乎没用。

### 2. 标注：训练可用仿真真值，推理禁用

**允许：** 单独的训练标注工具读 MuJoCo/3DGS 物体位姿，投影到头相机，写出 YOLO txt。这不是评分感知，不要进 `runtime/orchestrator.py`。投影时只给 **当前已加载 GS 的那些 body** 写框。

**不允许：** 评分 Client 解析任务 JSON 里未下发的坐标字段当货位真相。

若短期内拿不到投影标注：固定布局下可见的可乐可能在 **D-L2-C2（码 31）** 附近，任务 JSON 只有 `kind=kele`。名义 `product_032` 是 D-L2-C3。以头相机里实际罐子为准。9 类必须换多种子随机布局，手标几千张不现实。优先把投影脚本做出来。

规模：先 **2k～5k** 张，货架 A–E、三层、远近、轻微遮挡都要有。按 `GS_KINDS` 分目录，最后合并 train/val。不要把 `GS=0` 红块图混进正式 train。

### 3. 训练

与评测一致：Python 侧尽量用 **ultralytics 8.0.196** 导出的 `.pt`，避免 8.1+ 才有的模块。

```text
yolo detect train \
  model=yolov8s.pt \
  data=datasets/supermarket9/data.yaml \
  imgsz=640 epochs=80 batch=16 \
  device=0
```

4060 上训练时 **先停 Server**（否则显存不够）。把 `batch` 降到 4～8，或改 `yolov8n.pt` 做通路测试。4090 上 YOLOv8s、batch 16～32 通常一轮 **约 1～3 小时**。

网上可口可乐/苹果照片对初赛帮助有限；域必须是 **本仿真头相机 + GS=1**。

### 4. 接到 P3 验证（A 自测，B 改 CLASS_NAMES）

1. 权重拷到 `weights/supermarket9.pt`
2. 通知 B 扩展 `YoloBackend.CLASS_NAMES`
3. 机器人面向货架、对应 `GS_KINDS` 已加载后跑 `scripts/run_p3_preview.sh`（保持 CPU）
4. 记录：9 类里哪些稳、哪些和脉动/可乐、橙/苹果混淆

## 明确不要做

- 用 `GS=0` 红几何体当 9 类训练集主体
- `SUPERMARKET_GS_KINDS=all` 或一次加载 45 件 ply（4060）
- `SUPERMARKET_GS_NO_BACKGROUND=1` 采训练图（没有货架/ArUco）
- Server GS=1 还在跑时，在 Client 里把 YOLO/PyTorch 放到 GPU
- 把未加载贴图的红块标成 `kind`
- 评测镜像里 `pip install` 新检测库、升级 ultralytics 到 YOLO-World
- 把 DashScope / Qwen 当零售检测
- 从零训大检测器或占用 4090 做和货架无关的预训练
- 改 `client_task_1.py`、编排、限幅控制（那是 B 的范围）
- 文旅和零售 Docker 同时开（同一 `ROS_DOMAIN_ID=99`）

## A 的完成标准（可以叫 B 接入）

- [ ] 采图脚本可复现跑，且 **按 `SUPERMARKET_GS_KINDS` 分批重启 Server**
- [ ] train/val 划分清楚，names 与总览 9 类一致；图都是 GS=1 头相机
- [ ] `supermarket9.pt` 在 Client 容器内能被现有 `YoloBackend` 加载（改完 CLASS_NAMES 之后）
- [ ] 对着货架的 P3 预览里，至少可乐 + 另外 2～3 类有框（验证时 Server 要加载对应 KINDS）
- [ ] 已知失败模式写进本文件末尾

## 已知失败模式

- **全量 GS OOM：** 4060 上加载全部商品 ply 时进程无报错退出。改用 `SUPERMARKET_GS_KINDS` 分批。
- **YOLO cuda 挤死 Server：** P3/采图与 GS=1 Server 同卡。Client 用 CPU + `CUDA_VISIBLE_DEVICES=`。
- **无背景 = 无货架：** `SUPERMARKET_GS_NO_BACKGROUND=1` 时 `rgb_mean` 极低、ArUco 空。采图不要关背景。
- **窗口仍是红块：** 4060 上 GLFW 窗口不做 GS。看头相机话题，不要看窗口。
- **只加载可乐时其它类是红块或消失：** 不要标注它们。换 `GS_KINDS` 再采。
- **P3 空框：** 先确认车在拣货区对着货架，再查 GS 是否在头相机上。
