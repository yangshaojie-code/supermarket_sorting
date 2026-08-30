# A 线：仿真采图与 YOLO 训练

负责人：数据采集 / 训练。代码与权重最终都进 `D:\DG\supermarket_sorting_baseline`。不要在 `vlm_pipeline` 里训零售模型。

新 Agent 先读 [AGENTS.md](./AGENTS.md)。和 B 的边界、9 类名字、合并节点见 [DG-202606-分工总览.md](./DG-202606-分工总览.md)。

## 当前情况

| 项 | 状态 |
| --- | --- |
| 官方权重 | 仅 `weights/kele.pt`，一类 `kele` |
| 检测后端 | `perception/yolo_backend.py` 写死 `CLASS_NAMES = ["kele"]`，等你的 9 类权重再改（**改这一行属交接，A 先不要改评分路径**） |
| 采图脚本 | **还没有** |
| 训练集 | **还没有** |
| P3 预览 | 能加载 YOLO；机器人在送货区时看不到货架，空检测是视角问题 |
| 训练策略 | 不要从零训、不要 Genesis、不要评测时联网。用 COCO 预训练 **YOLOv8s**（或 `n` 做本机试跑）在本仿真 9 类上微调 |

本机 RTX 4060 笔记本：适合采图、试跑 `yolov8n`、推理。主训建议租 **4090（约 20GB）4 天**：有效训练一般 **1～3 小时/轮**，4 天是为了 GS=0/1 两套外观和难例复训，不是因为一张卡要算满 96 小时。

## 任务目标

做出能在官方 Client 里离线加载的 9 类检测器，类别字符串与任务 JSON 的 `kind` 完全一致。

交付物（放到零售仓库）：

```text
supermarket_sorting_baseline/
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

- 头部 RGB（话题 `/head_camera/color/image_raw`，640×480）
- 可选深度、`camera_info`、odom、头部关节、当时 `/supermarket_sorting/task`
- 文件名带时间戳，方便和标注对齐

调试采图：`SUPERMARKET_USE_GS=0`、`FIXED_BASELINE=1`、`RANDOMIZE=0`、`TASKS=product_032`。  
正式外观再采一批：`SUPERMARKET_USE_GS=1`、多种子 `SUPERMARKET_RANDOMIZE=1`。

B 线还没把车开到货架前时，A 可以：请 B 先给一个「面向货架停住」的最短动作；或先用键盘/临时 `cmd_vel` 把车开到拣货区 y∈[1.70, 3.25] 再录。对着送货台录的图对货架检测几乎没用。

### 2. 标注：训练可用仿真真值，推理禁用

**允许：** 单独的训练标注工具读 MuJoCo/3DGS 物体位姿，投影到头相机，写出 YOLO txt。这不是评分感知，不要进 `runtime/orchestrator.py`。

**不允许：** 评分 Client 解析任务 JSON 里未下发的坐标字段当货位真相。

若短期内拿不到投影标注：固定布局下 `product_032` 对应槽位 **ArUco 32 = D-L2-C3**、类别 `kele`，可先做可乐单类扩充；9 类必须换多种子随机布局，手标几千张不现实。优先把投影脚本做出来。

规模：先 **2k～5k** 张，货架 A–E、三层、远近、轻微遮挡都要有。`GS=0` 与 `GS=1` 最好分开目录，最后按比例混进 train/val。

### 3. 训练

与评测一致：Python 侧尽量用 **ultralytics 8.0.196** 导出的 `.pt`，避免 8.1+ 才有的模块。

```text
yolo detect train \
  model=yolov8s.pt \
  data=datasets/supermarket9/data.yaml \
  imgsz=640 epochs=80 batch=16 \
  device=0
```

4060 把 `batch` 降到 4～8，或改 `yolov8n.pt` 做通路测试。4090 上 YOLOv8s、batch 16～32 通常一轮 **约 1～3 小时**。

网上可口可乐/苹果照片对初赛帮助有限；域必须是 **本仿真头相机**。

### 4. 接到 P3 验证（A 自测，B 改 CLASS_NAMES）

1. 权重拷到 `weights/supermarket9.pt`
2. 通知 B 扩展 `YoloBackend.CLASS_NAMES`
3. 机器人面向货架后跑 `scripts/run_p3_preview.sh`
4. 记录：9 类里哪些稳、哪些和脉动/可乐、橙/苹果混淆

## 明确不要做

- 评测镜像里 `pip install` 新检测库、升级 ultralytics 到 YOLO-World
- 把 DashScope / Qwen 当零售检测
- 从零训大检测器或占用 4090 做和货架无关的预训练
- 改 `client_task_1.py`、编排、限幅控制（那是 B 的范围）
- 文旅和零售 Docker 同时开（同一 `ROS_DOMAIN_ID=99`）

## A 的完成标准（可以叫 B 接入）

- [ ] 采图脚本可复现跑
- [ ] train/val 划分清楚，names 与总览 9 类一致
- [ ] `supermarket9.pt` 在 Client 容器内能被现有 `YoloBackend` 加载（改完 CLASS_NAMES 之后）
- [ ] 对着货架的 P3 预览里，至少可乐 + 另外 2～3 类有框
- [ ] 已知失败模式写进本文件末尾（混淆对、GS 开关差异）

## 已知失败模式（交接后两人补）

- （待填）
