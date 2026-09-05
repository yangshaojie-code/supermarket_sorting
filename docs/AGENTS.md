# AGENTS.md

给后续 Agent 和协作者的**第一份文件**。

更新：2026-09-01。当前比赛 **DG-202606 智慧零售**，初赛提交 **2026-09-15**。

---

## 导读：文档在哪、先读哪份

零售文档有两处同名拷贝。**改评分代码、开新 Agent 时以代码仓为准：**

`D:\DG\supermarket_sorting_baseline\docs\`

`D:\DG\vlm_pipeline\docs\` 是较早的团队文档落点，文件名相同。两份不一致时，**只改代码仓 `supermarket_sorting_baseline/docs`，不要在文旅仓库里并行改一版。** 评分源码仍在 `supermarket_sorting_baseline/` 根下（`runtime/`、`perception/` 等），不要把 Client 写进 `vlm_pipeline/`。

工作区根目录不再放 `AGENTS.md`。零售仓库根目录的 `supermarket_sorting_baseline/AGENTS.md` 只是指针，会指到 `docs/AGENTS.md`。

### `supermarket_sorting_baseline/docs` 里每份做什么

在代码仓打开文档时，下面相对路径就是当前文件夹。在文旅 `vlm_pipeline/docs` 打开时，同名文件用途相同，但请改去代码仓那份。

| 文件 | 什么时候用 | 不要当成 |
| --- | --- | --- |
| [AGENTS.md](./AGENTS.md)（本文） | 第一份：比赛要求、背景、禁区、进度、文档地图 | 赛题 PDF 原文；不要在这里贴大段代码 |
| [DG-202606-AGENT-BRIEF.md](./DG-202606-AGENT-BRIEF.md) | 动手前：仓库地图、Docker/WSL、话题、P3 空检测、Agent 工作方式 | 分工日历（看总览）；官方 `docker run` 全文（看仓库 README） |
| [DG-202606-分工总览.md](./DG-202606-分工总览.md) | 两人拆分、交接物（`supermarket9.pt`）、日历 | 具体采图/导航步骤（看 A/B） |
| [DG-202606-A-数据采集与YOLO.md](./DG-202606-A-数据采集与YOLO.md) | A 线：采图脚本、仿真标注、YOLOv8s、交付权重。**4060 必须分批加载 3DGS ply** | 编排/限幅/`client_task_1.py` |
| [DG-202606-B-迁移与算法.md](./DG-202606-B-迁移与算法.md) | B 线：走到货架、雷达、抓放、评分入口 | A 的 `data.yaml` 类别顺序和训练超参 |
| [DG-202606-P4-导航与全局规划.md](./DG-202606-P4-导航与全局规划.md) | P4：先规划全图思路与实现逻辑；「已拍板」之前不写代码 | 赛题 PDF；Nav2 安装说明；未批准的实现任务 |
| [准备工作.txt](./准备工作.txt) | 宿主机/WSL：Docker、NVIDIA 驱动、Container Toolkit | 比赛任务说明 |
| [server物品清单.csv](./server物品清单.csv) | 核对 9 个 `kind` 与仿真模型路径 | YOLO 训练集（那是 `datasets/`，还没有） |

同仓库、但不在 `docs/` 里：

| 路径 | 用途 |
| --- | --- |
| `supermarket_sorting_baseline/README.md` | 官方启动命令、话题表、环境变量（`SUPERMARKET_*`） |
| `supermarket_sorting_baseline/AGENTS.md` | 短指针，指向 `docs/AGENTS.md` |
| `D:\DG\DG-202606智慧零售赛题说明详细版_20260801\` | 赛题 PDF / 变更说明原文 |

阅读顺序：本文 → BRIEF → 按任务选 A 或 B。P4 先读 [DG-202606-P4-导航与全局规划.md](./DG-202606-P4-导航与全局规划.md) 做规划，未写入「已拍板方案」前不要改导航代码。不要从文旅 `vlm_pipeline` 的 Task 脚本开始。

---

## 比赛要求（DG-202606）

赛题全称方向：**面向智慧零售的自主服务机器人**。评测形态是官方 Docker 里的 **Server 仿真 + 选手 Client**，不是把 Genesis World 换成评测场。

### 要做什么

在限定仿真时间内，控制 **MMK2** 移动双臂机器人，按 Server 下发的订单，从超市货架上**取出指定种类商品，送到配送桌**，能送几件算几件。

- 货架 **A–E**，每组 **3 层 × 3 列**，共 **45 个货位**。
- 货位用 ArUco **DICT_4X4_50**、ID **0–44**、边长 **3 cm**。码绑的是**槽位**，不是商品 SKU。
- 商品 **9 类 × 5 件 = 45**：`sanmingzhi heweidao shupian zhijin maidong kouxiangtang pingguo chengzi kele`（三明治、核味道、薯片、纸巾、脉动、口香糖、苹果、橙子、可乐）。字符串必须与任务 JSON 的 `kind` 逐字一致。
- 通道里 **5 个随机障碍**（箱体保持竖直，只随机偏航；官方保证货架入口到配送台入口有通路）。必须用二维雷达 `/slamware_ros_sdk_server_node/scan` 自己规划。官方 baseline 的固定航点**不算**正式解。
- 作业模式：**取一件 → 立刻送到配送桌 → 再取下一件**。正式局约 **5 个订单**。不要一次抓一堆再统一送。
- 时限：仿真（MuJoCo）时钟 **420 s**，不是墙钟。失败后**不得**重置 Server、布局或种子，只能从当前物理状态重试。

任务话题 `/supermarket_sorting/task` 示例：

```json
{"schema_version":1,"run_prefix":"run_a1b2c3d4e5f6","count":2,"targets":[{"id":"item_run_a1b2c3d4e5f6_01","kind":"kele"},{"id":"item_run_a1b2c3d4e5f6_02","kind":"kele"}]}
```

**只保证有 `id` 和 `kind`。不下发世界坐标。** 旧赛题简报里的位置字段若在调试 Server 里出现，也不得当推理真值。货在哪：自己看头相机、ArUco、深度；送到哪：固定配送区（见 `runtime/scene_zones.py`，桌约 `(-1.940, -3.410)`，目标高度约 `0.807 m`）。

裁判侧与搬离/放置相关的量级（已写入 `scene_zones.py`）：搬离水平位移约 **0.20 m**，放置区 `delivery_box` 等。以 Server 内置裁判为准。

### 怎么得分（初赛口径）

正式成绩由 **Server 裁判** 判，不是 Client 自己 print。方案口径：每成功完成一个取送循环大约 **12 分**，5 单大约 **60 分**，另有技术分。完整 5 单 + 9 类 + 平滑绕障在 9 月 15 日前很紧，策略是先稳定 **1～2 个循环** 再扩。

终审（约 11 月）会到真机，和仿真是**第二域**。现在用仿真数据训的 YOLO 不能假设直接够用终审；那是下一阶段。

### 评测与部署硬约束

| 项 | 要求 |
| --- | --- |
| 系统 | Ubuntu 22.04，ROS 2 Humble，Cyclone DDS，`ROS_DOMAIN_ID=99` |
| 镜像 | `supermarket_sorting:server` / `:client`（官方 tar；id 曾见 `eb0b58a600b8` / `dbe0bfd2c75e`） |
| 选手程序 | 挂载到 Client 的 `/workspace/baseline`，容器内启动 |
| 网络 | **评测无网**，不能下载权重或 pip 新库 |
| 感知栈 | 镜像内 PyTorch 2.7.1+cu128，**ultralytics 8.0.196** |
| 控制 | `/cmd_vel`、升降/头/左右臂 `*_forward_position_controller/commands` |
| 传感器 | 头 RGB-D 640×480、腕部 RGB、`joint_states`（17 关节）、odom、scan 360 点 |

禁止当作评分方案：DashScope/Qwen、联网 VLM、Genesis World / GENE、把官方 Server 换成别的仿真器。`SUPERMARKET_USE_GS=1` 是官方 **3DGS 外观**，不是 Genesis。

官方 `client_task_1.py` 只在**固定布局**抓**一次可乐**，用来证明环境通。它不读 5 单、不扫五组货架、不绕随机障碍。正式程序要另写入口。

---

## 项目背景（我们为什么在这里）

工作区 `D:\DG` 里叠了两场同系列比赛，机器人都是 MMK2 + ROS 2 Humble + 同一套底盘/升降/头/双臂话题，所以容易写错仓库。

1. **先做的是文旅 DG-202612**（`vlm_pipeline`）：三色箱、指令带 `place_world`、双臂 hug、无雷达、DashScope。平台层（限幅、急停、`MMK2Kdl`、头相机外参）已经打磨过。
2. **零售初赛更早截止（9 月 15 日）**。文旅还没做完的部分（三任务闭环、hug 插货架、色箱 HSV）正好**带不进零售**。继续把文旅做完再迁，会把联调时间耗在错误物体和错误场景上。
3. **2026-08-29 起转向零售**：底座用官方 `supermarket_sorting_baseline`，把搬运工程的控制/IK/相机**思想迁过去**，任务层重写。文旅评分 Client 暂停。
4. **零售文档主副本在 `supermarket_sorting_baseline/docs/`**（与代码同仓）。`vlm_pipeline/docs/` 可能还有同名拷贝；不一致时以代码仓为准。文旅源码仓库本身不是零售 Client。

已完成 **P0–P3**（镜像、预检、取送状态机干跑、ArUco 绑定 + 可乐 YOLO）。缺两块核心，并已按人拆开：

- **A** 仿真采图 → 9 类 YOLO（评测能离线加载的 `.pt`）
- **B** 走到货架、雷达、编排接真抓放

训练出第一版 9 类权重后再合并。详细边界见分工总览。

机器：Windows 11 + **WSL Ubuntu-22.04** 跑 Docker/GPU（PowerShell 里通常没有 docker）。本机 RTX **4060**；可租 **4090 ~20GB × 4 天** 主训 YOLO。用户验证方式是**仿真里看**，不是浏览器。

---

## 立刻要遵守的规则

**做**

- 只改 `supermarket_sorting_baseline/` 里的评分代码。新逻辑放 `runtime/` 和新的 `scripts/run_*.sh`（**LF**）。
- 任务当 `id`+`kind` 解析；槽位 ArUco 0–44；商品本地 YOLO；导航用雷达。
- 底盘指令走 `runtime/ros_robot_control.py`（±0.35 / ±0.65）。
- `source /opt/ros/humble/setup.bash` **之后**才能 `set -u`。Server 调试挂载 `patches/simulator.py`。
- 改完：`cd supermarket_sorting_baseline && python -m unittest discover -v`。

**不要**

- 去实现或「完成」文旅 Task1/2/3、hug、HSV、DashScope。
- 把 `client_task_1.py` 改成正式评分主程序。
- 推理时把未下发坐标当货位真相（训练投影标注可以，写在训练脚本里）。
- 升级 ultralytics、引入 YOLO-World/Genesis 作为评测依赖。
- 文旅与零售容器共用一个 `ROS_DOMAIN_ID` 同时开。
- 看到 P3 的 `markers=[]` 就改检测：先确认车是否对着货架（常见卡在送货区 y≈−3.17）。

9 类名与 `runtime/task_protocol.py` 的 `KNOWN_KINDS` 必须一致。A 未交出 `data.yaml` 前，不要两个人一起改 `YoloBackend.CLASS_NAMES`。

---

## 当前进度（2026-09-05）

已在仿真跑过：预检；P2 干跑 1× `kele` → `DONE`；车开到货架后 P3 在 **GS=1 slim 可乐 + 背景** 下能解 ArUco、YOLO 认出 `kele` 并绑槽。P3 推理必须走 **CPU**（与 Server GS 共享 4060 会 SIGKILL Server）。

P4 全图（静态墙 + 雷达栅格 + A*）已在 `runtime/grid_planner.py`。`--goal shelf` 最新一次 `arrived=true`；`--goal delivery` 仍停在西侧箱群约 `(-1.47, 0.48)`。下一任先读 [DG-202606-P4-导航与全局规划.md](./DG-202606-P4-导航与全局规划.md) 的「仿真交接（2026-09-05）」，用文末「完善用」提示词。不要重开 Nav2 / DWA。

未做：9 类数据集与训练；配送导航闭环；真抓真放；5 单随机；评分主入口。

下一步：A 按 [DG-202606-A-数据采集与YOLO.md](./DG-202606-A-数据采集与YOLO.md) **分批 GS 采图**；B 先把 P4 送到配送台，再接可乐取放。操作细节以 [AGENT-BRIEF](./DG-202606-AGENT-BRIEF.md) 为准。
