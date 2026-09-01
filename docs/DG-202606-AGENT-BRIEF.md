# AGENT BRIEF — DG-202606 操作手册

比赛要求、项目背景、文档导读在同目录 **[AGENTS.md](./AGENTS.md)**。本文只讲怎么改代码、怎么跑、哪些文件能碰。

更新：2026-09-01。**先做零售初赛，不要去收尾文旅 Client。** 截止 **2026-09-15**。

## 60 秒结论

1. 工作区 `D:\DG` 里有两场比赛。当前只做 **DG-202606 智慧零售**。
2. 改代码只动 `supermarket_sorting_baseline/`。文档先读 `docs/AGENTS.md`（本仓库）。`vlm_pipeline/` 是暂停的文旅工程，不要当零售 Client。
3. 评分程序是 ROS 2 **Client**，挂进官方 Docker，评测 **无网**。感知必须是本地 `.pt`，禁止 DashScope / Qwen / Genesis World / YOLO-World（除非把兼容依赖整包进仓库）。
4. 任务 JSON **只有** `id` + `kind`，没有坐标。货位靠 ArUco 0–44，商品靠 9 类 YOLO，通道靠雷达。
5. 已完成 P0–P3 平台与预览；缺：9 类数据与训练、走到货架、雷达导航、编排接真抓放。
6. 官方 `client_task_1.py` 是「固定货位抓一次可乐」演示，**不要改成评分主程序**。新逻辑放 `runtime/`。

## 两场比赛（禁止混用）

| | 智慧零售 DG-202606（当前） | 文旅 DG-202612（暂停） |
| --- | --- | --- |
| 仓库 | `D:\DG\supermarket_sorting_baseline` | `D:\DG\vlm_pipeline` |
| 镜像 | `supermarket_sorting:server` / `:client` | `material_sorting` |
| 任务 | `/supermarket_sorting/task` | `/material/instruction` |
| 物体 | 9 类 × 5；`kind` 见下 | 粉/黄/褐三色箱 |
| 定位 | ArUco `DICT_4X4_50` id 0–44，3 cm，绑槽位不是绑商品 | 无码；指令含 `place_world` |
| 雷达 | `/slamware_ros_sdk_server_node/scan` 必用 | 无 |
| 时限 | 仿真 **420 s**，取一件立刻送一件，5 单 | 600 s，双臂 hug |
| 在线模型 | **禁止** | DashScope 仅此处 |

`SUPERMARKET_USE_GS` 是官方场景的 **3D 高斯溅射外观**，不是 Genesis World。开源 [genesis-world](https://github.com/Genesis-Embodied-AI/genesis-world) 是仿真引擎，不能替换官方 Server，也进不了评测 Client。

## 9 个 `kind`（YOLO 类别必须逐字相同）

顺序 = 建议 class id，与 `runtime/task_protocol.py` 的 `KNOWN_KINDS` 一致：

`sanmingzhi`, `heweidao`, `shupian`, `zhijin`, `maidong`, `kouxiangtang`, `pingguo`, `chengzi`, `kele`

ArUco：`id//9` → 货架 A–E，`id%9//3` → 层 1–3，`id%3` → 列 1–3。例：**32 = D-L2-C3**。固定 Baseline `TASKS=product_032` 是可乐。

## 机器与运行环境

- 宿主机：Windows 11。Docker / GPU 在 **WSL Ubuntu-22.04**（路径 `/mnt/d/DG/...`）。
- 本机 GPU：RTX **4060** 笔记本。可租 **4090 ~20GB × 4 天** 做 YOLO 主训（有效训练是小时级）。
- Client 镜像：Ubuntu 22.04，ROS 2 Humble，PyTorch 2.7.1+cu128，**ultralytics 8.0.196**。
- `ROS_DOMAIN_ID=99`，`RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`。文旅与零售容器 **不要同时开**。
- 控制限幅（已实现）：底盘线速度 ±0.35、角速度 ±0.65。必须走 `runtime/ros_robot_control.py`。
- Shell 脚本：**LF**。不要在 `source /opt/ros/humble/setup.bash` **之前** `set -u`（会踩 `AMENT_TRACE_SETUP_FILES`）。source 之后可以 `set -u`，见现有 `scripts/run_preflight.sh`。

本机调试 Server 常加：

- `SUPERMARKET_USE_GS=0`（只省控制调试；**YOLO 训练必须 `1`，且 4060 要分批 ply，见 A 线文档）
- `SUPERMARKET_FIXED_BASELINE=1`，`RANDOMIZE=0`，`TASKS=product_032`
- 补丁挂载（`screeninfo` 无 `is_primary`，否则 Server 崩）：

```text
-v /mnt/d/DG/supermarket_sorting_baseline/patches/simulator.py:/workspace/supermarket_sorting_task/discoverse/envs/simulator.py:ro
```

Baseline 挂进 Client：`/mnt/d/DG/supermarket_sorting_baseline` → `/workspace/baseline`（开发用 `rw`）。完整 `docker run` 见该仓库 `README.md`。镜像 tar 在 `D:\DG\supermarket_sorting_{client,server}.tar`。

## 代码地图（只改零售仓库）

```text
supermarket_sorting_baseline/
  client_task_1.py          # 官方可乐演示。参考抓取，勿改成主入口
  runtime/
    ros_contract.py         # 话题名
    ros_robot_control.py    # 限幅发布 + stop_all
    ros_sensor_utils.py
    head_camera_kinematics.py
    motion_planning.py      # 包 kinematics.mmk2_kdl
    task_protocol.py        # 解析 task JSON，忽略未下发坐标
    orchestrator.py         # 取一件→送一件；尚未接运动
    scene_zones.py          # 420s、拣货区、送货台
    preflight.py / p2_preview.py / p3_preview.py
  perception/
    yolo_backend.py         # 现 CLASS_NAMES=["kele"]
    aruco_slots.py          # 0–44 → A-L1-C1 … E-L3-C3，框绑下方码
    kele_detect.py          # 官方可乐检测封装
  kinematics/               # MMK2 FK/IK
  patches/simulator.py      # Server 运行时补丁
  scripts/run_*.sh          # preflight / p2 / p3 / baseline
  weights/kele.pt           # 仅可乐
  test_*.py                 # 仓库根目录 unittest
```

文档目录与导读见 [AGENTS.md](./AGENTS.md)。本文是操作手册；分工见 `DG-202606-分工总览.md`。

## 已完成 / 未做

**已完成（仿真里跑过）：**

- P0 镜像可起；P1 预检：17 关节、odom、相机 TF、KDL、640×480 RGB-D、360 点雷达、任务 JSON。
- P2 编排干跑：1× `kele` → `DONE`。固定 Baseline `count:1` 正常。
- P3 ArUco+YOLO 绑定已实现。货架前、`GS=1` slim 可乐时能认出 `kele` 并绑槽。**不要**在 Server GS=1 时把 YOLO 放到 `cuda:0`（会挤死 Server）；`run_p3_preview.sh` 默认 CPU 且 `CUDA_VISIBLE_DEVICES=`。

**陷阱：** P3 预览若 `markers=[] products=[]`，先确认底盘是否对着货架（送货区 odom y≈−3.17）。`GS=0` 或 `SUPERMARKET_GS_NO_BACKGROUND=1` 时头相机没有货架/ArUco 贴图，空检测也不是「P3 代码坏了」。

**未做：** 9 类采图/训练（A 必须分批加载 3DGS，见 A 线文档）；雷达绕障；编排接真实抓放；随机商品+障碍的 5 单；评分主入口脚本。

送货：桌 `(-1.940, -3.410)`，目标 z `0.807`；`delivery_base` x `[-2.42,-1.46]` y `[-3.88,-2.62]`；`delivery_box` y `[-3.63,-3.19]` z `[0.74,1.05]`。详见 `runtime/scene_zones.py`。

## 两人分工（Agent 不要抢对方文件）

短期一人一块，**9 类权重进 `weights/` 之前不要两人都改 `yolo_backend.py` 的类别表**。

- **A 数据/YOLO：** 写采图脚本、仿真真值投影标注（仅训练）、YOLOv8s 微调、交出 `weights/supermarket9.pt` + `data.yaml`。采图必须 `GS=1` 且按 `SUPERMARKET_GS_KINDS` 分批（见 A 线文档）。不要改编排、限幅、`client_task_1.py`。
- **B 算法：** 面向货架、雷达、把 `orchestrator` 接到运动、参考官方可乐抓取迁到 `runtime/`。权重未到前用 `kele.pt` + `product_032`。不要改 A 的类别顺序和训练超参。

训练时用仿真 GT 投影打框 **可以**；评分 Client 把未下发坐标当真相 **不可以**。

## Agent 工作方式

- 实现 → 用户在仿真里验证。不要声称「已在浏览器验证」；这是 ROS/Docker 仿真。
- 改完跑：`cd supermarket_sorting_baseline && python -m unittest discover -v`（上次约 28 个测试）。
- 运动相关改动：请用户跑对应 `scripts/run_preflight.sh` / `run_p2_preview.sh` / `run_p3_preview.sh`。
- 不要提交除非用户明确要求。不要 `git commit --amend` 乱改。不要 `--force` push。
- 不要升级 Client 里的 ultralytics。不要把文旅 hug / HSV / Task1-2-3 拷进零售。
- 不要重写 `client_task_1.py`。新入口例如 `runtime/mission_client.py` + `scripts/run_mission.sh`。
- 用户规则：Web UI 改动才需要浏览器验证；本项目以仿真与 unittest 为准。

## 建议下一刀（按用户意图选线）

- **A：** `scripts/capture_yolo_frames.py`；**`GS=1` + `SUPERMARKET_GS_KINDS` 分批**对着货架采 RGB。不要用红几何体当 9 类训练集。
- **B：** 底盘从送货区开到货架，让 P3 非空；再雷达；再 1 次可乐取送。
- **联调：** A 的 `.pt` 就绪后改 `CLASS_NAMES` 为 9 类，换默认权重路径。

Docker 命令、镜像 tag、正式 vs 固定 Baseline：以 `supermarket_sorting_baseline/README.md` 为准，不要凭记忆发明话题名。
