# Supermarket Sorting Task

## 运行环境版本

- CUDA：12.8
- PyTorch：2.7.1+cu128
- ROS2：Humble

本仓库是独立 Baseline。Server 运行仿真，Client 提供固定运行环境，Baseline 源码通过
挂载进入 Client。固定 Baseline 用于验证一次抓取流程。正式任务中，选手需要在规定时间内
尽可能多地完成商品抓取和放置。

## 部署

宿主机需要 Docker、NVIDIA Driver（Linux >= 570.26）、NVIDIA Container Toolkit 和
NVIDIA GPU。下载离线镜像包：

- [Server 镜像 tar]链接: https://pan.baidu.com/s/1w6qeDOqi0hcLstdCetfGvQ 提取码: dhti 
- [Client 镜像 tar]链接: https://pan.baidu.com/s/1oSuz6v0cloXq1Mg74gF0tQ 提取码: d64j 

下载完成后，在 tar 文件所在目录加载镜像：

```bash
docker load -i supermarket_sorting_server.tar
docker load -i supermarket_sorting_client.tar

docker tag \
  crpi-1pzq998p9m7w0auy.cn-hangzhou.personal.cr.aliyuncs.com/challengecup/supermarket_sorting_final:server \
  supermarket_sorting:server
docker tag \
  crpi-1pzq998p9m7w0auy.cn-hangzhou.personal.cr.aliyuncs.com/challengecup/supermarket_sorting_final:client \
  supermarket_sorting:client
```

```bash
xhost +local:docker
docker volume create supermarket_sorting_cache
```

## 固定 Baseline

启动固定 Server：

```bash
docker run --rm -it \
  --gpus all \
  --network host \
  --ipc host \
  --name supermarket_sorting_server \
  -e DISPLAY=${DISPLAY} \
  -e ROS_DOMAIN_ID=99 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e MUJOCO_GL=glfw \
  -e SUPERMARKET_HEADLESS=0 \
  -e SUPERMARKET_ENABLE_RENDER=1 \
  -e SUPERMARKET_ENABLE_LIDAR=1 \
  -e SUPERMARKET_USE_GS=1 \
  -e TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions \
  -e SUPERMARKET_FIXED_BASELINE=1 \
  -e SUPERMARKET_RANDOMIZE=0 \
  -e SUPERMARKET_RANDOMIZE_OBSTACLES=0 \
  -e SUPERMARKET_TASKS=product_032 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v supermarket_sorting_cache:/root/.cache \
  supermarket_sorting:server \
  bash -lc "cd /workspace/supermarket_sorting_task && source /opt/ros/humble/setup.bash && python3 examples/supermarket_sorting/supermarket_sorting_server.py"
```

启动固定 Client，并挂载本仓库：

```bash
docker run --rm -dit \
  --gpus all \
  --network host \
  --ipc host \
  --name supermarket_sorting_client \
  -e ROS_DOMAIN_ID=99 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -v "$(pwd)":/workspace/baseline:ro \
  supermarket_sorting:client
```

在 Client 中启动 Baseline：

```bash
docker exec -it supermarket_sorting_client \
  bash -lc 'cd /workspace/baseline && ./scripts/run_baseline.sh'
```

## 正式运行

正式 Server 使用随机商品和随机障碍物：

```bash
docker run --rm -it \
  --gpus all \
  --network host \
  --ipc host \
  --name supermarket_sorting_server \
  -e DISPLAY=${DISPLAY} \
  -e ROS_DOMAIN_ID=99 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e MUJOCO_GL=glfw \
  -e SUPERMARKET_HEADLESS=0 \
  -e SUPERMARKET_ENABLE_RENDER=1 \
  -e SUPERMARKET_ENABLE_LIDAR=1 \
  -e SUPERMARKET_USE_GS=1 \
  -e TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions \
  -e SUPERMARKET_RANDOMIZE=1 \
  -e SUPERMARKET_RANDOMIZE_OBSTACLES=1 \
  -e SUPERMARKET_SEED=11 \
  -e SUPERMARKET_TASKS=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v supermarket_sorting_cache:/root/.cache \
  supermarket_sorting:server \
  bash -lc "cd /workspace/supermarket_sorting_task && source /opt/ros/humble/setup.bash && python3 examples/supermarket_sorting/supermarket_sorting_server.py"
```

正式 Client 挂载选手 Baseline，启动后保持运行：

```bash
docker run -dit \
  --gpus all \
  --network host \
  --ipc host \
  --name supermarket_sorting_client \
  -e ROS_DOMAIN_ID=99 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -v /path/to/your_baseline:/workspace/baseline:rw \
  supermarket_sorting:client
```

选手程序通过 `docker exec` 在 Client 容器内启动。

替换权重：

```text
weights/kele.pt
```

也可以设置 `SUPERMARKET_BASELINE_WEIGHTS=/workspace/baseline/weights/custom.pt`。

## 任务指令

Server 每次启动会随机放置 45 个商品和通道内 5 个障碍物。障碍物保持箱体竖直，只随机
改变平面偏航角，并通过路径检查保证货架入口至配送台入口存在通路。

Baseline 固定航点不处理随机障碍物。正式程序应读取二维雷达并自行规划：

```text
/slamware_ros_sdk_server_node/scan
```

任务消息示例：

```json
{"schema_version":1,"run_prefix":"run_a1b2c3d4e5f6","count":2,"targets":[{"id":"item_run_a1b2c3d4e5f6_01","kind":"kele"},{"id":"item_run_a1b2c3d4e5f6_02","kind":"kele"}]}
```

查看当前任务：

```bash
ros2 topic echo --once /supermarket_sorting/task
```

## ROS2 话题

`ROS_DOMAIN_ID` 必须在 Server 和 Client 之间保持一致。

### Server 发布

| Topic | Type | 说明 |
| --- | --- | --- |
| `/slamware_ros_sdk_server_node/odom` | `nav_msgs/msg/Odometry` | 底盘位姿和速度 |
| `/tf` | `tf2_msgs/msg/TFMessage` | 动态 TF |
| `/slamware_ros_sdk_server_node/scan` | `sensor_msgs/msg/LaserScan` | 二维激光雷达，默认 12 Hz |
| `/joint_states` | `sensor_msgs/msg/JointState` | 关节状态 |
| `/head_camera/color/image_raw` | `sensor_msgs/msg/Image` | 头部 RGB |
| `/head_camera/color/camera_info` | `sensor_msgs/msg/CameraInfo` | 头部 RGB 内参 |
| `/head_camera/aligned_depth_to_color/image_raw` | `sensor_msgs/msg/Image` | 头部深度，毫米 |
| `/head_camera/aligned_depth_to_color/camera_info` | `sensor_msgs/msg/CameraInfo` | 深度内参 |
| `/left_camera/color/image_raw` | `sensor_msgs/msg/Image` | 左腕 RGB |
| `/left_camera/color/camera_info` | `sensor_msgs/msg/CameraInfo` | 左腕内参 |
| `/right_camera/color/image_raw` | `sensor_msgs/msg/Image` | 右腕 RGB |
| `/right_camera/color/camera_info` | `sensor_msgs/msg/CameraInfo` | 右腕内参 |
| `/supermarket_sorting/task` | `std_msgs/msg/String` | JSON 任务清单 |

### Server 订阅

| Topic | Type | 控制格式 |
| --- | --- | --- |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | `linear.x`、`angular.z` |
| `/spine_forward_position_controller/commands` | `std_msgs/msg/Float64MultiArray` | 升降柱 |
| `/head_forward_position_controller/commands` | `std_msgs/msg/Float64MultiArray` | 头部关节 |
| `/left_arm_forward_position_controller/commands` | `std_msgs/msg/Float64MultiArray` | 左臂 6 轴和夹爪 |
| `/right_arm_forward_position_controller/commands` | `std_msgs/msg/Float64MultiArray` | 右臂 6 轴和夹爪 |

### Baseline 发布

| Topic | Type | 说明 |
| --- | --- | --- |
| `/kele/detections` | `vision_msgs/msg/Detection3DArray` | 可乐世界坐标检测结果 |
| `/kele/result_image` | `sensor_msgs/msg/Image` | 检测可视化图 |

`/joint_states` 顺序：

```text
slide_joint, head_yaw_joint, head_pitch_joint,
left_arm_joint1..left_arm_joint6, left_arm_eef_gripper_joint,
right_arm_joint1..right_arm_joint6, right_arm_eef_gripper_joint
```

## 参数说明

| 参数 | 推荐值 | 含义 |
| --- | --- | --- |
| `ROS_DOMAIN_ID` | `99` | Server 和 Client 通信域 |
| `RMW_IMPLEMENTATION` | `rmw_cyclonedds_cpp` | ROS2 RMW 实现 |
| `MUJOCO_GL` | `glfw` | 图形窗口；无头用 `egl` |
| `SUPERMARKET_HEADLESS` | `0` | 是否显示窗口 |
| `SUPERMARKET_ENABLE_RENDER` | `1` | 发布 RGB-D |
| `SUPERMARKET_ENABLE_LIDAR` | `1` | 发布雷达 |
| `SUPERMARKET_USE_GS` | `1` | 启用 3DGS |
| `SUPERMARKET_RANDOMIZE` | `1` | 随机商品位置 |
| `SUPERMARKET_SEED` | `11` | 商品随机种子 |
| `SUPERMARKET_RANDOMIZE_OBSTACLES` | `1` | 随机障碍物 |
| `SUPERMARKET_OBSTACLE_SEED` | 可选 | 障碍物随机种子 |
| `SUPERMARKET_TASKS` | `all` | 任务筛选 |
| `SUPERMARKET_BASELINE_WEIGHTS` | 可选 | 自定义权重路径 |

## 主要文件

```text
client_task_1.py             # 任务一控制程序
perception/kele_detect.py    # 可乐视觉检测
perception/yolo_backend.py   # YOLO 后端
kinematics/                  # MMK2 FK/IK
models/mmk2_head_fk.xml      # 相机 FK 模型
weights/kele.pt              # 默认权重
scripts/run_baseline.sh      # 启动脚本
```
