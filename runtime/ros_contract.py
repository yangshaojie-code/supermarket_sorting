"""ROS 2 topic names for the supermarket sorting Server/Client pair."""

TASK_TOPIC = "/supermarket_sorting/task"
RGB_TOPIC = "/head_camera/color/image_raw"
DEPTH_TOPIC = "/head_camera/aligned_depth_to_color/image_raw"
RGB_CAMERA_INFO_TOPIC = "/head_camera/color/camera_info"
DEPTH_CAMERA_INFO_TOPIC = "/head_camera/aligned_depth_to_color/camera_info"
LEFT_WRIST_RGB_TOPIC = "/left_camera/color/image_raw"
RIGHT_WRIST_RGB_TOPIC = "/right_camera/color/image_raw"
JOINT_STATES_TOPIC = "/joint_states"
ODOM_TOPIC = "/slamware_ros_sdk_server_node/odom"
SCAN_TOPIC = "/slamware_ros_sdk_server_node/scan"
TF_TOPIC = "/tf"
TF_STATIC_TOPIC = "/tf_static"

CMD_VEL_TOPIC = "/cmd_vel"
SPINE_COMMAND_TOPIC = "/spine_forward_position_controller/commands"
HEAD_COMMAND_TOPIC = "/head_forward_position_controller/commands"
LEFT_ARM_COMMAND_TOPIC = "/left_arm_forward_position_controller/commands"
RIGHT_ARM_COMMAND_TOPIC = "/right_arm_forward_position_controller/commands"

ROS_DOMAIN_ID = "99"
RMW_IMPLEMENTATION = "rmw_cyclonedds_cpp"
HEAD_CAMERA_FRAME = "head_camera"

REQUIRED_JOINT_NAMES = (
    "slide_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    *(f"left_arm_joint{i}" for i in range(1, 7)),
    "left_arm_eef_gripper_joint",
    *(f"right_arm_joint{i}" for i in range(1, 7)),
    "right_arm_eef_gripper_joint",
)


def topic_contract() -> dict:
    """Return a serializable contract useful for startup diagnostics."""
    return {
        "task": TASK_TOPIC,
        "rgb": RGB_TOPIC,
        "depth": DEPTH_TOPIC,
        "camera_info": [RGB_CAMERA_INFO_TOPIC, DEPTH_CAMERA_INFO_TOPIC],
        "wrist_rgb": [LEFT_WRIST_RGB_TOPIC, RIGHT_WRIST_RGB_TOPIC],
        "joint_states": JOINT_STATES_TOPIC,
        "odom": ODOM_TOPIC,
        "scan": SCAN_TOPIC,
        "tf": TF_TOPIC,
        "tf_static": TF_STATIC_TOPIC,
        "control": {
            "cmd_vel": CMD_VEL_TOPIC,
            "spine": SPINE_COMMAND_TOPIC,
            "head": HEAD_COMMAND_TOPIC,
            "left_arm": LEFT_ARM_COMMAND_TOPIC,
            "right_arm": RIGHT_ARM_COMMAND_TOPIC,
        },
        "required_joints": list(REQUIRED_JOINT_NAMES),
    }
