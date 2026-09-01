#!/usr/bin/env python3
"""P3 preview: detect shelf ArUco and bind visible kele bottles to slots.

Local YOLO + OpenCV only. Does not call DashScope or any online VLM.
Aims the head at aisle-scan poses before capturing, because the drive script
stops on the yellow line (~1 m from the shelf face).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from perception.aruco_slots import (
    KindSlotMap,
    ProductDetection,
    associate_products_to_slots,
    detect_aruco_markers,
    detect_marker_quads,
    infer_markers_from_quads,
    dedupe_markers_by_pixel,
    slot_from_marker_id,
)
from runtime.head_camera_kinematics import AISLE_SCAN_POSES, AISLE_SCAN_SPINE
from runtime.task_protocol import parse_task_payload


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "weights" / "kele.pt"
DEFAULT_SAVE = ROOT / "outputs" / "p3_preview.png"
HEAD_SETTLE_S = 0.9


def _products_from_detector(bgr, detector) -> list:
    if detector is None:
        return []
    return [
        ProductDetection(
            kind=item["class"],
            u=float(item["x"]),
            v=float(item["y"]),
            conf=float(item["conf"]),
            w=float(item["w"]),
            h=float(item["h"]),
        )
        for item in detector.detect(bgr)
        if item["class"] in {"kele", "maidong"}
    ]


def _detect_kele(bgr, weights: Path, confidence: float, device: str = "cpu") -> list:
    if not weights.is_file():
        return []
    from perception.yolo_backend import YoloBackend

    return _products_from_detector(
        bgr, YoloBackend(weights, confidence=confidence, device=device)
    )


def preview_from_rgb(
    rgb,
    task_raw=None,
    weights=DEFAULT_WEIGHTS,
    confidence=0.65,
    products=None,
    device: str = "cpu",
) -> dict:
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    markers = dedupe_markers_by_pixel(detect_aruco_markers(bgr))
    quads = detect_marker_quads(bgr)
    markers = list(markers) + infer_markers_from_quads(markers, quads)
    if products is None:
        products = _detect_kele(bgr, Path(weights), confidence, device=device)
    assignments = associate_products_to_slots(products, markers)
    mapping = KindSlotMap()
    for item in assignments:
        if item.get("marker_id") is None:
            continue
        mapping.observe(item["kind"], slot_from_marker_id(item["marker_id"]), item["conf"])

    task = None
    if task_raw:
        parsed = parse_task_payload(task_raw)
        task = {
            "run_prefix": parsed.run_prefix,
            "wanted_kinds": list(parsed.kinds()),
            "matched_wanted": {
                kind: mapping.lookup(kind)
                for kind in parsed.kinds()
            },
        }
    return {
        "aruco_dictionary": "DICT_4X4_50",
        "markers": [
            {
                "id": item.marker_id,
                "slot": item.slot.name,
                "pixel": [item.u, item.v],
                **({"inferred": True} if item.inferred else {}),
            }
            for item in markers
        ],
        "products": [
            {"kind": item.kind, "conf": item.conf, "pixel": [item.u, item.v]}
            for item in products
        ],
        "assignments": assignments,
        "kind_slot_map": mapping.as_dict(),
        "task": task,
        "note": "empty markers usually means the head camera is not facing a shelf",
    }


def red_primitive_blobs(bgr, min_area: int = 80) -> list:
    """GS=0 products are untextured red solids. Not used as scoring truth."""
    import cv2

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 70, 70), (12, 255, 255)),
        cv2.inRange(hsv, (168, 70, 70), (180, 255, 255)),
    )
    mask = cv2.medianBlur(mask, 5)
    _n, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    blobs = []
    for index in range(1, len(stats)):
        _x, _y, width, height, area = stats[index]
        if area < min_area:
            continue
        blobs.append({
            "pixel": [round(float(centroids[index][0]), 1), round(float(centroids[index][1]), 1)],
            "w": int(width),
            "h": int(height),
            "area": int(area),
        })
    blobs.sort(key=lambda item: item["area"], reverse=True)
    return blobs[:12]


def _annotate_preview(bgr, report: dict):
    import cv2

    vis = np.ascontiguousarray(bgr.copy())
    for quad in report.get("marker_quads") or []:
        pts = np.asarray(quad, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], True, (0, 255, 255), 2)
    for marker in report.get("markers") or []:
        u, v = int(marker["pixel"][0]), int(marker["pixel"][1])
        cv2.circle(vis, (u, v), 6, (0, 255, 0), -1)
        cv2.putText(vis, str(marker["id"]), (u + 6, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    for blob in report.get("red_blobs") or []:
        u, v = int(blob["pixel"][0]), int(blob["pixel"][1])
        cv2.rectangle(
            vis,
            (u - blob["w"] // 2, v - blob["h"] // 2),
            (u + blob["w"] // 2, v + blob["h"] // 2),
            (0, 0, 255),
            2,
        )
    for product in report.get("products") or []:
        u, v = int(product["pixel"][0]), int(product["pixel"][1])
        cv2.circle(vis, (u, v), 8, (255, 0, 0), 2)
        cv2.putText(vis, product["kind"], (u + 8, v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return vis


def _empty_marker_note(report: dict) -> str:
    mean = float(report.get("rgb_mean") or 0.0)
    if mean < 8.0 and not report.get("marker_quads"):
        return (
            "head RGB is almost black: shelf/ArUco live in background GS. "
            "Restart Server without SUPERMARKET_GS_NO_BACKGROUND=1. "
            "Keep YOLO on CPU so the 4060 is not OOM-killed."
        )
    if report.get("marker_quads"):
        return (
            "black squares are visible but ArUco IDs did not decode. "
            "If products still look like red primitives, GS is not on this camera "
            "or that SKU was not in SUPERMARKET_GS_KINDS."
        )
    return "empty markers usually means the head camera is not facing a shelf"


def _n_bound(report: dict) -> int:
    return sum(
        1
        for item in report.get("assignments") or []
        if item.get("marker_id") is not None
    )


def _score(report: dict) -> tuple:
    return (
        _n_bound(report),
        len(report.get("products") or []),
        len(report.get("markers") or []),
        len(report.get("marker_quads") or []),
        len(report.get("red_blobs") or []),
    )


def _save_bgr(path: Path, bgr) -> str:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), bgr)
    return str(path)


def capture_preview(
    weights: Path,
    confidence: float,
    save_path: Path,
    aim: bool,
    device: str = "cpu",
) -> dict:
    import rclpy
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import Image, JointState
    from std_msgs.msg import Float64MultiArray, String
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

    from runtime.ros_contract import JOINT_STATES_TOPIC, RGB_TOPIC, TASK_TOPIC
    from runtime.ros_robot_control import RosRobotController
    from runtime.ros_sensor_utils import SensorCache

    rclpy.init()
    node = rclpy.create_node("supermarket_p3_preview")
    sensors = SensorCache()
    controller = RosRobotController(node, Twist, Float64MultiArray, sensors)
    latch = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )

    def rgb_cb(message):
        try:
            sensors.update_rgb(message)
        except Exception:
            pass

    def joint_cb(message):
        try:
            sensors.update_joint_state(message)
        except Exception:
            pass

    def task_cb(message):
        sensors.update_task(message)

    node.create_subscription(Image, RGB_TOPIC, rgb_cb, 10)
    node.create_subscription(JointState, JOINT_STATES_TOPIC, joint_cb, 10)
    node.create_subscription(String, TASK_TOPIC, task_cb, latch)

    poses = list(AISLE_SCAN_POSES) if aim else [(None, None)]
    best = None
    tried = []
    try:
        deadline = time.monotonic() + 12.0
        while sensors.rgb is None:
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for head RGB")
            rclpy.spin_once(node, timeout_sec=0.1)

        detector = None
        if Path(weights).is_file():
            from perception.yolo_backend import YoloBackend

            print(f"[p3] loading YOLO on {device}", flush=True)
            detector = YoloBackend(weights, confidence=confidence, device=device)

        for yaw, pitch in poses:
            if yaw is not None:
                controller.command_spine(AISLE_SCAN_SPINE)
                controller.command_head((yaw, pitch))
                settle_until = time.monotonic() + HEAD_SETTLE_S
                while time.monotonic() < settle_until:
                    controller.command_spine(AISLE_SCAN_SPINE)
                    controller.command_head((yaw, pitch))
                    rclpy.spin_once(node, timeout_sec=0.05)
            else:
                for _ in range(10):
                    rclpy.spin_once(node, timeout_sec=0.05)
            if sensors.rgb is None:
                continue
            rgb = np.ascontiguousarray(sensors.rgb)
            bgr = np.ascontiguousarray(rgb[..., ::-1])
            report = preview_from_rgb(
                rgb,
                sensors.task_raw,
                weights,
                confidence,
                products=_products_from_detector(bgr, detector),
            )
            report["scan_pose"] = None if yaw is None else {"yaw": yaw, "pitch": pitch}
            report["rgb_mean"] = float(np.mean(rgb))
            report["marker_quads"] = detect_marker_quads(bgr)
            report["red_blobs"] = red_primitive_blobs(bgr)
            if not report["markers"]:
                report["note"] = _empty_marker_note(report)
            elif _n_bound(report) == 0 and report.get("products"):
                report["note"] = (
                    "product is visible but no same-column marker sits below it. "
                    "The L2 lip square may still be undecoded."
                )
            elif _n_bound(report) > 0:
                report["note"] = "bound product to the slot marker below it"
            tried.append({
                "pose": report["scan_pose"],
                "n_markers": len(report["markers"]),
                "n_products": len(report["products"]),
                "n_bound": _n_bound(report),
                "n_quads": len(report["marker_quads"]),
                "n_red_blobs": len(report["red_blobs"]),
            })
            if best is None or _score(report) > _score(best):
                best = report
                best["_bgr"] = bgr
            if _n_bound(report) > 0:
                break
        if best is None:
            raise TimeoutError("no RGB frame after aiming the head")
        bgr = best.pop("_bgr")
        best["tried_poses"] = tried
        best["saved_rgb"] = _save_bgr(save_path, _annotate_preview(bgr, best))
        values = dict(zip(sensors.joint_names, [float(v) for v in sensors.joint_positions]))
        best["head_joints"] = {
            "slide": values.get("slide_joint"),
            "head_yaw": values.get("head_yaw_joint"),
            "head_pitch": values.get("head_pitch_joint"),
        }
        return best
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Preview ArUco slots and kele binding")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--confidence", type=float, default=0.65)
    parser.add_argument("--save", type=Path, default=DEFAULT_SAVE)
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["auto", "cpu", "cuda"],
        help="YOLO device. cpu is required on an 8GB 4060 while Server GS=1 is running",
    )
    parser.add_argument(
        "--no-aim",
        action="store_true",
        help="capture the current head pose only; do not command spine/head",
    )
    args = parser.parse_args(argv)
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("[p3] CUDA_VISIBLE_DEVICES= (hidden; Server GS keeps the 4060)", flush=True)
    report = capture_preview(
        args.weights,
        args.confidence,
        args.save,
        aim=not args.no_aim,
        device=args.device,
    )
    print(json.dumps(report, indent=2))
    return 0 if report.get("markers") else 1


if __name__ == "__main__":
    raise SystemExit(main())
