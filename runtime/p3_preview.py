#!/usr/bin/env python3
"""P3 preview: detect shelf ArUco and bind visible kele bottles to slots.

Local YOLO + OpenCV only. Does not call DashScope or any online VLM.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from perception.aruco_slots import (
    KindSlotMap,
    ProductDetection,
    associate_products_to_slots,
    detect_aruco_markers,
    slot_from_marker_id,
)
from runtime.task_protocol import parse_task_payload


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "weights" / "kele.pt"


def _wait_rgb(timeout_sec: float = 12.0):
    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image
    from std_msgs.msg import String

    from runtime.ros_contract import RGB_TOPIC, TASK_TOPIC
    from runtime.ros_sensor_utils import decode_image

    rclpy.init()
    node = rclpy.create_node("supermarket_p3_preview")
    box = {"rgb": None, "task": None}
    latch = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )

    def rgb_cb(message):
        box["rgb"] = decode_image(message)

    def task_cb(message):
        box["task"] = message.data

    node.create_subscription(Image, RGB_TOPIC, rgb_cb, 10)
    node.create_subscription(String, TASK_TOPIC, task_cb, latch)
    deadline = time.monotonic() + timeout_sec
    try:
        while box["rgb"] is None:
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for head RGB")
            rclpy.spin_once(node, timeout_sec=0.1)
        for _ in range(10):
            rclpy.spin_once(node, timeout_sec=0.05)
        return box["rgb"], box["task"]
    finally:
        node.destroy_node()
        rclpy.shutdown()


def _detect_kele(bgr, weights: Path, confidence: float) -> list:
    if not weights.is_file():
        return []
    from perception.yolo_backend import YoloBackend

    detector = YoloBackend(weights, confidence=confidence, device="auto")
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


def preview_from_rgb(rgb, task_raw=None, weights=DEFAULT_WEIGHTS, confidence=0.65) -> dict:
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    markers = detect_aruco_markers(bgr)
    products = _detect_kele(bgr, Path(weights), confidence)
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
            {"id": item.marker_id, "slot": item.slot.name, "pixel": [item.u, item.v]}
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Preview ArUco slots and kele binding")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--confidence", type=float, default=0.65)
    args = parser.parse_args(argv)
    rgb, task_raw = _wait_rgb()
    print(json.dumps(preview_from_rgb(rgb, task_raw, args.weights, args.confidence), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
