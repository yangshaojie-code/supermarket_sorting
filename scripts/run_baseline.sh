#!/usr/bin/env bash
set -eo pipefail

baseline_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/humble/setup.bash
set -u
cd "$baseline_root"

python3 -m perception.kele_detect \
  --weights "${SUPERMARKET_BASELINE_WEIGHTS:-$baseline_root/weights/kele.pt}" \
  --device "${SUPERMARKET_DETECTOR_DEVICE:-auto}" &
detector_pid=$!

cleanup() {
  kill "$detector_pid" 2>/dev/null || true
  wait "$detector_pid" 2>/dev/null || true
}
trap cleanup EXIT

python3 client_task_1.py
