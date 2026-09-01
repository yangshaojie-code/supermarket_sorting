#!/usr/bin/env bash
set -eo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/humble/setup.bash
set -u
DEVICE="${SUPERMARKET_DETECTOR_DEVICE:-cpu}"
if [ "$DEVICE" = "cpu" ]; then
  # Hide the GPU from this process. import torch still creates a CUDA
  # context on --gpus all Client containers and OOM-kills Server GS=1.
  export CUDA_VISIBLE_DEVICES=""
fi
echo "[p3] device=${DEVICE} CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES-}'"
python3 -m runtime.p3_preview --device "${DEVICE}" "$@"
