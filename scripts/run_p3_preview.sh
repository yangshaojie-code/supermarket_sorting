#!/usr/bin/env bash
set -eo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/humble/setup.bash
set -u
python3 -m runtime.p3_preview "$@"
