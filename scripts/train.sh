#!/usr/bin/env bash
# Train using the default 0.5B config. Pass extra args through, e.g.
#   ./scripts/train.sh --override training.num_epochs=1
set -euo pipefail
cd "$(dirname "$0")/.."
python -m math_lora.train \
    --config configs/qwen-0.5b-lora.yaml \
    "$@"
