#!/usr/bin/env bash
# Pull a balanced math reasoning training set into data/.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m math_lora.build_dataset \
    --train-size 3000 \
    --val-size 300 \
    "$@"
