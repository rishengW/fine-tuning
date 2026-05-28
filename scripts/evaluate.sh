#!/usr/bin/env bash
# Run baseline + finetuned eval on GSM8K (200 samples) and print the lift.
#
# Usage:
#   ./scripts/evaluate.sh                     # uses the 0.5B config defaults
#   ADAPTER=outputs/custom ./scripts/evaluate.sh
set -euo pipefail
cd "$(dirname "$0")/.."

BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
ADAPTER=${ADAPTER:-outputs/qwen-0.5b-lora}
LIMIT=${LIMIT:-200}

mkdir -p reports

python -m math_lora.evaluate \
    --base-model "$BASE_MODEL" \
    --suite gsm8k --limit "$LIMIT" \
    --report-out reports/baseline.json

python -m math_lora.evaluate \
    --base-model "$BASE_MODEL" \
    --adapter "$ADAPTER" \
    --suite gsm8k --limit "$LIMIT" \
    --report-out reports/finetuned.json

python -m math_lora.report_diff reports/baseline.json reports/finetuned.json
