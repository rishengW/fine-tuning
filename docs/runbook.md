# Runbook

End-to-end recipe from a fresh clone to a benchmarked adapter.

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

GPU users on Linux/Windows: install a CUDA-matching wheel of `bitsandbytes`
if you plan to run the QLoRA config.

## 2. Build the dataset

```bash
make dataset                       # 3000 train + 300 val examples
```

This downloads GSM8K and streams a slice of NuminaMath-CoT, filters by
length, and writes `data/train.jsonl` + `data/val.jsonl`. The repo's seed
files are overwritten.

## 3. Train

Default 0.5B run on a single consumer GPU:

```bash
make train
```

Adapter and resolved config land under `outputs/qwen-0.5b-lora/`.

For 7B QLoRA (requires a 24GB GPU and `bitsandbytes`):

```bash
make train CONFIG=configs/qwen-7b-qlora.yaml \
           ADAPTER=outputs/qwen-7b-qlora
```

## 4. Evaluate

Run baseline (no adapter) and fine-tuned (with adapter) on the GSM8K test
split, then print the per-category lift:

```bash
make eval                          # baseline + finetuned + diff
```

Reports land in `reports/baseline.json` and `reports/finetuned.json`. Each
report contains the full prompt/response/extracted-answer/correct quadruple
per example, so failure-case analysis is just `jq` away:

```bash
jq '.records[] | select(.correct == false) | {id, expected_answer, extracted_answer}' \
    reports/finetuned.json | head -50
```

## 5. Serve (optional)

Start a single-process inference endpoint:

```bash
python -m math_lora.serve \
    --base-model Qwen/Qwen2.5-0.5B-Instruct \
    --adapter outputs/qwen-0.5b-lora \
    --port 8000
```

Hit it:

```bash
curl -s http://127.0.0.1:8000/generate \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"What is 7 * 8?"}]}'
```

## Common knobs

| Symptom | First lever to try |
| --- | --- |
| Validation loss diverging while train loss falls | `--override lora.r=4 --override lora.dropout=0.1` |
| OOM on a 24 GB GPU | switch to `configs/qwen-7b-qlora.yaml`; lower `data.max_seq_len` |
| Eval accuracy not moving | inspect `reports/finetuned.json`, check the model is actually emitting `Final answer:` markers |
| Need a quick smoke run | `make train CONFIG=configs/qwen-0.5b-lora.yaml` then `--override training.num_epochs=0.1` |
