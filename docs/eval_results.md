# Evaluation results

This file is intentionally checked in as a template. Once you run
`make eval`, paste (or script) the numbers into the tables below so the
repo carries a reproducible record of the lift achieved.

## Setup

| Field | Value |
| --- | --- |
| Base model | Qwen/Qwen2.5-0.5B-Instruct |
| Adapter | outputs/qwen-0.5b-lora |
| LoRA rank / alpha / dropout | 8 / 16 / 0.05 |
| Train examples | _fill in: e.g. 3000 (GSM8K + NuminaMath-CoT, length <=1500)_ |
| Val examples | _fill in_ |
| Hardware | _fill in: e.g. 1x RTX 4090, 24 GB_ |
| Wall-clock | _fill in_ |

## GSM8K test (limit 200)

| Metric | Baseline | Fine-tuned | Delta |
| --- | --- | --- | --- |
| Exact-match accuracy | _fill in_ | _fill in_ | _fill in_ |
| Wall-clock (eval) | _fill in_ | _fill in_ | - |

## Curated prompts (10 items)

| Category | Baseline | Fine-tuned |
| --- | --- | --- |
| word-problem (5) | _fill in_ | _fill in_ |
| calculus (3) | _fill in_ | _fill in_ |
| algebra (2) | _fill in_ | _fill in_ |

## Notable failures

After running `make eval`, paste 2-3 example failures here with a one-line
analysis. Failure-mode commentary is the single thing that signals
"actually looked at the outputs" to a reader.

```
example_id: gsm8k-test-0123
expected: 36
extracted: 360
note: model multiplied by 10 instead of dividing; common arithmetic-shift error.
```
