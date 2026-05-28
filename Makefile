.PHONY: install dataset train eval eval-baseline eval-finetuned eval-diff lint typecheck test fmt clean

PYTHON ?= python
CONFIG ?= configs/qwen-0.5b-lora.yaml
ADAPTER ?= outputs/qwen-0.5b-lora
BASE_MODEL ?= Qwen/Qwen2.5-0.5B-Instruct
LIMIT ?= 200

install:
	$(PYTHON) -m pip install -e ".[dev]"

dataset:
	$(PYTHON) -m math_lora.build_dataset --train-size 3000 --val-size 300

train:
	$(PYTHON) -m math_lora.train --config $(CONFIG)

eval-baseline:
	mkdir -p reports
	$(PYTHON) -m math_lora.evaluate \
		--base-model $(BASE_MODEL) \
		--suite gsm8k --limit $(LIMIT) \
		--report-out reports/baseline.json

eval-finetuned:
	mkdir -p reports
	$(PYTHON) -m math_lora.evaluate \
		--base-model $(BASE_MODEL) \
		--adapter $(ADAPTER) \
		--suite gsm8k --limit $(LIMIT) \
		--report-out reports/finetuned.json

eval-diff:
	$(PYTHON) -m math_lora.report_diff reports/baseline.json reports/finetuned.json

eval: eval-baseline eval-finetuned eval-diff

lint:
	$(PYTHON) -m ruff check src

typecheck:
	$(PYTHON) -m mypy src

test:
	$(PYTHON) -m pytest

fmt:
	$(PYTHON) -m ruff format src

clean:
	rm -rf outputs reports .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
