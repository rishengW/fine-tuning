# Implementation Plan: Math LoRA Fine-Tuning Pipeline

## Overview

This plan converts the design into an incremental, dependency-aware sequence of Python coding tasks that build the eight components of the math-LoRA fine-tuning pipeline (`Model_Selector`, `Hardware_Budget_Planner`, `Dataset_Builder`, `LoRA_Trainer`, `Training_Pipeline`, `Evaluator`, `Inference_Server`, `Experiment_Tracker`).

Implementation language: **Python**, using `transformers`, `peft`, `bitsandbytes`, `accelerate`, `datasets`, `sympy` for evaluation, `hypothesis` for property-based tests, and `pytest` as the test runner.

Each task references the specific requirement clauses it covers and (where applicable) the design property numbers it implements or tests. Property tests are written as sub-tasks placed close to their corresponding implementation so failures are caught early. There is exactly one property-based test per correctness property (43 in total). Test sub-tasks are marked optional with `*` and may be skipped for a faster MVP, but core implementation sub-tasks must be implemented.

Convert the feature design into a series of prompts for a code-generation LLM that will implement each step with incremental progress. Make sure that each prompt builds on the previous prompts, and ends with wiring things together. There should be no hanging or orphaned code that isn't integrated into a previous step. Focus ONLY on tasks that involve writing, modifying, or testing code.

## Tasks

- [x] 1. Set up project structure, dependencies, and shared domain types
  - [x] 1.1 Create Python project layout and dependency manifest
    - Create `pyproject.toml` (or `requirements.txt`) pinning `transformers`, `peft`, `bitsandbytes`, `accelerate`, `datasets`, `tokenizers`, `torch`, `sympy`, `pydantic`, `hypothesis`, `pytest`
    - Create source layout under `src/math_lora/` with one package per component (`model_selector/`, `planner/`, `dataset_builder/`, `lora_trainer/`, `pipeline/`, `evaluator/`, `inference_server/`, `tracker/`) and a shared `types/` package
    - Create `tests/` with `tests/unit/`, `tests/property/`, `tests/integration/` subdirectories and a top-level `conftest.py`
    - Configure `pytest` and `hypothesis` (settings profile with `max_examples >= 100` per property test)
    - _Requirements: foundational (supports all)_

  - [x] 1.2 Define shared domain types and schemas
    - Implement `BaseModelCandidate`, `HardwareProfile`, `BudgetProfile`, `QuantizationMode`, `LoRAConfig`, `DecodingParams`, `ReasoningRecord` as `pydantic` models with field-level validation matching the design's data-model section
    - Implement enums/literals for `Quantization_Mode` ∈ `{fp16, bf16, int8, nf4}` and `bias` ∈ `{none, all, lora_only}`
    - Implement schema validation errors that name the offending field (used later by `LoRAConfigInvalid`, `ConfigLoadError`, `DecodingParamsInvalid`)
    - _Requirements: 1.1, 1.2, 2.1, 2.2, 4.1, 7.7_

  - [x] 1.3 Write unit tests for domain-type schema validation
    - Example-based tests covering each field's accepted/rejected values for `BaseModelCandidate`, `HardwareProfile`, `BudgetProfile`, `LoRAConfig`, `DecodingParams`
    - _Requirements: 1.1, 1.2, 2.1, 2.2, 4.1, 7.7_

  - [x] 1.4 Implement Hypothesis generators for domain types
    - Strategies for `BaseModelCandidate` (valid + one-field-mutated-invalid variants), `HardwareProfile` (`vram_per_gpu_gb in [1, 1024]` including thresholds 1, 8, 16, 24, 80), `LoRAConfig`, `DecodingParams`, `ReasoningRecord` (with controlled probability of LaTeX/unicode math)
    - Strategies for simulated training-run event traces used by state-machine properties
    - Place generators in `tests/property/strategies.py` for reuse across property tests
    - _Requirements: foundational for all property tests_

- [ ] 2. Implement `Model_Selector`
  - [-] 2.1 Implement VRAM estimation function with documented coefficients
    - Function `estimate_min_vram_gb(candidate, hw_profile, quantization_mode, sequence_length)` using the formula in the design (base weights + LoRA gradients + optimizer state on trainable params + activations + overhead)
    - Expose coefficients (`bytes_per_param` per `Quantization_Mode`, optimizer multiplier, activation coefficient, overhead) as a documented dataclass returned alongside the estimate
    - Determinism guarantee: pure function, no global state
    - _Requirements: 1.3_

  - [~] 2.2 Write property test for VRAM estimation
    - **Property 1: VRAM estimation is deterministic and monotonic**
    - **Validates: Requirements 1.3**
    - Verify determinism, monotonicity in `param_count_b` and `sequence_length`, and monotone non-increase across `fp16 -> bf16 -> int8 -> nf4`; verify exposed coefficients reconstruct the estimate exactly
    - _Requirements: 1.3_

  - [~] 2.3 Implement candidate eligibility, feasibility, and scoring
    - Eligibility check: mark candidate ineligible and record `missing_fields` if any field from Req 1.1 / 1.8 is absent
    - Feasibility flag and `vram_shortfall_gb` computation rounded to one decimal place
    - Scoring function `score = w_gsm * normalize(gsm8k) + w_math * normalize(math) + w_params * normalize(params_b) + w_license * license_permissiveness` with weights summing to 1.0; lexicographic tie-breaker on `model_id`
    - _Requirements: 1.1, 1.4, 1.5, 1.8, 1.10_

  - [~] 2.4 Implement `select(...)` orchestration and `SelectionReport` emission
    - Build the full `SelectionReport` (per-candidate entries, weights, tie-breaker rule, VRAM coefficients, chosen `model_id` or `null`)
    - Empty-candidate-list error path (`EmptyCandidateList`, no selection performed)
    - _Requirements: 1.6, 1.9_

  - [~] 2.5 Write property test for selection report soundness
    - **Property 2: Selection report soundness**
    - **Validates: Requirements 1.4, 1.5, 1.6, 1.8, 1.10**
    - _Requirements: 1.4, 1.5, 1.6, 1.8, 1.10_

  - [~] 2.6 Implement fallback suggestion for the all-infeasible case
    - For the highest-scoring candidate compute the smallest VRAM increase that makes it feasible at the current mode and the alternative `Quantization_Mode` from `{fp16, bf16, int8, nf4}` that makes it feasible without VRAM increase, or state none exists
    - _Requirements: 1.7_

  - [~] 2.7 Write property test for fallback suggestion correctness
    - **Property 3: Fallback suggestion is corrective**
    - **Validates: Requirements 1.7**
    - _Requirements: 1.7_

  - [~] 2.8 Write unit test for empty-candidate-list error
    - Example test asserting `EmptyCandidateList` is surfaced when input list is empty
    - _Requirements: 1.9_

- [ ] 3. Implement `Hardware_Budget_Planner`
  - [-] 3.1 Implement `Hardware_Profile` and `Budget_Profile` config loading with field-level errors
    - Load YAML/JSON config files for both profiles
    - Surface `ConfigLoadError` identifying file and missing/invalid field if loading fails or any required field is absent
    - _Requirements: 2.1, 2.2, 2.3_

  - [-] 3.2 Implement default `Quantization_Mode` resolution
    - When `vram_per_gpu_gb < 24` and `quantization_mode` is unset, resolve to `nf4`; respect explicit override otherwise
    - _Requirements: 2.4_

  - [~] 3.3 Write property test for default quantization mode
    - **Property 5: Default Quantization_Mode for low VRAM**
    - **Validates: Requirements 2.4**
    - _Requirements: 2.4_

  - [~] 3.4 Implement projected wall-clock, cost, and peak-VRAM estimation
    - Pure functions for projected hours, projected cost = `gpu_hours * cost_rate_per_gpu_hour`, and projected peak VRAM per GPU using the same coefficients exposed by `Model_Selector`
    - Inputs: `Base_Model`, `Quantization_Mode`, `batch_size`, `sequence_length`, `gradient_accumulation_steps`, `gradient_checkpointing`
    - _Requirements: 2.5_

  - [~] 3.5 Implement `plan(...)` and the halt-with-suggestion knob search
    - Produce `PreFlightReport`; if peak-VRAM, time, or cost projection exceeds its limit, halt and emit `suggested_knob_change` (single-knob smallest reduction over `batch_size`, `sequence_length`, `gradient_accumulation_steps`, `dataset_size`, `max_steps` that brings the projection within the limit)
    - Halt categories: `VRAMExceeded`, `TimeBudgetExceeded`, `CostBudgetExceeded`
    - _Requirements: 2.5, 2.6, 2.7, 2.8_

  - [~] 3.6 Write property test for halt suggestion correctness
    - **Property 4: Pre-flight halt suggestion is corrective**
    - **Validates: Requirements 2.6, 2.7, 2.8**
    - For any over-limit configuration, applying the suggested knob change and rerunning `plan(...)` brings the projection at or below its limit
    - _Requirements: 2.6, 2.7, 2.8_

  - [~] 3.7 Implement VRAM-sampling cadence helper used by `Training_Pipeline`
    - Helper that, given a step counter and a sampling policy, produces sample events at least once per 100 steps within any window of 100 consecutive steps
    - Records `(step, peak_vram_gb_per_gpu)` for the manifest
    - _Requirements: 2.10_

  - [~] 3.8 Write property test for periodic VRAM sampling coverage
    - **Property 6: Periodic VRAM sampling coverage**
    - **Validates: Requirements 2.10**
    - _Requirements: 2.10_

  - [~] 3.9 Implement cost reconciliation
    - `reconcile_cost(pre_flight, actual_gpu_hours)` returning `actual_cost`, `absolute_diff`, `pct_diff`
    - Emitted at end-of-run when `Hardware_Profile.deployment == "cloud"`
    - _Requirements: 2.12_

  - [~] 3.10 Write property test for cost reconciliation arithmetic
    - **Property 7: Cost reconciliation arithmetic**
    - **Validates: Requirements 2.12**
    - _Requirements: 2.12_

  - [~] 3.11 Write unit/integration tests for feature-toggle availability and consumer-GPU configuration
    - Smoke tests that gradient checkpointing, gradient accumulation, mixed precision (`bf16`/`fp16`), and `nf4` quantization are exposed configuration toggles consumed by the planner
    - One configuration test for the 8–16 GB consumer-GPU path that runs the planner with the smallest feasible base model and confirms no halt
    - _Requirements: 2.9, 2.11_

- [ ] 4. Implement `Dataset_Builder`
  - [-] 4.1 Implement source connectors (smoke-level)
    - Connectors for GSM8K training split, MATH training split, a documented open-source step-by-step corpus, and operator-supplied integral/derivation pairs
    - Each connector returns an iterable of raw records and a source identifier
    - _Requirements: 3.1_

  - [~] 4.2 Write smoke tests for each source connector
    - One example test per source verifying interface availability and a small fixture round-trip
    - _Requirements: 3.1_

  - [~] 4.3 Implement `Reasoning_Format` normalization and per-record validation
    - Normalize each raw record into `ReasoningRecord(problem, solution_steps, final_answer)`
    - Reject records with empty `problem`, empty/zero-length `solution_steps`, or empty `final_answer`; record reason counts
    - Preserve LaTeX delimiters and mathematical notation (no stripping)
    - _Requirements: 3.2, 3.3, 3.6_

  - [~] 4.4 Write property test for reasoning-format normalization invariants
    - **Property 8: Reasoning_Format normalization preserves invariants**
    - **Validates: Requirements 3.2, 3.6**
    - _Requirements: 3.2, 3.6_

  - [~] 4.5 Write property test for dataset accounting balance
    - **Property 9: Dataset accounting balance**
    - **Validates: Requirements 3.3**
    - `len(input_records) == len(accepted_records) + sum(rejection_reasons.values())`
    - _Requirements: 3.3_

  - [~] 4.6 Implement canonicalization and deduplication
    - Default canonicalization: lowercase, trim, collapse whitespace, strip documented trailing punctuation
    - Function exposes `canonicalization_fn_id` and `canonicalization_fn_version` written into the dataset card
    - Deduplicate by canonicalized `problem`, recording the dedup count
    - _Requirements: 3.4_

  - [~] 4.7 Write property test for deduplication completeness
    - **Property 10: Deduplication completeness**
    - **Validates: Requirements 3.4**
    - _Requirements: 3.4_

  - [~] 4.8 Implement train/validation split with anti-leakage guarantee
    - Deterministic seeded split with `val_fraction in [0.05, 0.20]`
    - Guarantee no `Eval_Dataset` problem appears in `Training_Dataset` by exact match on canonicalized `problem`
    - _Requirements: 3.5_

  - [~] 4.9 Write property test for validation split correctness
    - **Property 11: Validation split correctness**
    - **Validates: Requirements 3.5**
    - _Requirements: 3.5_

  - [~] 4.10 Implement tokenizer-aware truncation that preserves `final_answer`
    - When tokenized record exceeds `max_seq_len` and tokenized `final_answer` alone fits, truncate the prefix while retaining the full `final_answer` token sequence
    - When tokenized `final_answer` alone exceeds `max_seq_len`, reject the record into `final_answer_too_long_rejection_count` (distinct from `truncation_count`)
    - _Requirements: 3.8, 3.9_

  - [~] 4.11 Write property test for tokenizer-aware truncation
    - **Property 12: Tokenizer-aware truncation preserves final_answer**
    - **Validates: Requirements 3.8, 3.9**
    - _Requirements: 3.8, 3.9_

  - [~] 4.12 Implement `Custom_Integral_Set` exclusion from training
    - Use the same canonicalization function from task 4.6 to remove every `Custom_Integral_Set` problem from the training dataset; record the count in the dataset card
    - _Requirements: 3.10_

  - [~] 4.13 Write property test for `Custom_Integral_Set` isolation
    - **Property 13: Custom_Integral_Set isolation**
    - **Validates: Requirements 3.10**
    - _Requirements: 3.10_

  - [~] 4.14 Implement dataset-card emission and content hashing
    - Per-source `DatasetCard` with all fields from the design (record counts at each stage, rejection reasons, truncation count, dedup count, val-split seed/fraction, content hash)
    - Top-level `content_hash` over the final concatenated training set
    - Persist artifacts under `{artifact_root}/datasets/{content_hash}/` with `training.parquet`, `validation.parquet`, and `cards/` per the design's storage layout
    - _Requirements: 3.7_

  - [~] 4.15 Write unit test for dataset-card emission
    - Example test producing a small dataset and asserting card field completeness
    - _Requirements: 3.7_

- [~] 5. Checkpoint - Pre-training components built
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. Implement `LoRA_Trainer`
  - [~] 6.1 Implement `LoRA_Config` validation and field-level error reporting
    - Validate `r in [4, 128]`, `dropout in [0.0, 1.0]`, `bias in {none, all, lora_only}`, and that every entry in `target_modules` matches a real module name in the loaded base model
    - Raise `LoRAConfigInvalid` naming the offending field
    - _Requirements: 4.1, 4.2, 4.9_

  - [~] 6.2 Write property test for `LoRA_Config` field-level validation
    - **Property 18: LoRA_Config field-level validation**
    - **Validates: Requirements 4.2, 4.9**
    - _Requirements: 4.2, 4.9_

  - [~] 6.3 Implement `configure(...)` that applies LoRA via `peft` with default `target_modules` resolution
    - When `target_modules` is unset, default to `["q_proj", "v_proj"]` and record the resolved names
    - Freeze every base-model parameter; ensure only LoRA adapter parameters on resolved targets have `requires_grad == True`
    - Report trainable count, total count, and ratio
    - _Requirements: 4.3, 4.4, 4.6_

  - [~] 6.4 Write property test for trainable parameter set correctness
    - **Property 14: Trainable parameter set equals LoRA target set**
    - **Validates: Requirements 4.3, 4.4, 4.6**
    - Use a mocked parameter graph (list of named parameters with `requires_grad` flags) per the design's mocking strategy
    - _Requirements: 4.3, 4.4, 4.6_

  - [~] 6.5 Implement `nf4` quantization layout
    - When `Quantization_Mode == nf4`, load base weights in 4-bit via `bitsandbytes`; keep LoRA weights in `bf16`/`fp16`
    - _Requirements: 4.8_

  - [~] 6.6 Write property test for `nf4` quantization layout
    - **Property 17: nf4 honors quantization layout**
    - **Validates: Requirements 4.8**
    - Mocked dtype check across base and adapter tensors
    - _Requirements: 4.8_

  - [~] 6.7 Implement adapter serialization (`save_adapter`) and load
    - Write LoRA weights, resolved `lora_config.json`, `base_model_ref.json`, and `adapter_card.json` per the design's storage layout
    - Reject overwrite atomically: if the path exists for the given `run_id`, raise `AdapterAlreadyExists` and leave existing adapter byte-identical
    - Implement `load_adapter` for round-trip use by `Inference_Server` and tests
    - _Requirements: 4.5, 4.7_

  - [~] 6.8 Write property test for adapter serialization round-trip
    - **Property 15: Adapter serialization round-trip**
    - **Validates: Requirements 4.5**
    - Bit-identical weights, field-equal `LoRAConfig` and `base_model_ref` after save/load
    - _Requirements: 4.5_

  - [~] 6.9 Write property test for atomic adapter-overwrite refusal
    - **Property 16: Adapter overwrite is refused atomically**
    - **Validates: Requirements 4.7**
    - Use the in-memory filesystem stub from the design's mocking strategy
    - _Requirements: 4.7_

- [ ] 7. Implement `Training_Pipeline` orchestration
  - [~] 7.1 Implement seed setting before any data shuffling
    - Set seeds for `random`, `numpy`, framework CPU, and framework accelerator generators; record all four in a structure available to `Experiment_Tracker`
    - _Requirements: 5.6_

  - [~] 7.2 Write property test for seed-controlled determinism
    - **Property 24: Seed-controlled determinism**
    - **Validates: Requirements 5.6**
    - Two simulated runs with identical seeds produce identical first-batch token sequences and identical recorded seeds
    - _Requirements: 5.6_

  - [~] 7.3 Implement `pre_train_gate()`
    - Call `Model_Selector.select`, require chosen model
    - Call `Hardware_Budget_Planner.plan`, require all projections within limits
    - Call `LoRA_Trainer.validate(lora_config)`
    - When resuming, require checkpoint integrity
    - On any failure halt before the first training step and trigger `Experiment_Tracker.persist_halt_manifest()`
    - _Requirements: 2.3, 2.6, 2.7, 2.8, 4.9, 5.4_

  - [~] 7.4 Implement training-step scheduling (`max_steps`, `max_epochs`, `min(...)` halt)
    - Halt at `min(max_steps, ceil(max_epochs * dataset_size / batch_size))` steps
    - _Requirements: 5.1_

  - [~] 7.5 Write property test for training halt step
    - **Property 19: Training halts at the correct step**
    - **Validates: Requirements 5.1**
    - State-machine property over simulated event traces (no real training)
    - _Requirements: 5.1_

  - [~] 7.6 Implement logging and validation-evaluation cadence
    - Log loss, validation loss, learning rate, tokens/sec at every `logging_interval_steps`
    - Evaluate validation split at every `validation_interval_steps`
    - _Requirements: 5.5, 5.8_

  - [~] 7.7 Write property test for periodic event scheduling
    - **Property 23: Periodic event scheduling**
    - **Validates: Requirements 5.5, 5.8**
    - _Requirements: 5.5, 5.8_

  - [~] 7.8 Implement checkpoint write policy and retention
    - Write checkpoint every `checkpoint_interval_steps` containing adapter weights, optimizer state, scheduler state, RNG states, step counter
    - On-disk checkpoint set always equals `{latest, best_val}`; when `latest == best_val` retain only one
    - _Requirements: 5.2_

  - [~] 7.9 Write property test for checkpoint retention invariant
    - **Property 20: Checkpoint retention invariant**
    - **Validates: Requirements 5.2**
    - Use in-memory filesystem stub
    - _Requirements: 5.2_

  - [~] 7.10 Implement `resume(checkpoint_path, configs)` with integrity check
    - Restore adapter weights, optimizer state, scheduler state, RNG states, step counter before any further training step
    - On missing path, integrity-check failure, or missing field, halt with `CheckpointInvalid` naming the path and field
    - _Requirements: 5.3, 5.4_

  - [~] 7.11 Write property test for resume round-trip equivalence
    - **Property 21: Resume round-trip equivalence**
    - **Validates: Requirements 5.3**
    - Train 0..N vs. train 0..k then resume k..N produce equal metrics within numeric tolerance over simulated traces
    - _Requirements: 5.3_

  - [~] 7.12 Write property test for corrupt-checkpoint rejection
    - **Property 22: Corrupt checkpoint rejection**
    - **Validates: Requirements 5.4**
    - _Requirements: 5.4_

  - [~] 7.13 Implement non-finite loss halt and diagnostic
    - On non-finite loss at step `k`, halt within step `k` and emit a diagnostic record with `step`, last learning rate, and path to the most recent finite-val-loss checkpoint
    - Surface `NonFiniteLoss` and trigger halt-manifest persistence
    - _Requirements: 5.7_

  - [~] 7.14 Write property test for non-finite loss handling
    - **Property 25: Non-finite loss triggers immediate halt with diagnostic**
    - **Validates: Requirements 5.7**
    - _Requirements: 5.7_

  - [~] 7.15 Implement multi-GPU metric aggregation
    - Aggregate per-rank loss as mean, tokens/sec as sum before each logging event
    - Record `strategy`, `world_size`, `per_rank_batch_size` in the run manifest input
    - _Requirements: 5.9_

  - [~] 7.16 Write property test for multi-GPU metric aggregation
    - **Property 26: Multi-GPU metric aggregation**
    - **Validates: Requirements 5.9**
    - _Requirements: 5.9_

  - [~] 7.17 Wire VRAM sampling helper from task 3.7 into the training loop
    - Sample peak VRAM per GPU at least once per 100 steps; append to manifest input via `Experiment_Tracker`
    - _Requirements: 2.10_

- [~] 8. Checkpoint - Trainer and pipeline core ready
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 9. Implement `Evaluator`
  - [~] 9.1 Implement decoding parameter validation
    - Validate `temperature in [0.0, 2.0]`, `top_p in [0.0, 1.0]`, `top_k >= 0`, `max_new_tokens > 0`
    - Raise `DecodingParamsInvalid` naming the field on violation
    - _Requirements: 7.7 (used by Evaluator), 6.3_

  - [~] 9.2 Write property test for decoding-parameter validation
    - **Property 35: Decoding parameter validation**
    - **Validates: Requirements 7.7**
    - _Requirements: 7.7_

  - [~] 9.3 Implement GSM8K and MATH answer-extraction and exact-match scoring
    - GSM8K: `#### <answer>` extraction with exact-match on the numeric answer
    - MATH: `\boxed{...}` extraction with the published normalization rules
    - Cite protocol source and version in the result
    - _Requirements: 6.5_

  - [~] 9.4 Write integration test for GSM8K and MATH scoring on a small fixed sample
    - 10 items per benchmark with reference scores; verify protocol output matches
    - _Requirements: 6.5_

  - [~] 9.5 Implement `Custom_Integral_Set` loader and structural validation
    - Enforce `len >= 50`, at least 10 problems per category in `{indefinite, definite, derivation}`, every problem declares `reference_final_answer` and an `equivalence_rule` ∈ `{string_equality, numerical_equality_with_tolerance, symbolic_equivalence}`
    - _Requirements: 6.6_

  - [~] 9.6 Write property test for `Custom_Integral_Set` structural constraints
    - **Property 29: Custom_Integral_Set structural constraints**
    - **Validates: Requirements 6.6**
    - _Requirements: 6.6_

  - [~] 9.7 Implement equivalence-rule scoring
    - `string_equality`: whitespace-normalized exact match
    - `numerical_equality_with_tolerance`: `abs(a - b) <= tolerance` per problem
    - `symbolic_equivalence`: SymPy `simplify(a - b) == 0`
    - Unparseable responses count as parse failures and as incorrect
    - _Requirements: 6.7, 6.8_

  - [~] 9.8 Write property test for equivalence-rule semantics
    - **Property 30: Equivalence rule semantics**
    - **Validates: Requirements 6.7**
    - May use real SymPy in-process per the design
    - _Requirements: 6.7_

  - [~] 9.9 Write property test for evaluation accounting balance
    - **Property 31: Evaluation accounting balance**
    - **Validates: Requirements 6.8**
    - `parse_failure_count + semantic_error_count + correct_count == S`
    - _Requirements: 6.8_

  - [~] 9.10 Implement `quick_eval` mode with stratified sampling
    - Stratified subset with `fraction in [0.10, 1.00]`, deterministic seed
    - Label every score in the report and run manifest as `quick_eval`
    - _Requirements: 6.10_

  - [~] 9.11 Write property test for `quick_eval` determinism and stratification
    - **Property 32: quick_eval determinism and stratification**
    - **Validates: Requirements 6.10**
    - _Requirements: 6.10_

  - [~] 9.12 Implement baseline and post-training evaluation harness
    - `evaluate(model_under_test, benchmarks, decoding_params, mode, quick_eval_fraction, seed)` returning `EvaluationReport`
    - Run `baseline_eval` on `Base_Model` alone before the first training step
    - Run `post_training_eval` on `Base_Model + Adapter` after the final training step
    - Both runs use identical decoding parameters (including seed) and identical prompts
    - Compute absolute diff and relative pct change per benchmark; produce stratified `Custom_Integral_Set` breakdown
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.9_

  - [~] 9.13 Write property test for evaluation regime invariants
    - **Property 27: Evaluation regime invariants**
    - **Validates: Requirements 6.1, 6.2, 6.3**
    - _Requirements: 6.1, 6.2, 6.3_

  - [~] 9.14 Write property test for score difference arithmetic
    - **Property 28: Score difference arithmetic**
    - **Validates: Requirements 6.4**
    - Including the documented sentinel for `baseline_score == 0`
    - _Requirements: 6.4_

  - [~] 9.15 Write unit test for evaluation report structure
    - Example test that `EvaluationReport` contains the per-benchmark fields and the `indefinite/definite/derivation` breakdown required by Req 6.9
    - _Requirements: 6.9_

- [ ] 10. Implement `Inference_Server`
  - [~] 10.1 Implement `load_base` and `load_adapter` with `(base_model_id, revision)` validation
    - Reject load when adapter's recorded base id or revision mismatches the loaded base; surface `AdapterBaseMismatch` naming the mismatched field; leave adapter unloaded
    - _Requirements: 7.1, 7.2_

  - [~] 10.2 Write property test for adapter-base compatibility check
    - **Property 33: Adapter-base compatibility check**
    - **Validates: Requirements 7.2**
    - _Requirements: 7.2_

  - [~] 10.3 Implement adapter management with `set_active_adapter` and `no_adapter`
    - Switch active adapter without unloading the base model
    - Honor `no_adapter` for base-only inference
    - Enforce maximum number of loaded adapters from inference config
    - _Requirements: 7.1, 7.4, 7.5_

  - [~] 10.4 Write property test for adapter switching preserving base load
    - **Property 34: Adapter switching preserves base load**
    - **Validates: Requirements 7.4, 7.5**
    - _Requirements: 7.4, 7.5_

  - [~] 10.5 Implement `generate(...)` producing `Reasoning_Format` responses
    - Response is an ordered list of intermediate steps followed by a final answer separated by the configured delimiter
    - Validate decoding parameters via the validator from task 9.1
    - Record decoding params in run-manifest input on benchmark evaluation requests
    - _Requirements: 7.3, 7.7_

  - [~] 10.6 Implement adapter merge into a standalone artifact and equivalence check
    - `merge_adapter(handle, adapter_name, output_path)` produces a merged model whose token sequences match the unmerged `Base + Adapter` on a fixed prompt set of at least 10 prompts under greedy decoding (`temperature 0.0, top_p 1.0, top_k 0`) with a recorded seed
    - Record prompt set identifier and comparison result in run-manifest input
    - _Requirements: 7.8_

  - [~] 10.7 Write property test for adapter merge equivalence
    - **Property 36: Adapter merge equivalence**
    - **Validates: Requirements 7.8**
    - State-machine property over a mocked transformer; integration coverage handled in task 12.x
    - _Requirements: 7.8_

  - [~] 10.8 Implement 4-bit inference path and accuracy-delta recording
    - When training used `Quantization_Mode == nf4`, load the base model in 4-bit for inference and record per-benchmark accuracy delta vs. inference at the training precision on the evaluation set defined in Req 6
    - _Requirements: 7.6_

  - [~] 10.9 Write unit test for inference-server config and 4-bit accuracy-delta recording
    - Smoke test that the 4-bit accuracy delta is computed and routed into the manifest field
    - _Requirements: 7.1, 7.3, 7.6_

- [ ] 11. Implement `Experiment_Tracker`
  - [~] 11.1 Implement `open_run`, environment capture, and config recording
    - Capture `run_id`, start UTC timestamp, git commit hash, OS identifier, Python version, accelerator driver version
    - Record resolved configs, `Base_Model` id/revision, `Quantization_Mode`, `LoRAConfig`, `Hardware_Profile`, `Budget_Profile`, all four random seeds, dataset cards, dataset content hashes
    - Resolve dependency lockfile (every package in the dependency manifest pinned to a non-null version)
    - _Requirements: 8.1, 8.2, 8.3, 8.9_

  - [~] 11.2 Implement metric streaming (`log_metric`, `record_event`)
    - Append metrics from `Training_Pipeline` and `Evaluator` to the in-progress run manifest, including `vram_samples`, `metrics`, multi-GPU info, baseline/post-training evaluation reports
    - _Requirements: 8.4_

  - [~] 11.3 Write property test for faithful manifest recording
    - **Property 37: Faithful manifest recording**
    - **Validates: Requirements 8.1, 8.2, 8.3, 8.4**
    - _Requirements: 8.1, 8.2, 8.3, 8.4_

  - [~] 11.4 Write property test for dependency-pinning completeness
    - **Property 41: Dependency pinning completeness**
    - **Validates: Requirements 8.9**
    - _Requirements: 8.9_

  - [~] 11.5 Implement dirty-tree recording
    - When `git status` is dirty, record `git_dirty == True`, parent commit hash, and a content hash over uncommitted changes; otherwise `git_dirty == False` and `git_dirty_content_hash == null`
    - _Requirements: 8.8_

  - [~] 11.6 Write property test for dirty-tree recording
    - **Property 40: Dirty-tree recording**
    - **Validates: Requirements 8.8**
    - _Requirements: 8.8_

  - [~] 11.7 Implement `close_run` persistence with retry contract
    - Persist `Run_Manifest` within 60 seconds of the terminal-state event for both `completed` and `halted` statuses; halted manifests record `halt_reason` and `last_completed_step`
    - Retry persistence up to 3 times on failure; on exhaustion surface `ManifestPersistenceFailed` with manifest path and failure cause
    - _Requirements: 8.5, 8.6, 8.7_

  - [~] 11.8 Write property test for manifest persistence latency
    - **Property 38: Manifest persistence latency**
    - **Validates: Requirements 8.5, 8.6**
    - Use the mock clock from the design's mocking strategy
    - _Requirements: 8.5, 8.6_

  - [~] 11.9 Write property test for persistence retry contract
    - **Property 39: Persistence retry contract**
    - **Validates: Requirements 8.7**
    - Inject failure patterns over four consecutive attempts
    - _Requirements: 8.7_

  - [~] 11.10 Implement reproducibility tolerance recording
    - Record `reproducibility_tolerance == 0.001` in every persisted manifest
    - _Requirements: 8.10_

  - [~] 11.11 Write property test for reproducibility tolerance
    - **Property 42: Reproducibility tolerance**
    - **Validates: Requirements 8.10**
    - For two manifests with byte-equal inputs, every shared evaluation metric differs by at most `0.001`
    - _Requirements: 8.10_

  - [~] 11.12 Implement `diff_runs(manifest_a, manifest_b, tolerance=0.001)`
    - Empty diff if and only if the two manifests' serialized bytes are equal
    - Metrics whose absolute difference is at most the tolerance do not appear in the diff
    - Configuration fields and dataset content hashes that differ always appear in the diff
    - _Requirements: 8.11_

  - [~] 11.13 Write property test for manifest diff identity
    - **Property 43: Manifest diff identity**
    - **Validates: Requirements 8.11**
    - _Requirements: 8.11_

- [ ] 12. Wire components into the end-to-end run
  - [~] 12.1 Wire `Training_Pipeline.run(configs)` end-to-end
    - Sequence: `pre_train_gate -> Experiment_Tracker.open_run -> Evaluator.baseline_eval -> training loop (with checkpoints, logging, validation, VRAM sampling) -> Evaluator.post_training_eval -> Experiment_Tracker.close_run` per the design's sequence diagram
    - Halt paths route to `close_run(status="halted", halt_reason=...)` per the halt-vs-continue policy
    - _Requirements: 5.1, 5.2, 5.5, 5.7, 5.8, 6.1, 6.2, 8.5, 8.6_

  - [~] 12.2 Wire `Training_Pipeline.resume(checkpoint_path, configs)` path
    - Reuses `pre_train_gate` (with checkpoint integrity check) and the same end-to-end sequence
    - _Requirements: 5.3, 5.4_

  - [~] 12.3 Wire end-of-run cost reconciliation for cloud deployments
    - When `Hardware_Profile.deployment == "cloud"`, call `Hardware_Budget_Planner.reconcile_cost(...)` and attach the result to the manifest
    - _Requirements: 2.12_

  - [~] 12.4 Wire `Inference_Server` to consume saved adapters and contribute manifest fields
    - Hand merge-comparison results and 4-bit accuracy delta back to `Experiment_Tracker` for the manifest
    - _Requirements: 7.6, 7.8_

  - [~] 12.5 Write integration test: end-to-end smoke run
    - One small base model (e.g., a tiny test transformer or `Qwen2.5-0.5B` if locally feasible), small dataset, ≤100 steps; assert the `Run_Manifest` is produced and contains every field required by Property 37
    - _Requirements: 5.1, 5.2, 5.5, 5.6, 5.7, 5.8, 6.1, 6.2, 6.3, 6.4, 6.9, 8.1, 8.2, 8.3, 8.4, 8.5_

  - [~] 12.6 Write integration test: real adapter merge round-trip
    - Greedy decoding token-equality on a fixed prompt set ≥ 10 between unmerged and merged artifacts
    - _Requirements: 7.8_

  - [~] 12.7 Write integration test: cloud-cost reconciliation
    - Stubbed cloud GPU; verify `actual_cost`, `absolute_diff`, `pct_diff` and manifest attachment
    - _Requirements: 2.12_

  - [~] 12.8 Write integration test: reproducibility within tolerance
    - Two runs with byte-identical inputs (configs, dataset hashes, seeds, `git_dirty == False`, dependency versions, hardware profile); assert per-metric absolute diff ≤ 0.001
    - _Requirements: 8.10_

- [~] 13. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP; they cover the property tests, smoke tests, and integration tests called out in the design's Testing Strategy.
- Each property-based test corresponds to exactly one of the 43 correctness properties in the design and is annotated with its property number and the requirement clauses it validates.
- Property tests are placed close to the implementation they cover so failures surface during the same task group.
- Checkpoints (tasks 5, 8, 13) are deliberate gates where the user can review test status before continuing.
- Per the design's testing strategy, property tests use mocked backends (parameter graph, optimizer/scheduler/RNG state, clock, in-memory filesystem, GPU-memory measurement) so 100+ Hypothesis iterations remain cost-effective. Real GPU and benchmark behavior is covered by integration tests instead.
- Tasks for Req 2.9 (feature-toggle availability) and Req 2.11 (consumer-GPU configuration) are covered by smoke/integration sub-tasks (3.11) per the design's "Items Not Suited to Property-Based Testing" list.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2"] },
    { "id": 2, "tasks": ["1.3", "1.4"] },
    { "id": 3, "tasks": ["2.1", "3.1", "3.2", "4.1"] },
    { "id": 4, "tasks": ["2.2", "2.3", "3.3", "3.4", "3.7", "3.9", "4.2", "4.3", "4.6", "6.1"] },
    { "id": 5, "tasks": ["2.4", "3.5", "3.8", "3.10", "4.4", "4.5", "4.7", "4.8", "4.10", "4.12", "6.2", "6.3", "6.5", "6.7", "9.1"] },
    { "id": 6, "tasks": ["2.5", "2.6", "2.8", "3.6", "3.11", "4.9", "4.11", "4.13", "4.14", "6.4", "6.6", "6.8", "9.2", "9.3", "9.5", "9.7", "9.10", "10.1"] },
    { "id": 7, "tasks": ["2.7", "4.15", "6.9", "9.4", "9.6", "9.8", "9.9", "9.11", "10.2", "10.3", "10.5", "10.8"] },
    { "id": 8, "tasks": ["7.1", "7.3", "7.4", "9.12", "10.4", "10.6"] },
    { "id": 9, "tasks": ["7.2", "7.5", "7.6", "7.8", "7.10", "7.13", "7.15", "7.17", "9.13", "9.14", "9.15", "10.7", "10.9"] },
    { "id": 10, "tasks": ["7.7", "7.9", "7.11", "7.12", "7.14", "7.16", "11.1", "11.2", "11.5", "11.7", "11.10", "11.12"] },
    { "id": 11, "tasks": ["11.3", "11.4", "11.6", "11.8", "11.9", "11.11", "11.13", "12.1", "12.2", "12.3", "12.4"] },
    { "id": 12, "tasks": ["12.5", "12.6", "12.7", "12.8"] }
  ]
}
```

