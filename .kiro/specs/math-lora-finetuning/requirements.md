# Requirements Document

## Introduction

This feature delivers a LoRA (Low-Rank Adaptation) fine-tuning pipeline to enrich an existing open-weight large language model with stronger mathematical reasoning capability. The target capability includes step-by-step solutions for medium-level university mathematics problems, complicated indefinite and definite integrals, and symbolic derivations.

The project is operated by a single student under tight hardware and monetary budget constraints. Candidate base models include Qwen, DeepSeek, and Doubao families, and the choice of base model and hardware scale are first-class concerns of this specification rather than incidental implementation details.

The system covers eight concerns: base model selection, hardware and budget feasibility, training data preparation, LoRA configuration, training pipeline execution, evaluation against math benchmarks, inference and adapter serving, and reproducibility and experiment tracking.

## Glossary

- **LoRA_Trainer**: The training component that applies Low-Rank Adaptation to a frozen base LLM, producing LoRA adapter weights.
- **Base_Model**: The pretrained open-weight LLM that is loaded in frozen form and adapted via LoRA. Candidates considered are Qwen, DeepSeek, and Doubao open-weight checkpoints.
- **Model_Selector**: The component that scores and selects a Base_Model from the candidate set against declared hardware and capability constraints.
- **Hardware_Profile**: A declarative description of the available compute, including GPU model, VRAM in gigabytes, system RAM, disk space, and whether the environment is local consumer hardware or rented cloud GPUs.
- **Budget_Profile**: A declarative description of monetary and time constraints, including a total compute budget in monetary units and a maximum wall-clock training time.
- **Quantization_Mode**: The numerical precision used to load Base_Model weights during training. Supported values are `fp16`, `bf16`, `int8`, and `nf4` (the QLoRA 4-bit format).
- **LoRA_Config**: The set of LoRA hyperparameters: rank `r`, scaling factor `alpha`, dropout, target module names, and bias mode.
- **Adapter**: The set of trained LoRA weight matrices stored as a separate artifact from Base_Model.
- **Training_Dataset**: The collection of math problem and solution pairs in step-by-step format used to train the Adapter.
- **Eval_Dataset**: The held-out collection of math problems used to score model capability before and after fine-tuning.
- **Math_Benchmark**: A standardized math evaluation set. In scope: GSM8K, MATH, and a custom integral and derivation test set authored for this project.
- **Custom_Integral_Set**: An evaluator-authored test set of indefinite integrals, definite integrals, and symbolic derivations with reference answers.
- **Training_Pipeline**: The end-to-end orchestration that loads data, runs LoRA_Trainer, writes checkpoints, and emits metrics.
- **Inference_Server**: The component that loads Base_Model plus one or more Adapters and answers math prompts.
- **Experiment_Tracker**: The component that records hyperparameters, dataset versions, git commit, hardware fingerprint, and metrics for each training run.
- **Run_Manifest**: A serialized record produced by Experiment_Tracker that fully describes one training run.
- **Reasoning_Format**: The structured prompt and response format that contains the problem statement, a chain of intermediate steps, and a final answer delimited from the steps.

## Requirements

### Requirement 1: Base Model Selection

**User Story:** As a student researcher with limited hardware, I want the system to select a base model from the Qwen, DeepSeek, and Doubao families that fits my hardware while preserving math capability, so that I do not waste compute training a model that cannot run or that has weak math priors.

#### Acceptance Criteria

1. THE Model_Selector SHALL accept as input a list of candidate Base_Models, where each candidate declares its parameter count in billions, license identifier, native context length in tokens, tokenizer family, a baseline GSM8K score expressed as accuracy in the range 0.0 to 1.0, and a baseline MATH score expressed as accuracy in the range 0.0 to 1.0.
2. THE Model_Selector SHALL accept a Hardware_Profile declaring total VRAM in gigabytes as an integer between 1 and 1024, system RAM in gigabytes, and accelerator family, together with a Quantization_Mode whose value is exactly one of `fp16`, `bf16`, `int8`, or `nf4`.
3. THE Model_Selector SHALL accept a training sequence length in tokens, where the accepted range is 128 to 8192 inclusive, and SHALL compute for each candidate Base_Model an estimated minimum VRAM requirement in gigabytes for LoRA training under the given Quantization_Mode, accounting for base model weights, LoRA adapter gradients, optimizer state for trainable parameters only, and activation memory for the given sequence length, using a documented formula whose inputs and coefficients are exposed in the selection report.
4. WHEN the estimated minimum VRAM requirement for a candidate exceeds the Hardware_Profile VRAM, THE Model_Selector SHALL mark that candidate as infeasible and SHALL record the shortfall in gigabytes rounded to one decimal place.
5. THE Model_Selector SHALL rank feasible candidates by a deterministic scoring function that produces a numeric score in the range 0.0 to 1.0 from a documented weighted combination of normalized baseline GSM8K score, normalized baseline MATH score, parameter count, and a license permissiveness value derived from whether the license permits fine-tuning, redistribution of Adapters, and commercial use, where the weights and the tie-breaker rule used when two candidates produce identical scores are listed in the selection report.
6. THE Model_Selector SHALL emit a selection report that lists, for every candidate, its declared parameter count, its estimated minimum VRAM requirement in gigabytes, its feasibility status, its score, the weights used in the scoring function, and the identifier of the chosen Base_Model.
7. IF no candidate Base_Model is feasible under the given Hardware_Profile and Quantization_Mode, THEN THE Model_Selector SHALL report, for the highest-scoring candidate, the smallest VRAM increase in gigabytes that would make it feasible at the current Quantization_Mode and the alternative Quantization_Mode from the set `fp16`, `bf16`, `int8`, `nf4` that would make it feasible without any VRAM increase, or SHALL state that no such alternative exists.
8. THE Model_Selector SHALL record, for each candidate Base_Model, three boolean flags indicating whether the candidate's published license file permits fine-tuning, redistribution of Adapters, and commercial use, together with the license identifier, and SHALL include these flags in the selection report.
9. IF the input candidate list is empty, THEN THE Model_Selector SHALL not select any Base_Model and SHALL emit an error indicating that no candidates were supplied.
10. IF a candidate Base_Model is missing any of the declared fields required by criterion 1 or any of the license flags required by criterion 8, THEN THE Model_Selector SHALL mark that candidate as ineligible, exclude it from ranking, and record the missing field names in the selection report.

### Requirement 2: Hardware and Budget Feasibility

**User Story:** As a student on a tight budget, I want the system to plan training within an explicit Hardware_Profile and Budget_Profile, so that I do not start a run that exceeds my GPU memory or my available compute budget.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL load a Hardware_Profile from a configuration file that declares GPU model, GPU count, VRAM per GPU in gigabytes, system RAM in gigabytes, and disk space in gigabytes.
2. THE Training_Pipeline SHALL load a Budget_Profile from a configuration file that declares a maximum monetary cost in a stated currency, a maximum wall-clock training time in hours, and a cost rate per GPU-hour in the same stated currency.
3. IF the Hardware_Profile or Budget_Profile fails to load or is missing any field declared in criterion 1 or criterion 2, THEN THE Training_Pipeline SHALL halt before training begins and SHALL surface an error identifying the file and the missing or invalid field.
4. WHEN the Hardware_Profile declares VRAM per GPU below 24 gigabytes, THE Training_Pipeline SHALL set Quantization_Mode to `nf4` (QLoRA 4-bit) by default, and an operator SHALL be able to override this default by setting Quantization_Mode explicitly in the training configuration.
5. WHEN training is invoked, THE Training_Pipeline SHALL produce, before any training step executes, a pre-flight report that records the projected wall-clock training time in hours, the projected monetary cost in the declared currency derived from the cost rate per GPU-hour times the projected GPU-hours, and the projected peak VRAM in gigabytes per GPU, computed from the Base_Model size, Quantization_Mode, batch size, sequence length, and gradient accumulation settings.
6. IF the projected peak VRAM per GPU from criterion 5 exceeds the Hardware_Profile VRAM per GPU, THEN THE Training_Pipeline SHALL halt before any training step and SHALL include in the halt report the projected VRAM, the available VRAM, and a suggested reduction in batch size, sequence length, or activation-memory setting that brings the projection within the available VRAM.
7. IF the projected wall-clock time from criterion 5 exceeds the Budget_Profile time limit, THEN THE Training_Pipeline SHALL halt with a report that includes the projected wall-clock time, the time limit, and a suggested reduction in steps, batch size, sequence length, or dataset size, with the suggested value for the chosen knob, that brings the projection within the time limit.
8. IF the projected monetary cost from criterion 5 exceeds the Budget_Profile cost limit, THEN THE Training_Pipeline SHALL halt with a report that includes the projected cost, the cost limit, and a suggested reduction in steps, batch size, sequence length, or dataset size, with the suggested value for the chosen knob, that brings the projection within the cost limit.
9. THE Training_Pipeline SHALL support gradient checkpointing, gradient accumulation, mixed precision in `bf16` or `fp16`, and 4-bit quantization via `nf4` to reduce VRAM consumption.
10. WHILE training is in progress, THE Training_Pipeline SHALL sample peak VRAM usage per GPU at least once per 100 training steps within a window of 100 consecutive steps, and SHALL record each sample together with its training step index to the Run_Manifest.
11. WHERE a single consumer GPU with 8 to 16 gigabytes of VRAM is the only available hardware, THE Training_Pipeline SHALL provide a documented configuration for the smallest feasible Base_Model identified by the Model_Selector under which training completes the configured number of steps without an out-of-memory error and with every recorded peak-VRAM sample at or below the Hardware_Profile VRAM per GPU.
12. WHERE rented cloud GPUs are used, THE Training_Pipeline SHALL emit, at the end of the run, a cost reconciliation that records the projected cost from criterion 5, the actual elapsed GPU-hours, the actual cost computed as actual elapsed GPU-hours times the declared cost rate, and the absolute and percentage difference between projected and actual cost.

### Requirement 3: Training Data Preparation

**User Story:** As a fine-tuning operator, I want a curated math dataset that contains step-by-step solutions, integrals, and derivations in a single Reasoning_Format, so that the Adapter learns to produce explicit intermediate steps rather than only final answers.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL ingest math training data from at least one of the following sources: GSM8K training split, MATH training split, a documented open-source step-by-step math corpus, and operator-supplied integral and derivation pairs.
2. THE Training_Pipeline SHALL normalize every Training_Dataset record into the Reasoning_Format, where each record contains a `problem` field as a non-empty string, a `solution_steps` field as an ordered list of one or more non-empty strings, and a `final_answer` field as a non-empty string that is delimited from the steps by a documented delimiter recorded in the dataset card.
3. THE Training_Pipeline SHALL reject any Training_Dataset record that lacks a non-empty `problem`, a `solution_steps` list with at least one non-empty entry, or a non-empty `final_answer` field, and SHALL record the count and reason for each rejection.
4. THE Training_Pipeline SHALL deduplicate Training_Dataset records by a documented canonicalization function applied to the `problem` text, where the canonicalization function is recorded in the dataset card, and SHALL record the deduplication count.
5. THE Training_Pipeline SHALL hold out a validation split that contains between five and twenty percent of the deduplicated Training_Dataset records, selected using a deterministic seed recorded in the dataset card, and SHALL guarantee that no Eval_Dataset problem appears in the Training_Dataset by exact-match on canonicalized `problem` text.
6. WHERE LaTeX is used in problems or solutions, THE Training_Pipeline SHALL preserve LaTeX delimiters and SHALL not strip mathematical notation during normalization.
7. THE Training_Pipeline SHALL produce, for each ingested source, a dataset card that records source name, license, record count after ingestion, record count after normalization, record count after deduplication, and a content hash of the final Training_Dataset.
8. THE Training_Pipeline SHALL apply tokenizer-aware truncation that preserves the `final_answer` field when the tokenized record exceeds the configured maximum sequence length, and SHALL count and record every truncation.
9. IF the `final_answer` field alone, after tokenization, exceeds the configured maximum sequence length, THEN THE Training_Pipeline SHALL reject that record rather than truncate the `final_answer`, and SHALL record the rejection count separately from the truncation count in the dataset card.
10. WHERE the operator supplies a Custom_Integral_Set for evaluation, THE Training_Pipeline SHALL exclude every problem in the Custom_Integral_Set from the Training_Dataset by exact-match on canonicalized `problem` text using the same canonicalization function from criterion 4, and SHALL record the count of excluded problems in the dataset card.

### Requirement 4: LoRA Configuration

**User Story:** As a fine-tuning operator, I want explicit, recorded LoRA hyperparameters and target module choices, so that I can reproduce results and reason about adapter quality versus size.

#### Acceptance Criteria

1. THE LoRA_Trainer SHALL accept a LoRA_Config that declares rank `r` as an integer, scaling factor `alpha` as a positive number, dropout as a real number in the closed interval from 0.0 to 1.0 inclusive, target module names as a list of strings, and bias mode as one of the values `none`, `all`, or `lora_only`.
2. THE LoRA_Trainer SHALL accept rank values in the closed interval from 4 to 128 inclusive.
3. THE LoRA_Trainer SHALL apply LoRA only to the Base_Model modules whose names match entries in `target_modules`, and SHALL freeze every other Base_Model parameter such that those parameters receive no gradient updates during training.
4. IF `target_modules` is unset in the LoRA_Config, THEN THE LoRA_Trainer SHALL apply LoRA to the attention query and value projection modules of the Base_Model and SHALL record the resolved module names in the Run_Manifest.
5. THE LoRA_Trainer SHALL serialize the trained Adapter to a directory that contains the LoRA weights, the resolved LoRA_Config including all values from criterion 1, and a reference to the Base_Model identifier and revision.
6. WHEN training completes, THE LoRA_Trainer SHALL report the trainable parameter count, the total Base_Model parameter count, and the ratio of trainable to total parameters.
7. THE LoRA_Trainer SHALL save Adapters from separate runs to storage paths keyed by run identifier, and IF an Adapter already exists at the resolved storage path for a given run identifier, THEN THE LoRA_Trainer SHALL reject the save operation and SHALL preserve the existing Adapter unchanged.
8. WHERE Quantization_Mode is `nf4`, THE LoRA_Trainer SHALL load Base_Model weights in 4-bit precision and SHALL keep LoRA weights in `bf16` or `fp16` precision.
9. IF the LoRA_Config contains a rank outside the closed interval from 4 to 128, a dropout value outside the closed interval from 0.0 to 1.0, a bias mode not in the set `none`, `all`, `lora_only`, or a target module name that does not match any module in the Base_Model, THEN THE LoRA_Trainer SHALL reject the configuration before training begins and SHALL surface an error indicating which field is invalid.

### Requirement 5: Training Pipeline Execution

**User Story:** As a fine-tuning operator, I want a deterministic, checkpointed training loop with clear failure modes, so that an interrupted run can resume without losing progress.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL execute training until the configured maximum number of training steps is reached or the configured maximum number of epochs is reached, whichever occurs first, and SHALL halt training at that limit.
2. THE Training_Pipeline SHALL write a checkpoint of the Adapter weights and the optimizer state at every `checkpoint_interval_steps` training steps, where `checkpoint_interval_steps` is a positive integer declared in the training configuration, and SHALL retain on disk the most recent checkpoint and the checkpoint with the lowest recorded validation loss.
3. WHEN the Training_Pipeline is started with a checkpoint path argument, THE Training_Pipeline SHALL resume from that checkpoint by restoring Adapter weights, optimizer state, learning rate scheduler state, the random number generator states recorded in criterion 6, and the training step counter, before executing any further training step.
4. IF the Training_Pipeline is started with a checkpoint path that does not exist, that fails an integrity check, or that lacks any of the fields listed in criterion 3, THEN THE Training_Pipeline SHALL halt before executing any training step and SHALL surface an error identifying the checkpoint path and the missing or invalid field.
5. THE Training_Pipeline SHALL log training loss, validation loss, learning rate, and tokens-per-second at every `logging_interval_steps` training steps, where `logging_interval_steps` is a positive integer declared in the training configuration.
6. WHEN the Training_Pipeline starts a run and before any data shuffling occurs, THE Training_Pipeline SHALL set the random seeds for the standard library random generator, the numerical computing library generator, and the deep learning framework CPU and accelerator generators in use, and SHALL record those seeds in the Run_Manifest.
7. IF a non-finite loss value is observed during training, THEN THE Training_Pipeline SHALL halt within the same training step and SHALL emit a diagnostic record that includes the step number, the most recent learning rate, and the path to the most recent checkpoint at which the validation loss was finite.
8. THE Training_Pipeline SHALL evaluate the validation split at every `validation_interval_steps` training steps, where `validation_interval_steps` is a positive integer declared in the training configuration, and SHALL record validation loss to the Experiment_Tracker for that step.
9. WHERE multi-GPU training is configured, THE Training_Pipeline SHALL record the data and model parallelism strategy, the world size, and the per-rank batch size in the Run_Manifest, and SHALL aggregate training loss, validation loss, and tokens-per-second across all ranks before each logging event in criterion 5.

### Requirement 6: Evaluation

**User Story:** As a student researcher, I want quantitative before-and-after evaluation on standard math benchmarks plus a custom integral and derivation set, so that I can defend the claim that LoRA fine-tuning improved mathematical capability.

#### Acceptance Criteria

1. WHEN a training run is invoked, THE Training_Pipeline SHALL evaluate the Base_Model without any Adapter on GSM8K, MATH, and the Custom_Integral_Set before the first training step executes, and SHALL record the resulting per-benchmark accuracy scores in the Run_Manifest as the baseline.
2. WHEN training completes, THE Training_Pipeline SHALL evaluate the Base_Model with the trained Adapter on the same GSM8K, MATH, and Custom_Integral_Set used in criterion 1, and SHALL record the resulting per-benchmark accuracy scores in the Run_Manifest as the post-training result.
3. THE Training_Pipeline SHALL run the baseline evaluation in criterion 1 and the post-training evaluation in criterion 2 with identical decoding parameters (temperature, top-p, top-k, max new tokens, random seed) and identical prompts, and SHALL record those decoding parameters once per run in the Run_Manifest.
4. THE Training_Pipeline SHALL compute, for each Math_Benchmark, the absolute score difference as a real number in the closed interval from -1.0 to 1.0 and the relative percentage change as a real number, between the baseline from criterion 1 and the post-training result from criterion 2.
5. THE Training_Pipeline SHALL score GSM8K and MATH using the published official answer-extraction and exact-match protocols for each benchmark, and SHALL cite the protocol source and version in the evaluation report.
6. THE Custom_Integral_Set SHALL contain at least 50 problems drawn from the categories of indefinite integrals, definite integrals, and symbolic derivations, with at least 10 problems in each category, and each problem SHALL declare a reference final answer and an equivalence rule whose value is exactly one of `string_equality`, `numerical_equality_with_tolerance`, or `symbolic_equivalence`.
7. THE Training_Pipeline SHALL score each Custom_Integral_Set problem by checking whether the model's extracted final answer matches the reference answer under the equivalence rule declared for that problem, where `string_equality` requires exact string match after whitespace normalization, `numerical_equality_with_tolerance` requires the absolute difference to be at most a per-problem tolerance value, and `symbolic_equivalence` requires the difference to simplify to zero under a documented symbolic-math library.
8. WHEN a model output cannot be parsed into a final answer for a benchmark item, THE Training_Pipeline SHALL count that response as incorrect for the affected benchmark and SHALL record the parse failure in a separate parse-failure counter distinct from the semantic-error counter.
9. THE Training_Pipeline SHALL emit an evaluation report that contains, per Math_Benchmark, the baseline score, the post-training score, the absolute score difference, the relative percentage change, the sample size, the parse-failure count, and a stratified breakdown of the Custom_Integral_Set by the categories indefinite, definite, and derivation.
10. WHERE the operator selects `quick_eval` mode, THE Training_Pipeline SHALL evaluate a stratified subset of each benchmark whose size is between 10 and 100 percent of the full benchmark, drawn using a deterministic seed recorded in the Run_Manifest, and SHALL label every score produced under `quick_eval` mode in the evaluation report and the Run_Manifest with the value `quick_eval`.

### Requirement 7: Inference and Adapter Serving

**User Story:** As a user of the fine-tuned model, I want to load the Base_Model with one or more Adapters and answer math problems with explicit step-by-step reasoning, so that I can use the trained capability and compare adapters side by side.

#### Acceptance Criteria

1. THE Inference_Server SHALL load a Base_Model identified by a name string and a revision string, and SHALL load between zero and an operator-configured maximum number of named Adapters into that Base_Model, where the maximum number is a positive integer declared in the inference configuration.
2. WHEN an Adapter is loaded, THE Inference_Server SHALL compare the Adapter's recorded Base_Model identifier and revision to the loaded Base_Model identifier and revision, and IF either value does not match, THEN THE Inference_Server SHALL reject the load operation, leave the Adapter unloaded, and surface an error identifying the mismatched field.
3. THE Inference_Server SHALL accept a math prompt and SHALL respond using the Reasoning_Format, where the response contains an ordered list of one or more intermediate reasoning steps followed by the final answer separated by a delimiter declared in the inference configuration.
4. WHEN a request specifies an active Adapter name that is currently loaded, THE Inference_Server SHALL switch the active Adapter to that name without unloading the Base_Model and SHALL serve the request using the newly active Adapter.
5. WHEN a request specifies `no_adapter` as the active Adapter, THE Inference_Server SHALL serve the request using the Base_Model alone with no Adapter applied.
6. WHERE Quantization_Mode `nf4` was used during training, THE Inference_Server SHALL load the Base_Model in 4-bit precision for inference, and SHALL record in the Run_Manifest the absolute accuracy difference per Math_Benchmark between 4-bit inference and inference at the precision used during training, measured on the same evaluation set defined in Requirement 6.
7. THE Inference_Server SHALL accept decoding parameters consisting of temperature as a real number in the closed interval from 0.0 to 2.0, top-p as a real number in the closed interval from 0.0 to 1.0, top-k as a non-negative integer, max new tokens as a positive integer, and a random seed as an integer, and SHALL record the values used for any benchmark evaluation request in the Run_Manifest.
8. WHEN an Adapter is merged into the Base_Model weights to produce a standalone model artifact, THE Inference_Server SHALL produce, for the resulting merged artifact, outputs whose token sequences are identical to the outputs of the unmerged Base_Model plus Adapter on a fixed set of at least 10 prompts evaluated with greedy decoding (temperature 0.0, top-p 1.0, top-k 0) and a recorded random seed, and SHALL record the prompt set identifier and the comparison result in the Run_Manifest.

### Requirement 8: Reproducibility and Experiment Tracking

**User Story:** As a student researcher submitting work for review, I want every training run to be fully described by a Run_Manifest, so that any result can be reproduced or audited.

#### Acceptance Criteria

1. THE Experiment_Tracker SHALL produce, for every training run, a Run_Manifest that records the run identifier, start timestamp in UTC, end timestamp in UTC, the git commit hash of the training code, the resolved configuration files, the operating system identifier, the Python interpreter version, and the accelerator driver version.
2. THE Experiment_Tracker SHALL record, in the Run_Manifest, the Base_Model identifier and revision, the Quantization_Mode, the LoRA_Config, the Hardware_Profile, the Budget_Profile, and the random seeds set in Requirement 5 criterion 6.
3. THE Experiment_Tracker SHALL record, in the Run_Manifest, the dataset cards from Requirement 3 and a content hash of the final Training_Dataset and Eval_Dataset.
4. THE Experiment_Tracker SHALL record, in the Run_Manifest, every metric required by Requirement 5 and Requirement 6.
5. WHEN a training run completes, THE Experiment_Tracker SHALL persist the Run_Manifest to the documented manifest location alongside the Adapter artifacts within 60 seconds of the final training step.
6. WHEN a training run halts before completion, THE Experiment_Tracker SHALL persist a Run_Manifest annotated with the halt reason and the last completed training step to the documented manifest location within 60 seconds of the halt event.
7. IF a Run_Manifest persistence operation fails, THEN THE Experiment_Tracker SHALL retry the persistence operation up to three times and SHALL surface an error identifying the manifest path and the failure cause if all retries fail.
8. WHERE the working tree contains uncommitted changes at run start, THE Experiment_Tracker SHALL record in the Run_Manifest the parent commit hash, a dirty flag set to true, and a content hash computed over the uncommitted changes.
9. THE Experiment_Tracker SHALL pin, in the Run_Manifest, the resolved version of every package listed in the Training_Pipeline dependency manifest, including the deep learning framework, the LoRA library, the tokenizer library, and the evaluation library.
10. WHEN two Run_Manifests share identical resolved configuration, dataset content hashes, code commit hash with dirty flag false, dependency versions, Hardware_Profile, and random seeds, THE Training_Pipeline SHALL reproduce evaluation scores from Requirement 6 such that the absolute per-metric difference is at most 0.001, and the tolerance value SHALL be recorded in the Run_Manifest.
11. WHEN the Experiment_Tracker is invoked to compare two Run_Manifests, THE Experiment_Tracker SHALL emit a diff that lists configuration changes, dataset content-hash changes, and metric changes whose absolute difference exceeds the tolerance recorded in criterion 10, and IF the two Run_Manifests are byte-identical THEN the emitted diff SHALL be empty.
