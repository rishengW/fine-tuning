"""Re-export of curated eval prompts.

The canonical prompt set now lives in ``math_lora.evaluation.eval_prompts``
so that the evaluation harness has a single namespace to import from. This
shim is kept so existing imports (``from math_lora.eval_prompts import
EVAL_PROMPTS``) continue to work.
"""

from math_lora.evaluation.eval_prompts import EVAL_PROMPTS, EvalPrompt

__all__ = ["EVAL_PROMPTS", "EvalPrompt"]
