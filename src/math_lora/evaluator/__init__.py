"""Evaluator component (Requirement 6).

Runs the baseline evaluation on Base_Model alone before the first training step
and the post-training evaluation on Base_Model + Adapter using identical
decoding parameters, scores GSM8K and MATH using their published protocols,
scores the Custom_Integral_Set under per-problem equivalence rules
(string_equality, numerical_equality_with_tolerance, symbolic_equivalence),
counts parse failures separately from semantic errors, and supports a
quick_eval stratified-subset mode.
"""

__all__: list[str] = []
