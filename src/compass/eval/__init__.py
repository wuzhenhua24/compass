"""Evaluation engine module.

.. deprecated:: 0.2.0
    This module is deprecated. Use :mod:`compass.graders` instead.

    The new grader system provides:
    - Transcript/Outcome separation for agent evaluation
    - Three-tier grading: Code, Model, Human
    - GraderScope: OUTCOME, TRANSCRIPT, BOTH

    Migration guide:
    - Evaluator -> Grader (or CodeGrader/ModelGrader/HumanGrader)
    - EvalContext -> GradeContext
    - get_evaluator -> get_grader
    - list_evaluators -> list_graders
"""

import warnings

warnings.warn(
    "compass.eval is deprecated. Use compass.graders instead. "
    "See compass.graders for the new Transcript/Outcome grading system.",
    DeprecationWarning,
    stacklevel=2,
)

from compass.eval.base import Evaluator, EvalContext
from compass.eval.registry import register_evaluator, get_evaluator, list_evaluators
from compass.eval.aggregator import Aggregator, AggregationStrategy

# Import built-in evaluators to trigger registration
from compass.eval import semantic  # noqa: F401
from compass.eval import aesthetic  # noqa: F401
from compass.eval import vlm_judge  # noqa: F401
from compass.eval import safety  # noqa: F401
from compass.eval import technical  # noqa: F401

__all__ = [
    "Evaluator",
    "EvalContext",
    "register_evaluator",
    "get_evaluator",
    "list_evaluators",
    "Aggregator",
    "AggregationStrategy",
]
