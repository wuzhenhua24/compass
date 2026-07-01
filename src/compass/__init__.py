"""Compass - Agent QA Framework."""

from compass.core.result import EvalResult, CaseResult
from compass.core.scenario import Scenario
from compass.core.metrics import TrialMetrics
from compass.core.trial import TaskResult, TrialResult

__version__ = "0.1.0"
__all__ = [
    "Scenario",
    "EvalResult",
    "CaseResult",
    "TrialMetrics",
    "TaskResult",
    "TrialResult",
]
