"""Compass - Agent QA Framework."""

from compass.core.result import EvalResult, CaseResult
from compass.core.runner import Compass
from compass.core.scenario import Scenario
from compass.core.metrics import TrialMetrics
from compass.core.trial import TaskResult, TrialResult

__version__ = "0.1.0"
__all__ = [
    "Compass",
    "Scenario",
    "EvalResult",
    "CaseResult",
    "TrialMetrics",
    "TaskResult",
    "TrialResult",
]
