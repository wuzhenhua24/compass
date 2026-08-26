"""Compass - Agent QA Framework."""

from typing import TYPE_CHECKING, Any

from compass.core.metrics import TrialMetrics
from compass.core.result import CaseResult, EvalResult
from compass.core.scenario import Scenario
from compass.core.trial import TaskResult, TrialResult

if TYPE_CHECKING:
    from compass.core.runner import Compass

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


# `Compass` is the only export here that *drives* an agent, so it is the only
# one that needs the adapter registry — 66 modules that grading a recorded run,
# importing a trajectory or publishing a site never touch. PEP 562 keeps
# ``from compass import Compass`` working and defers that bill to whoever
# actually asks for it.
def __getattr__(name: str) -> Any:
    if name != "Compass":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from compass.core.runner import Compass

    globals()["Compass"] = Compass  # __getattr__ runs once per name
    return Compass


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
