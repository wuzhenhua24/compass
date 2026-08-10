"""Grader system with three categories: Code, Model, Human.

Key design: Graders independently access Transcript and Outcome.
See GraderScope for details.
"""

from compass.graders.base import (
    CodeGrader,
    GradeContext,
    Grader,
    GradeResult,
    GraderScope,
    GraderType,
    HumanGrader,
    ModelGrader,
)

# Coding Agent graders
from compass.graders.code.coding import (  # noqa: F401
    diff,
    functional,
    integration,
    quality,
    security,
)

# Import built-in graders to trigger registration
# Common graders
from compass.graders.code.common import (
    structure,  # noqa: F401
    transcript_graders,  # noqa: F401
)

# Data Agent graders
from compass.graders.code.data import (
    data_correctness,  # noqa: F401
    query_quality,  # noqa: F401
    reasoning_trace,  # noqa: F401
    self_correction,  # noqa: F401
    sql_equivalence,  # noqa: F401
)

# Image Agent graders
from compass.graders.code.image import assertions, technical  # noqa: F401

# Human graders
from compass.graders.human import base as _human_base  # noqa: F401
from compass.graders.human import pairwise as _human_pairwise  # noqa: F401

# Model graders
from compass.graders.model import aesthetic, safety, semantic, vlm  # noqa: F401
from compass.graders.registry import get_grader, list_graders, register_grader

__all__ = [
    # Base classes
    "Grader",
    "CodeGrader",
    "ModelGrader",
    "HumanGrader",
    "GradeContext",
    "GradeResult",
    "GraderScope",
    "GraderType",
    # Registry
    "register_grader",
    "get_grader",
    "list_graders",
]
