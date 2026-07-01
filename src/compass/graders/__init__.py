"""Grader system with three categories: Code, Model, Human.

Key design: Graders independently access Transcript and Outcome.
See GraderScope for details.
"""

from compass.graders.base import (
    Grader,
    CodeGrader,
    ModelGrader,
    HumanGrader,
    GradeContext,
    GradeResult,
    GraderScope,
    GraderType,
)
from compass.graders.registry import register_grader, get_grader, list_graders

# Import built-in graders to trigger registration
# Common graders
from compass.graders.code.common import structure  # noqa: F401
from compass.graders.code.common import transcript_graders  # noqa: F401
# Coding Agent graders
from compass.graders.code.coding import functional, quality, diff, security, integration  # noqa: F401
# Data Agent graders
from compass.graders.code.data import sql_equivalence  # noqa: F401
from compass.graders.code.data import data_correctness  # noqa: F401
from compass.graders.code.data import query_quality  # noqa: F401
from compass.graders.code.data import reasoning_trace  # noqa: F401
from compass.graders.code.data import self_correction  # noqa: F401
# Image Agent graders
from compass.graders.code.image import assertions, technical  # noqa: F401
# Model graders
from compass.graders.model import semantic, vlm, aesthetic, safety  # noqa: F401
# Human graders
from compass.graders.human import base as _human_base  # noqa: F401
from compass.graders.human import pairwise as _human_pairwise  # noqa: F401

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
