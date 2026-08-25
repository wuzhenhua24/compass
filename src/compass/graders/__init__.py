"""Grader system with three categories: Code, Model, Human.

Key design: Graders independently access Transcript and Outcome.
See GraderScope for details.

Importing this package registers the framework's own graders — the
domain-agnostic ones. Domain correctness graders live in
:mod:`compass.graders.domains` and load on first use; see that module for why.
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

# Import built-in graders to trigger registration.
from compass.graders.code.common import (
    danger,  # noqa: F401
    exact,  # noqa: F401
    external,  # noqa: F401
    skill_graders,  # noqa: F401
    structure,  # noqa: F401
    style,  # noqa: F401
    transcript_graders,  # noqa: F401
)
from compass.graders.content import extract_content
from compass.graders.human import base as _human_base  # noqa: F401
from compass.graders.human import pairwise as _human_pairwise  # noqa: F401
from compass.graders.model import (
    groundedness,  # noqa: F401
    rubric,  # noqa: F401
    trajectory,  # noqa: F401
)
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
    # Context helpers
    "extract_content",
    # Registry
    "register_grader",
    "get_grader",
    "list_graders",
]
