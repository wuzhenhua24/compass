"""Graders for Data Agent evaluation.

Design principles inspired by OpenAI Kepler:
- Result equivalence over syntax matching
- Golden Sets methodology
- Self-correction capability evaluation

Graders:
- SqlEquivalenceGrader: Compare SQL results (not syntax)
- DataCorrectnessGrader: Verify data accuracy with tolerance
- QueryQualityGrader: Detect SQL anti-patterns
- ReasoningTraceGrader: Evaluate analytical reasoning
- SelfCorrectionGrader: Assess error recovery capability
- SqlSyntaxGrader: Validate SQL syntax (optional ``sqlparse``)
"""

# Re-export utilities for external use
from compass.graders.domains.data._utils import (
    compare_rows,
    compare_values,
    extract_sql_from_text,
    normalize_sql,
)
from compass.graders.domains.data.data_correctness import DataCorrectnessGrader
from compass.graders.domains.data.query_quality import QueryQualityGrader
from compass.graders.domains.data.reasoning_trace import ReasoningTraceGrader
from compass.graders.domains.data.self_correction import SelfCorrectionGrader
from compass.graders.domains.data.sql_equivalence import SqlEquivalenceGrader
from compass.graders.domains.data.sql_syntax import SqlSyntaxGrader

__all__ = [
    # Graders
    "DataCorrectnessGrader",
    "QueryQualityGrader",
    "ReasoningTraceGrader",
    "SelfCorrectionGrader",
    "SqlEquivalenceGrader",
    "SqlSyntaxGrader",
    # Utilities
    "compare_rows",
    "compare_values",
    "extract_sql_from_text",
    "normalize_sql",
]
