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
"""

# Re-export utilities for external use
from compass.graders.code.data._utils import (
    compare_rows,
    compare_values,
    extract_sql_from_text,
    normalize_sql,
)
from compass.graders.code.data.data_correctness import DataCorrectnessGrader
from compass.graders.code.data.query_quality import QueryQualityGrader
from compass.graders.code.data.reasoning_trace import ReasoningTraceGrader
from compass.graders.code.data.self_correction import SelfCorrectionGrader
from compass.graders.code.data.sql_equivalence import SqlEquivalenceGrader

__all__ = [
    # Graders
    "DataCorrectnessGrader",
    "QueryQualityGrader",
    "ReasoningTraceGrader",
    "SelfCorrectionGrader",
    "SqlEquivalenceGrader",
    # Utilities
    "compare_rows",
    "compare_values",
    "extract_sql_from_text",
    "normalize_sql",
]
