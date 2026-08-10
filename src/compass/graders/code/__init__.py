"""Code-based graders for deterministic evaluation.

Organized by agent type:
- common/  - Shared graders (structure validation, transcript analysis)
- coding/  - Coding Agent graders (tests, linting, diffs)
- data/    - Data Agent graders (SQL equivalence, data correctness)
- image/   - Image Agent graders (assertions, technical quality)
"""

# Common graders
# Coding Agent graders
from compass.graders.code.coding import (
    DiffAccuracyGrader,
    DiffSizeGrader,
    ExitCodeGrader,
    IntegrationGrader,
    LintGrader,
    SecurityScanGrader,
    TestRunnerGrader,
    TypeCheckGrader,
)
from compass.graders.code.common import (
    CostBudgetGrader,
    EfficiencyGrader,
    JsonSchemaGrader,
    LatencyBudgetGrader,
    SqlSyntaxGrader,
    StructureCheckGrader,
    ToolUsageGrader,
)

# Data Agent graders
from compass.graders.code.data import (
    DataCorrectnessGrader,
    QueryQualityGrader,
    ReasoningTraceGrader,
    SelfCorrectionGrader,
    SqlEquivalenceGrader,
)

# Image Agent graders
from compass.graders.code.image import (
    ImageAssertionGrader,
    TechnicalQualityGrader,
)

__all__ = [
    # Common graders
    "CostBudgetGrader",
    "EfficiencyGrader",
    "JsonSchemaGrader",
    "LatencyBudgetGrader",
    "SqlSyntaxGrader",
    "StructureCheckGrader",
    "ToolUsageGrader",
    # Coding Agent graders
    "DiffAccuracyGrader",
    "DiffSizeGrader",
    "ExitCodeGrader",
    "IntegrationGrader",
    "LintGrader",
    "SecurityScanGrader",
    "TestRunnerGrader",
    "TypeCheckGrader",
    # Data Agent graders
    "DataCorrectnessGrader",
    "QueryQualityGrader",
    "ReasoningTraceGrader",
    "SelfCorrectionGrader",
    "SqlEquivalenceGrader",
    # Image Agent graders
    "ImageAssertionGrader",
    "TechnicalQualityGrader",
]
