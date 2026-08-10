"""Core engine module."""

from compass.core.artifacts import (
    Artifact,
    CodeArtifact,
    ExecutionResult,
    GeneratedFile,
    ImageArtifact,
    TextArtifact,
    deserialize_artifact,
    register_artifact,
)
from compass.core.metrics import (
    TrialMetrics,
    calculate_metrics,
    estimate_pass_all_k,
    estimate_pass_at_k,
)
from compass.core.result import CaseResult, EvalResult, EvaluatorResult, TestStatus
from compass.core.scenario import (
    AgentConfig,
    AggregationConfig,
    DefaultsConfig,
    # Backwards compatibility
    EvaluatorConfig,
    GraderConfig,
    GraderType,
    InputConfig,
    MetricsConfig,
    Scenario,
    TestCase,
)
from compass.core.sweep import SweepConfig, SweepSummary, expand_sweeps, group_sweep_results
from compass.core.transcript import (
    KNOWN_ADAPTERS,
    KNOWN_STATE_KINDS,
    KNOWN_STATE_OPS,
    KNOWN_TOOL_TYPES,
    TOOLCALL_PROTOCOL_VERSION,
    CostInfo,
    Environment,
    Outcome,
    StateChange,
    TokenUsage,
    ToolCall,
    # tool_name utilities
    ToolNameInfo,
    Transcript,
    TranscriptRecorder,
    format_tool_name,
    parse_tool_name,
    validate_tool_name,
)
from compass.core.trial import TaskResult, TrialManager, TrialResult

__all__ = [
    # Scenario
    "Scenario",
    "TestCase",
    "GraderConfig",
    "GraderType",
    "AggregationConfig",
    "MetricsConfig",
    "DefaultsConfig",
    "AgentConfig",
    "InputConfig",
    "EvaluatorConfig",  # Backwards compatibility
    # Results
    "EvalResult",
    "CaseResult",
    "EvaluatorResult",
    "TestStatus",
    # Metrics
    "TrialMetrics",
    "calculate_metrics",
    "estimate_pass_at_k",
    "estimate_pass_all_k",
    # Trials
    "TrialResult",
    "TaskResult",
    "TrialManager",
    # Transcript
    "Transcript",
    "TranscriptRecorder",
    "ToolCall",
    "StateChange",
    "Outcome",
    "Environment",
    "CostInfo",
    "TokenUsage",
    "TOOLCALL_PROTOCOL_VERSION",
    # tool_name utilities
    "ToolNameInfo",
    "parse_tool_name",
    "format_tool_name",
    "validate_tool_name",
    "KNOWN_ADAPTERS",
    "KNOWN_TOOL_TYPES",
    "KNOWN_STATE_KINDS",
    "KNOWN_STATE_OPS",
    # Sweep
    "SweepConfig",
    "expand_sweeps",
    "SweepSummary",
    "group_sweep_results",
    # Artifacts
    "Artifact",
    "ImageArtifact",
    "CodeArtifact",
    "TextArtifact",
    "GeneratedFile",
    "ExecutionResult",
    "register_artifact",
    "deserialize_artifact",
]
