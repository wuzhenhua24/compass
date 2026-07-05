"""Core engine module."""

from compass.core.scenario import (
    Scenario,
    TestCase,
    GraderConfig,
    GraderType,
    AggregationConfig,
    MetricsConfig,
    DefaultsConfig,
    AgentConfig,
    InputConfig,
    # Backwards compatibility
    EvaluatorConfig,
)
from compass.core.result import EvalResult, CaseResult, EvaluatorResult, TestStatus
from compass.core.metrics import (
    TrialMetrics,
    calculate_metrics,
    estimate_pass_all_k,
    estimate_pass_at_k,
)
from compass.core.trial import TrialResult, TaskResult, TrialManager
from compass.core.transcript import (
    Transcript,
    TranscriptRecorder,
    ToolCall,
    StateChange,
    Outcome,
    Environment,
    CostInfo,
    TokenUsage,
    TOOLCALL_PROTOCOL_VERSION,
    # tool_name utilities
    ToolNameInfo,
    parse_tool_name,
    format_tool_name,
    validate_tool_name,
    KNOWN_ADAPTERS,
    KNOWN_TOOL_TYPES,
    KNOWN_STATE_KINDS,
    KNOWN_STATE_OPS,
)
from compass.core.sweep import SweepConfig, expand_sweeps, SweepSummary, group_sweep_results
from compass.core.artifacts import (
    Artifact,
    ImageArtifact,
    CodeArtifact,
    TextArtifact,
    GeneratedFile,
    ExecutionResult,
    register_artifact,
    deserialize_artifact,
)

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
