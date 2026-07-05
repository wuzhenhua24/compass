"""Transcript recording for complete execution history."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Type, TYPE_CHECKING

from compass.core.artifacts import (
    Artifact,
    ImageArtifact,
    deserialize_artifact,
)

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)

# Current protocol version
# 1.2: promoted turn_index / agent_name to first-class ToolCall fields.
# 1.3: added state_delta (list[StateChange]) to ToolCall — what the call
#      changed in the environment, as opposed to what it returned.
TOOLCALL_PROTOCOL_VERSION = "1.3"


def _normalize_error(error: dict[str, Any] | str | None) -> dict[str, Any] | None:
    """Normalize error to dict format or None."""
    if error is None:
        return None
    if isinstance(error, str):
        return {"message": error}
    return error


def _event_type_order(event_type: str) -> int:
    """Return sort order for event types (used for stable sorting within same timestamp)."""
    order = {
        "transcript.started": 0,
        "tool_call.started": 1,
        "tool_call.completed": 2,
        "reasoning.step": 3,
        "outcome.set": 4,
        "transcript.completed": 5,
    }
    return order.get(event_type, 99)


# ============================================================================
# tool_name naming convention: {adapter}.{action} or {adapter}.{resource}.{action}
# ============================================================================

# Known adapters for validation
KNOWN_ADAPTERS = frozenset({
    # LLM providers
    "openai", "anthropic", "google", "cohere", "mistral", "azure",
    # Image generation
    "image", "stablediffusion", "midjourney", "dalle",
    # Code execution
    "sandbox", "docker", "e2b",
    # Tools/Agents
    "browser", "mcp", "langchain", "autogpt",
    # Generic
    "custom",
})

# Known tool types
KNOWN_TOOL_TYPES = frozenset({
    "llm", "image", "code", "browser", "mcp", "api", "file", "search",
})


@dataclass
class ToolNameInfo:
    """Parsed tool name components.

    Naming convention: {adapter}.{action} or {adapter}.{resource}.{action}

    Examples:
        - "openai.chat.completion" -> adapter="openai", resource="chat", action="completion"
        - "sandbox.exec" -> adapter="sandbox", resource=None, action="exec"
        - "image.generate" -> adapter="image", resource=None, action="generate"
    """

    adapter: str
    action: str
    resource: str | None = None

    @property
    def full_name(self) -> str:
        """Return the full tool name."""
        if self.resource:
            return f"{self.adapter}.{self.resource}.{self.action}"
        return f"{self.adapter}.{self.action}"

    def matches_adapter(self, adapter: str) -> bool:
        """Check if this tool belongs to the given adapter."""
        return self.adapter.lower() == adapter.lower()

    def matches_pattern(self, pattern: str) -> bool:
        """Check if tool name matches a pattern (supports * wildcard).

        Examples:
            - "openai.*" matches "openai.chat.completion"
            - "*.exec" matches "sandbox.exec"
            - "*" matches everything
        """
        if pattern == "*":
            return True
        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            return self.full_name.startswith(prefix + ".")
        if pattern.startswith("*."):
            suffix = pattern[1:]  # includes the dot
            return self.full_name.endswith(suffix)
        return self.full_name == pattern


def parse_tool_name(tool_name: str) -> ToolNameInfo:
    """Parse a tool name into its components.

    Args:
        tool_name: Tool name in format "{adapter}.{action}" or "{adapter}.{resource}.{action}"

    Returns:
        ToolNameInfo with parsed components.

    Examples:
        >>> parse_tool_name("openai.chat.completion")
        ToolNameInfo(adapter='openai', action='completion', resource='chat')
        >>> parse_tool_name("sandbox.exec")
        ToolNameInfo(adapter='sandbox', action='exec', resource=None)
    """
    parts = tool_name.split(".")
    if len(parts) == 2:
        return ToolNameInfo(adapter=parts[0], action=parts[1])
    elif len(parts) >= 3:
        return ToolNameInfo(adapter=parts[0], resource=parts[1], action=".".join(parts[2:]))
    else:
        # Single part - treat as action with "custom" adapter
        return ToolNameInfo(adapter="custom", action=tool_name)


def format_tool_name(adapter: str, action: str, resource: str | None = None) -> str:
    """Format a tool name from components.

    Args:
        adapter: Adapter/provider name (e.g., "openai", "sandbox")
        action: Action name (e.g., "completion", "exec")
        resource: Optional resource name (e.g., "chat", "messages")

    Returns:
        Formatted tool name string.

    Examples:
        >>> format_tool_name("openai", "completion", "chat")
        'openai.chat.completion'
        >>> format_tool_name("sandbox", "exec")
        'sandbox.exec'
    """
    if resource:
        return f"{adapter}.{resource}.{action}"
    return f"{adapter}.{action}"


def validate_tool_name(tool_name: str, strict: bool = False) -> list[str]:
    """Validate a tool name against naming conventions.

    Args:
        tool_name: Tool name to validate.
        strict: If True, require known adapter names.

    Returns:
        List of warning messages (empty if valid).

    Examples:
        >>> validate_tool_name("openai.chat.completion")
        []
        >>> validate_tool_name("foo")
        ["Tool name 'foo' should use format '{adapter}.{action}'"]
    """
    warnings = []

    if not tool_name:
        warnings.append("Tool name is empty")
        return warnings

    parts = tool_name.split(".")
    if len(parts) < 2:
        warnings.append(f"Tool name '{tool_name}' should use format '{{adapter}}.{{action}}'")
        return warnings

    adapter = parts[0].lower()

    if strict and adapter not in KNOWN_ADAPTERS:
        warnings.append(
            f"Unknown adapter '{adapter}'. Known adapters: {', '.join(sorted(KNOWN_ADAPTERS))}"
        )

    # Check for common naming issues
    if any(p == "" for p in parts):
        warnings.append(f"Tool name '{tool_name}' contains empty segments")

    if any(" " in p for p in parts):
        warnings.append(f"Tool name '{tool_name}' should not contain spaces")

    return warnings


@dataclass
class CostInfo:
    """Standardized cost information for a tool call."""

    total_usd: float = 0.0
    input_cost_usd: float | None = None
    output_cost_usd: float | None = None
    currency: str = "USD"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_usd": self.total_usd,
            "input_cost_usd": self.input_cost_usd,
            "output_cost_usd": self.output_cost_usd,
            "currency": self.currency,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CostInfo":
        return cls(
            total_usd=data.get("total_usd", 0.0),
            input_cost_usd=data.get("input_cost_usd"),
            output_cost_usd=data.get("output_cost_usd"),
            currency=data.get("currency", "USD"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TokenUsage:
    """Standardized token usage for a tool call."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Auto-calculate total if not set but input/output provided
        if self.total_tokens == 0 and (self.input_tokens or self.output_tokens):
            self.total_tokens = self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TokenUsage":
        return cls(
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            metadata=data.get("metadata", {}),
        )


# Suggested vocabularies for StateChange.kind / StateChange.op.
# Open sets — any string is accepted; these exist so independent producers
# converge on the same words for the same things.
KNOWN_STATE_KINDS = frozenset({
    "file", "db", "env", "git", "http", "browser", "memory", "process",
})
KNOWN_STATE_OPS = frozenset({"create", "update", "delete", "move", "execute"})


@dataclass
class StateChange:
    """A single observed change to the environment (ToolCall Protocol v1.3).

    Captures the *state delta* a tool call produced — what the action changed
    in the world, as opposed to what the tool returned in its output. Final
    output and state delta can disagree ("scheduled the meeting" vs. a
    duplicate invite in the calendar), and stateful tasks can only be graded
    on the latter.

    Compass defines the slot and vocabulary; *capturing* deltas (snapshotting,
    diffing) is adapter / harness / importer code. An empty ``state_delta``
    therefore means "nothing recorded", not "nothing changed".

    Fields are domain-agnostic; anything domain-specific goes in ``metadata``.
    """

    kind: str  # Resource type: "file" | "db" | "env" | "git" | ... (open vocabulary)
    op: str  # Operation: "create" | "update" | "delete" | "move" | "execute" | ... (open)
    target: str = ""  # What changed: path, table/row id, env key, URL, git ref
    before: str | None = None  # Hash / pointer / small repr of prior state
    after: str | None = None  # Hash / pointer / small repr of new state
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "kind": self.kind,
            "op": self.op,
            "target": self.target,
            "before": self.before,
            "after": self.after,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StateChange":
        """Create from dictionary."""
        return cls(
            kind=data.get("kind", ""),
            op=data.get("op", ""),
            target=data.get("target", ""),
            before=data.get("before"),
            after=data.get("after"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ToolCall:
    """Record of a single tool call (ToolCall Protocol v1).

    Core fields (protocol):
      - call_id, tool_name, input, output, status, duration_ms, timestamp
      - error, cost, tokens, tool_type, retry_count, trace, metadata, redacted

    Backwards compatibility:
      - `tool`, `args`, `result` are provided as properties.
    """

    # Protocol fields
    tool_name: str
    input: dict[str, Any] = field(default_factory=dict)
    output: Any = None
    status: str = "ok"  # ok | error | blocked
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    # Optional protocol fields
    call_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    error: dict[str, Any] | None = None  # Normalized in __post_init__
    cost: CostInfo | None = None
    tokens: TokenUsage | None = None
    tool_type: str | None = None
    retry_count: int = 0
    trace: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    redacted: bool = False

    # Multi-agent / multi-turn context (first-class; populated by adapters and
    # by the trace importers in ``compass.integrations``).
    turn_index: int | None = None
    agent_name: str | None = None

    # State delta (first-class since protocol 1.3): what this call changed in
    # the environment. Empty means "nothing recorded", not "nothing changed".
    state_delta: list[StateChange] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Normalize error to dict | None
        self.error = _normalize_error(self.error)  # type: ignore[assignment]

        # Auto-convert dict entries to StateChange
        self.state_delta = [
            sc if isinstance(sc, StateChange) else StateChange.from_dict(sc)
            for sc in self.state_delta
        ]

        # Status correction
        if self.status == "ok" and self.error:
            self.status = "error"

        # Auto-convert dict to CostInfo
        if isinstance(self.cost, dict):
            self.cost = CostInfo.from_dict(self.cost)

        # Auto-convert dict to TokenUsage
        if isinstance(self.tokens, dict):
            self.tokens = TokenUsage.from_dict(self.tokens)

    # ------------------------------------------------------------------
    # Backwards-compatible aliases
    # ------------------------------------------------------------------

    @property
    def tool(self) -> str:
        return self.tool_name

    @tool.setter
    def tool(self, value: str) -> None:
        self.tool_name = value

    @property
    def args(self) -> dict[str, Any]:
        return self.input

    @args.setter
    def args(self, value: dict[str, Any]) -> None:
        self.input = value

    @property
    def result(self) -> Any:
        return self.output

    @result.setter
    def result(self, value: Any) -> None:
        self.output = value

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (protocol + legacy aliases)."""
        return {
            # Protocol fields
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "input": self.input,
            "output": self.output,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
            "error": self.error,  # Already normalized in __post_init__
            "cost": self.cost.to_dict() if self.cost else None,
            "tokens": self.tokens.to_dict() if self.tokens else None,
            "tool_type": self.tool_type,
            "retry_count": self.retry_count,
            "trace": self.trace,
            "metadata": self.metadata,
            "redacted": self.redacted,
            "turn_index": self.turn_index,
            "agent_name": self.agent_name,
            "state_delta": [sc.to_dict() for sc in self.state_delta],
            # Legacy aliases for backward compatibility
            "tool": self.tool_name,
            "args": self.input,
            "result": self.output,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolCall":
        """Reconstruct ToolCall from dict (handles both protocol and legacy fields)."""
        tool_name = data.get("tool_name") or data.get("tool", "")
        input_data = data.get("input") or data.get("args", {})
        output_data = data.get("output") if "output" in data else data.get("result")
        error_data = data.get("error")
        status = data.get("status")
        if status is None:
            status = "error" if error_data else "ok"

        # call_id: use existing or generate new
        call_id = data.get("call_id") or uuid.uuid4().hex[:12]

        # cost: CostInfo or None
        cost_data = data.get("cost")
        cost = CostInfo.from_dict(cost_data) if cost_data else None

        # tokens: TokenUsage or None
        tokens_data = data.get("tokens")
        tokens = TokenUsage.from_dict(tokens_data) if tokens_data else None

        # turn_index / agent_name: first-class since protocol 1.2, but fall back
        # to the legacy metadata location so older transcripts still populate them.
        meta = data.get("metadata", {})
        turn_index = data.get("turn_index", meta.get("turn_index"))
        agent_name = data.get("agent_name", meta.get("agent_name"))

        return cls(
            tool_name=tool_name,
            input=input_data,
            output=output_data,
            status=status,
            duration_ms=data.get("duration_ms", 0),
            timestamp=data.get("timestamp", time.time()),
            call_id=call_id,
            error=error_data,
            cost=cost,
            tokens=tokens,
            tool_type=data.get("tool_type"),
            retry_count=data.get("retry_count", 0),
            trace=data.get("trace"),
            metadata=meta,
            redacted=data.get("redacted", False),
            turn_index=turn_index,
            agent_name=agent_name,
            state_delta=[
                StateChange.from_dict(sc) for sc in data.get("state_delta") or []
            ],
        )


@dataclass
class Outcome:
    """Final outcome state of a trial.

    Represents the end result of an agent execution, separate from
    the execution trace. Graders can independently access this to
    evaluate "what was produced" without caring about "how".
    """

    # For image generation (legacy fields, kept for backward compatibility)
    image: Image.Image | None = field(default=None, repr=False)
    image_path: str | None = None
    image_hash: str | None = None

    # Whether the agent was blocked (e.g., by safety filter)
    blocked: bool = False
    blocked_reason: str = ""

    # Generic output
    output_data: dict[str, Any] = field(default_factory=dict)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Typed artifacts (new extensible system)
    artifacts: list[Artifact] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def has_image(self) -> bool:
        """Whether an image was produced (checks legacy fields + ImageArtifact)."""
        if self.image is not None:
            return True
        return any(isinstance(a, ImageArtifact) for a in self.artifacts)

    @property
    def has_output(self) -> bool:
        """Whether any artifact produced meaningful output (type-agnostic)."""
        return any(a.has_output for a in self.artifacts)

    # ------------------------------------------------------------------
    # Artifact accessors
    # ------------------------------------------------------------------

    def get_artifact(self, artifact_type: Type[Artifact]) -> Artifact | None:
        """Return the first artifact matching the given type, or None."""
        for a in self.artifacts:
            if isinstance(a, artifact_type):
                return a
        return None

    def get_artifacts(self, artifact_type: Type[Artifact]) -> list[Artifact]:
        """Return all artifacts matching the given type."""
        return [a for a in self.artifacts if isinstance(a, artifact_type)]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (image excluded, not serializable)."""
        return {
            "image_path": self.image_path,
            "image_hash": self.image_hash,
            "blocked": self.blocked,
            "blocked_reason": self.blocked_reason,
            "output_data": self.output_data,
            "metadata": self.metadata,
            "artifacts": [a.to_dict() for a in self.artifacts],
        }


@dataclass
class Environment:
    """Environment snapshot at time of execution."""

    model_version: str = ""
    adapter_version: str = ""
    random_seed: int | None = None

    # Resource usage
    gpu_memory_mb: float | None = None
    cpu_percent: float | None = None

    # Custom attributes
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_version": self.model_version,
            "adapter_version": self.adapter_version,
            "random_seed": self.random_seed,
            "gpu_memory_mb": self.gpu_memory_mb,
            "cpu_percent": self.cpu_percent,
            "attributes": self.attributes,
        }


@dataclass
class Transcript:
    """Complete record of a trial execution."""

    task_id: str
    trial_id: str

    # Protocol version
    protocol_version: str = TOOLCALL_PROTOCOL_VERSION

    # Input
    input_prompt: str = ""
    input_params: dict[str, Any] = field(default_factory=dict)

    # Free-form trace-level metadata (e.g. handoffs/guardrails from imported
    # traces, or anything an adapter/importer wants to attach to the run).
    metadata: dict[str, Any] = field(default_factory=dict)

    # Execution trace
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning_steps: list[str] = field(default_factory=list)

    # Output
    outcome: Outcome = field(default_factory=Outcome)

    # Environment
    environment: Environment = field(default_factory=Environment)

    # Timing
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    total_duration_ms: float = 0.0

    # Grading
    grader_results: list[dict[str, Any]] = field(default_factory=list)
    final_score: float = 0.0
    final_passed: bool = False

    # ------------------------------------------------------------------
    # Aggregation methods
    # ------------------------------------------------------------------

    def sum_cost(self) -> CostInfo:
        """Sum cost across all tool calls.

        Returns a CostInfo with aggregated totals. Only sums from tool calls
        that have cost information; others are skipped.
        """
        total_usd = 0.0
        input_cost_usd = 0.0
        output_cost_usd = 0.0
        has_input = False
        has_output = False

        for tc in self.tool_calls:
            if tc.cost is not None:
                total_usd += tc.cost.total_usd
                if tc.cost.input_cost_usd is not None:
                    input_cost_usd += tc.cost.input_cost_usd
                    has_input = True
                if tc.cost.output_cost_usd is not None:
                    output_cost_usd += tc.cost.output_cost_usd
                    has_output = True

        return CostInfo(
            total_usd=total_usd,
            input_cost_usd=input_cost_usd if has_input else None,
            output_cost_usd=output_cost_usd if has_output else None,
        )

    def sum_tokens(self) -> TokenUsage:
        """Sum token usage across all tool calls.

        Returns a TokenUsage with aggregated totals. Only sums from tool calls
        that have token information; others are skipped.
        """
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0

        for tc in self.tool_calls:
            if tc.tokens is not None:
                input_tokens += tc.tokens.input_tokens
                output_tokens += tc.tokens.output_tokens
                total_tokens += tc.tokens.total_tokens

        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    def sum_duration(self) -> float:
        """Sum duration_ms across all tool calls.

        Returns total duration in milliseconds.
        """
        return sum(tc.duration_ms for tc in self.tool_calls)

    def add_tool_call(
        self,
        tool_name: str | None = None,
        input: dict[str, Any] | None = None,
        output: Any = None,
        *,
        status: str | None = None,
        duration_ms: float = 0.0,
        error: dict[str, Any] | str | None = None,
        cost: dict[str, Any] | None = None,
        tokens: dict[str, Any] | None = None,
        tool_type: str | None = None,
        retry_count: int = 0,
        trace: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        redacted: bool = False,
        turn_index: int | None = None,
        agent_name: str | None = None,
        state_delta: list[StateChange | dict[str, Any]] | None = None,
        # Legacy aliases
        tool: str | None = None,
        args: dict[str, Any] | None = None,
        result: Any = None,
    ) -> None:
        """Add a tool call to the transcript (ToolCall Protocol v1)."""
        resolved_tool = tool_name or tool or ""
        resolved_input = input if input is not None else (args or {})
        resolved_output = output if output is not None else result
        resolved_status = status or ("error" if error else "ok")

        self.tool_calls.append(
            ToolCall(
                tool_name=resolved_tool,
                input=resolved_input,
                output=resolved_output,
                status=resolved_status,
                duration_ms=duration_ms,
                error=error,
                cost=cost,
                tokens=tokens,
                tool_type=tool_type,
                retry_count=retry_count,
                trace=trace,
                metadata=metadata or {},
                redacted=redacted,
                turn_index=turn_index,
                agent_name=agent_name,
                state_delta=list(state_delta) if state_delta else [],  # type: ignore[arg-type]
            )
        )

    def add_reasoning_step(self, step: str) -> None:
        """Add a reasoning step to the transcript."""
        self.reasoning_steps.append(step)

    def all_state_changes(self) -> list[StateChange]:
        """Flatten state deltas across all tool calls (chronological order)."""
        return [sc for tc in self.tool_calls for sc in tc.state_delta]

    def set_outcome(
        self,
        image: Image.Image | None = None,
        image_path: str | None = None,
        image_bytes: bytes | None = None,
        blocked: bool = False,
        blocked_reason: str = "",
        output_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        artifacts: list[Artifact] | None = None,
    ) -> None:
        """Set the outcome of this trial.

        If legacy image parameters are provided and no ImageArtifact is
        present in *artifacts*, an ImageArtifact is automatically created
        for backward compatibility.
        """
        image_hash = hashlib.sha256(image_bytes).hexdigest() if image_bytes else None

        resolved_artifacts: list[Artifact] = list(artifacts) if artifacts else []

        # Auto-create ImageArtifact from legacy params when not already present
        if (image is not None or image_path is not None) and not any(
            isinstance(a, ImageArtifact) for a in resolved_artifacts
        ):
            resolved_artifacts.append(
                ImageArtifact(
                    image=image,
                    image_path=image_path,
                    image_hash=image_hash,
                )
            )

        self.outcome = Outcome(
            image=image,
            image_path=image_path,
            image_hash=image_hash,
            blocked=blocked,
            blocked_reason=blocked_reason,
            output_data=output_data or {},
            metadata=metadata or {},
            artifacts=resolved_artifacts,
        )

    def finalize(
        self,
        grader_results: list[dict[str, Any]],
        final_score: float,
        final_passed: bool,
    ) -> None:
        """Finalize the transcript with grading results."""
        self.end_time = datetime.now()
        self.total_duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.grader_results = grader_results
        self.final_score = final_score
        self.final_passed = final_passed

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "protocol_version": self.protocol_version,
            "task_id": self.task_id,
            "trial_id": self.trial_id,
            "input": {
                "prompt": self.input_prompt,
                "params": self.input_params,
            },
            "metadata": self.metadata,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "reasoning_steps": self.reasoning_steps,
            "outcome": self.outcome.to_dict(),
            "environment": self.environment.to_dict(),
            "timing": {
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "total_duration_ms": self.total_duration_ms,
            },
            "grading": {
                "results": self.grader_results,
                "final_score": self.final_score,
                "final_passed": self.final_passed,
            },
        }

    # ------------------------------------------------------------------
    # JSONL Export/Import (Event Stream Format)
    # ------------------------------------------------------------------

    def to_jsonl(self) -> str:
        """Export transcript as JSONL event stream.

        Each line is a JSON object representing an event in chronological order.
        Event types:
          - transcript.started: Execution began
          - tool_call.started: Tool call initiated
          - tool_call.completed: Tool call finished
          - reasoning.step: Reasoning step recorded
          - outcome.set: Outcome was set
          - transcript.completed: Execution finished

        Returns:
            JSONL string with one event per line.

        Example output:
            {"type":"transcript.started","ts":1234567890.0,"task_id":"test1",...}
            {"type":"tool_call.started","ts":1234567890.1,"call_id":"abc123",...}
            {"type":"tool_call.completed","ts":1234567890.5,"call_id":"abc123",...}
            {"type":"transcript.completed","ts":1234567891.0,"final_passed":true,...}
        """
        events: list[dict[str, Any]] = []
        start_ts = self.start_time.timestamp()

        # Event 1: transcript.started
        events.append({
            "type": "transcript.started",
            "ts": start_ts,
            "task_id": self.task_id,
            "trial_id": self.trial_id,
            "protocol_version": self.protocol_version,
            "input": {
                "prompt": self.input_prompt,
                "params": self.input_params,
            },
        })

        # Events 2-N: tool_call.started + tool_call.completed pairs
        for tc in self.tool_calls:
            # tool_call.started
            started_ts = tc.timestamp
            started_event: dict[str, Any] = {
                "type": "tool_call.started",
                "ts": started_ts,
                "call_id": tc.call_id,
                "tool_name": tc.tool_name,
                "tool_type": tc.tool_type,
                "input": tc.input,
            }
            if tc.turn_index is not None:
                started_event["turn_index"] = tc.turn_index
            if tc.agent_name is not None:
                started_event["agent_name"] = tc.agent_name
            events.append(started_event)

            # tool_call.completed
            completed_ts = started_ts + (tc.duration_ms / 1000.0)
            completed_event: dict[str, Any] = {
                "type": "tool_call.completed",
                "ts": completed_ts,
                "call_id": tc.call_id,
                "tool_name": tc.tool_name,
                "status": tc.status,
                "duration_ms": tc.duration_ms,
            }

            # Include output (may be large, but useful for checks)
            if tc.output is not None:
                completed_event["output"] = tc.output

            # Include error if present
            if tc.error:
                completed_event["error"] = tc.error

            # Include cost if present
            if tc.cost:
                completed_event["cost"] = tc.cost.to_dict()

            # Include tokens if present
            if tc.tokens:
                completed_event["tokens"] = tc.tokens.to_dict()

            # Include retry_count if non-zero
            if tc.retry_count > 0:
                completed_event["retry_count"] = tc.retry_count

            # Include state delta if recorded
            if tc.state_delta:
                completed_event["state_delta"] = [sc.to_dict() for sc in tc.state_delta]

            events.append(completed_event)

        # Events: reasoning.step (interleaved based on order)
        # Note: reasoning steps don't have timestamps, so we place them
        # after all tool calls but before outcome
        for i, step in enumerate(self.reasoning_steps):
            # Estimate timestamp based on position
            if self.end_time:
                step_ts = start_ts + (self.total_duration_ms / 1000.0) * 0.9
            else:
                step_ts = start_ts
            events.append({
                "type": "reasoning.step",
                "ts": step_ts,
                "index": i,
                "step": step,
            })

        # Event: outcome.set
        if self.outcome:
            outcome_ts = self.end_time.timestamp() if self.end_time else start_ts
            outcome_event: dict[str, Any] = {
                "type": "outcome.set",
                "ts": outcome_ts,
                "blocked": self.outcome.blocked,
            }
            if self.outcome.blocked_reason:
                outcome_event["blocked_reason"] = self.outcome.blocked_reason
            if self.outcome.has_image:
                outcome_event["has_image"] = True
                if self.outcome.image_path:
                    outcome_event["image_path"] = self.outcome.image_path
            if self.outcome.artifacts:
                outcome_event["artifacts"] = [
                    {"type": a.artifact_type, "has_output": a.has_output}
                    for a in self.outcome.artifacts
                ]
            events.append(outcome_event)

        # Final event: transcript.completed
        end_ts = self.end_time.timestamp() if self.end_time else start_ts
        events.append({
            "type": "transcript.completed",
            "ts": end_ts,
            "total_duration_ms": self.total_duration_ms,
            "final_score": self.final_score,
            "final_passed": self.final_passed,
            "tool_call_count": len(self.tool_calls),
            "total_cost_usd": self.sum_cost().total_usd,
            "total_tokens": self.sum_tokens().total_tokens,
        })

        # Sort events by timestamp for chronological order
        events.sort(key=lambda e: (e["ts"], _event_type_order(e["type"])))

        # Convert to JSONL
        lines = [json.dumps(event, ensure_ascii=False, separators=(",", ":")) for event in events]
        return "\n".join(lines)

    def save_jsonl(self, path: str | Path) -> Path:
        """Save transcript as JSONL event stream file.

        Args:
            path: Output file path (typically .jsonl extension).

        Returns:
            Path to the saved file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_jsonl())
            f.write("\n")  # Trailing newline

        return path

    @classmethod
    def from_jsonl(cls, jsonl_content: str) -> "Transcript":
        """Reconstruct Transcript from JSONL event stream.

        Args:
            jsonl_content: JSONL string with one event per line.

        Returns:
            Reconstructed Transcript object.

        Note:
            Some information may be lost in the round-trip as JSONL format
            is optimized for event streaming, not full fidelity storage.
            For full fidelity, use to_dict()/from_dict() or save()/load().
        """
        events = [
            json.loads(line)
            for line in jsonl_content.strip().split("\n")
            if line.strip()
        ]

        if not events:
            raise ValueError("Empty JSONL content")

        # Find transcript.started event
        started = next((e for e in events if e["type"] == "transcript.started"), None)
        if not started:
            raise ValueError("Missing transcript.started event")

        transcript = cls(
            task_id=started["task_id"],
            trial_id=started["trial_id"],
            protocol_version=started.get("protocol_version", TOOLCALL_PROTOCOL_VERSION),
            input_prompt=started.get("input", {}).get("prompt", ""),
            input_params=started.get("input", {}).get("params", {}),
        )
        transcript.start_time = datetime.fromtimestamp(started["ts"])

        # Collect tool_call events (match started with completed by call_id)
        started_events: dict[str, dict] = {}
        for e in events:
            if e["type"] == "tool_call.started":
                started_events[e["call_id"]] = e
            elif e["type"] == "tool_call.completed":
                call_id = e["call_id"]
                start_event = started_events.get(call_id, {})

                # Reconstruct cost and tokens
                cost_data = e.get("cost")
                tokens_data = e.get("tokens")

                transcript.tool_calls.append(ToolCall(
                    call_id=call_id,
                    tool_name=e.get("tool_name", start_event.get("tool_name", "")),
                    tool_type=start_event.get("tool_type"),
                    input=start_event.get("input", {}),
                    output=e.get("output"),
                    status=e.get("status", "ok"),
                    duration_ms=e.get("duration_ms", 0.0),
                    timestamp=start_event.get("ts", e["ts"]),
                    error=e.get("error"),
                    cost=CostInfo.from_dict(cost_data) if cost_data else None,
                    tokens=TokenUsage.from_dict(tokens_data) if tokens_data else None,
                    retry_count=e.get("retry_count", 0),
                    turn_index=start_event.get("turn_index"),
                    agent_name=start_event.get("agent_name"),
                    state_delta=[
                        StateChange.from_dict(sc)
                        for sc in e.get("state_delta") or []
                    ],
                ))

        # Collect reasoning steps
        reasoning_events = sorted(
            [e for e in events if e["type"] == "reasoning.step"],
            key=lambda e: e.get("index", 0)
        )
        transcript.reasoning_steps = [e["step"] for e in reasoning_events]

        # Reconstruct outcome from outcome.set event
        outcome_event = next((e for e in events if e["type"] == "outcome.set"), None)
        if outcome_event:
            transcript.outcome = Outcome(
                blocked=outcome_event.get("blocked", False),
                blocked_reason=outcome_event.get("blocked_reason", ""),
                image_path=outcome_event.get("image_path"),
            )

        # Reconstruct timing and grading from transcript.completed
        completed = next((e for e in events if e["type"] == "transcript.completed"), None)
        if completed:
            transcript.end_time = datetime.fromtimestamp(completed["ts"])
            transcript.total_duration_ms = completed.get("total_duration_ms", 0.0)
            transcript.final_score = completed.get("final_score", 0.0)
            transcript.final_passed = completed.get("final_passed", False)

        return transcript

    @classmethod
    def load_jsonl(cls, path: str | Path) -> "Transcript":
        """Load transcript from JSONL event stream file.

        Args:
            path: Path to JSONL file.

        Returns:
            Reconstructed Transcript object.
        """
        path = Path(path)
        content = path.read_text(encoding="utf-8")
        return cls.from_jsonl(content)

    def save(self, path: str | Path) -> Path:
        """Save transcript to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        return path

    @classmethod
    def load(cls, path: str | Path) -> "Transcript":
        """Load transcript from JSON file."""
        path = Path(path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Check protocol version (warn if newer than current)
        file_version = data.get("protocol_version", "1.0")
        if file_version > TOOLCALL_PROTOCOL_VERSION:
            logger.warning(
                "Transcript file has newer protocol version %s (current: %s). "
                "Some fields may not be recognized.",
                file_version,
                TOOLCALL_PROTOCOL_VERSION,
            )

        transcript = cls(
            task_id=data["task_id"],
            trial_id=data["trial_id"],
            protocol_version=file_version,
            input_prompt=data["input"]["prompt"],
            input_params=data["input"]["params"],
        )
        transcript.metadata = data.get("metadata", {})

        # Restore tool calls
        for tc_data in data.get("tool_calls", []):
            transcript.tool_calls.append(ToolCall.from_dict(tc_data))

        transcript.reasoning_steps = data.get("reasoning_steps", [])

        # Restore outcome
        outcome_data = data.get("outcome", {})

        # Deserialize artifacts (tolerate unknown types gracefully)
        loaded_artifacts: list[Artifact] = []
        for art_data in outcome_data.get("artifacts", []):
            try:
                loaded_artifacts.append(deserialize_artifact(art_data))
            except KeyError:
                logger.warning(
                    "Skipping unknown artifact type: %s",
                    art_data.get("artifact_type"),
                )

        transcript.outcome = Outcome(
            image_path=outcome_data.get("image_path"),
            image_hash=outcome_data.get("image_hash"),
            output_data=outcome_data.get("output_data", {}),
            metadata=outcome_data.get("metadata", {}),
            artifacts=loaded_artifacts,
        )

        # Restore grading
        grading = data.get("grading", {})
        transcript.grader_results = grading.get("results", [])
        transcript.final_score = grading.get("final_score", 0.0)
        transcript.final_passed = grading.get("final_passed", False)

        return transcript


class TranscriptRecorder:
    """Context manager for recording transcripts."""

    def __init__(self, task_id: str, trial_id: str):
        """Initialize recorder."""
        self.transcript = Transcript(task_id=task_id, trial_id=trial_id)
        self._start_time = None

    def __enter__(self) -> Transcript:
        """Start recording."""
        self._start_time = time.time()
        self.transcript.start_time = datetime.now()
        return self.transcript

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop recording."""
        self.transcript.end_time = datetime.now()
        self.transcript.total_duration_ms = (time.time() - self._start_time) * 1000
