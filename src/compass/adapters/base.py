"""Base adapter class for agent integration."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from compass.core.artifacts import Artifact
from compass.core.transcript import ToolCall


@dataclass
class AgentInput:
    """Input to an agent."""

    prompt: str
    negative_prompt: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    # Optional context for observability/recording:
    #   - transcript: Transcript
    #   - tool_call_recorder: Callable[[ToolCall], None]
    #   - tool_calls: list[ToolCall]
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentOutput:
    """Output from an agent."""

    image: Image.Image | None = None
    images: list[Image.Image] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_response: Any = None
    error: str | None = None
    artifacts: list[Artifact] = field(default_factory=list)

    @property
    def has_output(self) -> bool:
        """Whether any artifact produced meaningful output."""
        return any(a.has_output for a in self.artifacts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "has_image": self.image is not None,
            "num_images": len(self.images),
            "metadata": self.metadata,
            "error": self.error,
            "artifacts": [a.to_dict() for a in self.artifacts],
        }


class Adapter(ABC):
    """Base class for agent adapters."""

    name: str = "base"

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize adapter with config.

        Args:
            config: Adapter-specific configuration.
        """
        self.config = config or {}

    @abstractmethod
    async def run(self, input: AgentInput) -> AgentOutput:
        """Run the agent with given input.

        Args:
            input: Agent input.

        Returns:
            Agent output with generated image(s).
        """
        pass

    async def health_check(self) -> bool:
        """Check if the agent is available.

        Returns:
            True if agent is healthy and available.
        """
        return True

    def validate_config(self) -> bool:
        """Validate adapter configuration.

        Returns:
            True if config is valid.
        """
        return True

    # ------------------------------------------------------------------
    # ToolCall recording (optional)
    # ------------------------------------------------------------------

    def _record_tool_call(self, agent_input: AgentInput, **kwargs: Any) -> None:
        """Emit a ToolCall record if a recorder or transcript is available."""
        tool_call = ToolCall(**kwargs)

        recorder = agent_input.context.get("tool_call_recorder")
        if callable(recorder):
            recorder(tool_call)
            return

        transcript = agent_input.context.get("transcript")
        if transcript is not None and hasattr(transcript, "add_tool_call"):
            transcript.add_tool_call(
                tool_name=tool_call.tool_name,
                input=tool_call.input,
                output=tool_call.output,
                status=tool_call.status,
                duration_ms=tool_call.duration_ms,
                error=tool_call.error,
                cost=tool_call.cost,
                tokens=tool_call.tokens,
                tool_type=tool_call.tool_type,
                retry_count=tool_call.retry_count,
                trace=tool_call.trace,
                metadata=tool_call.metadata,
                redacted=tool_call.redacted,
            )
            return

        tool_calls = agent_input.context.get("tool_calls")
        if isinstance(tool_calls, list):
            tool_calls.append(tool_call)
