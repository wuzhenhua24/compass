"""Typed artifact system for agent outputs.

Provides a plugin-based registry of artifact types so that new agent kinds
(Coding Agent, Content Agent, Audio Agent, …) can be supported by defining
a new Artifact subclass without touching Outcome, GradeContext, or any other
core framework code.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Artifact base class
# ---------------------------------------------------------------------------


@dataclass
class Artifact(ABC):
    """Abstract base for all typed artifacts."""

    artifact_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    @abstractmethod
    def has_output(self) -> bool:
        """Whether this artifact contains meaningful output."""
        ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict (must include ``artifact_type``)."""
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> Artifact:
        """Reconstruct from a dict produced by ``to_dict``."""
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_artifact_registry: dict[str, type[Artifact]] = {}


def register_artifact(artifact_type: str) -> Callable[[type[Artifact]], type[Artifact]]:
    """Decorator to register an Artifact subclass.

    Example:
        @register_artifact("audio")
        @dataclass
        class AudioArtifact(Artifact):
            ...
    """

    def decorator(cls: type[Artifact]) -> type[Artifact]:
        cls.artifact_type = artifact_type
        _artifact_registry[artifact_type] = cls
        return cls

    return decorator


def get_artifact_class(artifact_type: str) -> type[Artifact]:
    """Look up an Artifact class by its registered type string.

    Raises:
        KeyError: If the type is not registered.
    """
    if artifact_type not in _artifact_registry:
        raise KeyError(
            f"Artifact type '{artifact_type}' not found. "
            f"Available: {list(_artifact_registry.keys())}"
        )
    return _artifact_registry[artifact_type]


def deserialize_artifact(data: dict[str, Any]) -> Artifact:
    """Reconstruct an Artifact from its dict representation.

    Raises:
        KeyError: If the artifact type is unknown.
    """
    artifact_type = data.get("artifact_type", "")
    cls = get_artifact_class(artifact_type)
    return cls.from_dict(data)


# ---------------------------------------------------------------------------
# Helper dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GeneratedFile:
    """A single file produced by a coding agent."""

    path: str = ""
    content: str = ""
    language: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "content": self.content, "language": self.language}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GeneratedFile:
        return cls(
            path=data.get("path", ""),
            content=data.get("content", ""),
            language=data.get("language", ""),
        )


@dataclass
class ExecutionResult:
    """Result of executing generated code."""

    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionResult:
        return cls(
            exit_code=data.get("exit_code", 0),
            stdout=data.get("stdout", ""),
            stderr=data.get("stderr", ""),
            duration_ms=data.get("duration_ms", 0.0),
        )


# ---------------------------------------------------------------------------
# Concrete artifact types
# ---------------------------------------------------------------------------


@register_artifact("image")
@dataclass
class ImageArtifact(Artifact):
    """Artifact wrapping an image output (mirrors legacy Outcome image fields)."""

    artifact_type: str = "image"

    image: Any = field(default=None, repr=False)  # PIL Image at runtime
    image_path: str | None = None
    image_hash: str | None = None

    @property
    def has_output(self) -> bool:
        return self.image is not None or self.image_path is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "image_path": self.image_path,
            "image_hash": self.image_hash,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImageArtifact:
        return cls(
            image_path=data.get("image_path"),
            image_hash=data.get("image_hash"),
            metadata=data.get("metadata", {}),
        )


@register_artifact("code")
@dataclass
class CodeArtifact(Artifact):
    """Artifact wrapping code generation output."""

    artifact_type: str = "code"

    files: list[GeneratedFile] = field(default_factory=list)
    language: str = ""
    diff: str = ""
    execution: ExecutionResult | None = None

    @property
    def has_output(self) -> bool:
        return len(self.files) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "files": [f.to_dict() for f in self.files],
            "language": self.language,
            "diff": self.diff,
            "execution": self.execution.to_dict() if self.execution else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CodeArtifact:
        files = [GeneratedFile.from_dict(f) for f in data.get("files", [])]
        execution_data = data.get("execution")
        execution = ExecutionResult.from_dict(execution_data) if execution_data else None
        return cls(
            files=files,
            language=data.get("language", ""),
            diff=data.get("diff", ""),
            execution=execution,
            metadata=data.get("metadata", {}),
        )


@register_artifact("text")
@dataclass
class TextArtifact(Artifact):
    """Artifact wrapping textual content (articles, reports, etc.)."""

    artifact_type: str = "text"

    content: str = ""
    format: str = "plain"
    word_count: int = 0

    @property
    def has_output(self) -> bool:
        return len(self.content) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "content": self.content,
            "format": self.format,
            "word_count": self.word_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TextArtifact:
        return cls(
            content=data.get("content", ""),
            format=data.get("format", "plain"),
            word_count=data.get("word_count", 0),
            metadata=data.get("metadata", {}),
        )
