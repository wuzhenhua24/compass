"""Base grader classes for the three-tier grading system.

Key design principle (from Anthropic "Demystifying Evals for AI Agents"):
  Graders should be able to independently access the execution transcript
  (how the agent worked) and the final outcome (what was produced).

  - Outcome graders:    evaluate the final result (image quality, semantic match)
  - Transcript graders: evaluate the execution trace (tool usage, cost, latency)
  - Combined graders:   evaluate both together (efficiency vs quality tradeoff)
"""

import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from PIL import Image

from compass.core.artifacts import Artifact, CodeArtifact, ImageArtifact, TextArtifact
from compass.core.transcript import Outcome, Transcript


class GraderType(str, Enum):
    """Type of grader."""

    CODE = "code"
    MODEL = "model"
    HUMAN = "human"


class GraderScope(str, Enum):
    """What a grader evaluates.

    Declares which data the grader needs to perform its evaluation.
    The runner uses this to validate that required data is available
    before calling the grader.
    """

    OUTCOME = "outcome"        # Only needs final result (image, output_data)
    TRANSCRIPT = "transcript"  # Only needs execution trace (tool_calls, timing)
    BOTH = "both"              # Needs both transcript and outcome


@dataclass
class GradeContext:
    """Context for grading - provides independent access to Transcript and Outcome.

    This is the central data object passed to every grader. It cleanly separates:
    - Input:      what was requested (prompt, params)
    - Transcript: how the agent executed (tool calls, reasoning, timing)
    - Outcome:    what was produced (image, output data, blocked status)

    Graders declare their scope (outcome/transcript/both) and access
    the relevant parts of this context independently.
    """

    # === Input ===
    prompt: str = ""
    negative_prompt: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    # === Transcript: execution trace ===
    # The full execution record - tool calls, reasoning, timing, environment
    transcript: Transcript | None = None

    # === Outcome: final result ===
    # The end state - image, output data, blocked status
    outcome: Outcome | None = None

    # === Reference data (for comparison grading) ===
    reference_image: Image.Image | None = None
    reference_artifact: Artifact | None = None
    reference_images: dict[str, Image.Image] = field(default_factory=dict)
    # Golden/expected answer text for QA-style evaluation (symmetric with
    # ``reference_image``). Graders compare the agent's answer against this.
    reference_answer: str = ""

    # === Shared grade workspace ===
    # A directory all graders for one trial share, in declared order. A grader
    # may write intermediates (an extracted SVG, a rendered PNG, a diff) for
    # later graders to consume, turning a flat list of independent checks into
    # a pipeline. Whatever is left in it is kept as scoring evidence — the
    # `details` dict can hold structured data, but not a rendered image.
    workspace: "Path | None" = None

    # === Leak detection ===
    # Unique marker strings embedded in grader/solution files.
    # If any marker appears in the transcript, the trial is tainted.
    leak_markers: list[str] = field(default_factory=list)

    # === Extra metadata ===
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience accessors (so graders don't need deep attribute chains)
    # ------------------------------------------------------------------

    @property
    def image(self) -> Image.Image | None:
        """Shortcut: the generated image from the outcome.

        Checks the legacy ``outcome.image`` field first, then falls back to
        looking for an ``ImageArtifact`` in the artifacts list.
        """
        if self.outcome is not None:
            if self.outcome.image is not None:
                return self.outcome.image
            img_art = self.outcome.get_artifact(ImageArtifact)
            if img_art is not None:
                return img_art.image
        return None

    @property
    def is_blocked(self) -> bool:
        """Shortcut: whether the agent was blocked (e.g., safety filter)."""
        if self.outcome is not None:
            return self.outcome.blocked
        return False

    @property
    def answer(self) -> str:
        """Shortcut: the agent's final text answer.

        Two sources, in order:

        1. ``outcome.output_data['final_output']`` — the explicit contract, set
           by every trace importer.
        2. the first ``TextArtifact`` — the natural way an *adapter* returns
           text, since ``AgentOutput.to_dict()`` has no text field of its own.

        Without the second source, ``answer`` was empty for every
        adapter-driven run, and the graders that read it (``groundedness``,
        ``trajectory_judge``, ``external_checker``) silently judged nothing.
        Returns ``""`` when the agent produced no text at all.
        """
        if self.outcome is None:
            return ""

        val = self.outcome.output_data.get("final_output")
        if isinstance(val, str) and val:
            return val

        artifact = self.text_artifact
        if artifact is not None and artifact.content:
            return artifact.content
        return ""

    @property
    def tool_calls(self) -> list[Any]:
        """Shortcut: tool calls from the transcript."""
        if self.transcript is not None:
            return self.transcript.tool_calls
        return []

    @property
    def reasoning_steps(self) -> list[str]:
        """Shortcut: reasoning steps from the transcript."""
        if self.transcript is not None:
            return self.transcript.reasoning_steps
        return []

    @property
    def state_changes(self) -> list[Any]:
        """Shortcut: state changes flattened across all tool calls."""
        if self.transcript is not None:
            return self.transcript.all_state_changes()
        return []

    @property
    def total_duration_ms(self) -> float:
        """Shortcut: total execution duration from the transcript."""
        if self.transcript is not None:
            return self.transcript.total_duration_ms
        return 0.0

    def workspace_file(self, name: str) -> Path:
        """Path to *name* inside the shared grade workspace.

        Raises RuntimeError when no workspace was provided, rather than
        silently writing into the current working directory.
        """
        if self.workspace is None:
            raise RuntimeError(
                "No grade workspace available — this grader was called outside "
                "a runner that provides one"
            )
        self.workspace.mkdir(parents=True, exist_ok=True)
        return self.workspace / name

    def read_workspace_file(self, name: str) -> str | None:
        """Text of a file an earlier grader left in the workspace, or None."""
        if self.workspace is None:
            return None
        path = self.workspace / name
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8")

    @property
    def has_transcript(self) -> bool:
        """Whether transcript data is available."""
        return self.transcript is not None

    @property
    def has_outcome(self) -> bool:
        """Whether outcome data is available."""
        return self.outcome is not None

    def get_reference_image(self, name: str) -> Image.Image | None:
        """Get a named reference image from the reference_images dict."""
        return self.reference_images.get(name)

    @property
    def has_reference_images(self) -> bool:
        """Whether any reference images are available (single or dict)."""
        if self.reference_image is not None:
            return True
        return len(self.reference_images) > 0

    @property
    def has_output(self) -> bool:
        """Type-agnostic check: does the outcome contain any artifact with output?"""
        if self.outcome is not None:
            return self.outcome.has_output
        return False

    @property
    def code_artifact(self) -> CodeArtifact | None:
        """Shortcut: first CodeArtifact from the outcome, or None."""
        if self.outcome is not None:
            return self.outcome.get_artifact(CodeArtifact)
        return None

    @property
    def text_artifact(self) -> TextArtifact | None:
        """Shortcut: first TextArtifact from the outcome, or None."""
        if self.outcome is not None:
            return self.outcome.get_artifact(TextArtifact)
        return None

    def artifact(self, artifact_type: type[Artifact]) -> Artifact | None:
        """Generic accessor: first artifact of any registered type."""
        if self.outcome is not None:
            return self.outcome.get_artifact(artifact_type)
        return None

    # ------------------------------------------------------------------
    # Leak detection (Stripe-style answer leakage check)
    # ------------------------------------------------------------------

    def check_leaks(
        self,
        extra_patterns: list[str] | None = None,
    ) -> "LeakCheckResult":
        """Scan transcript for leaked grader/solution markers.

        Searches tool call inputs, outputs, and reasoning steps for any
        of the configured ``leak_markers`` (plus optional extra patterns).
        If any marker is found, the trial should be considered tainted.

        Args:
            extra_patterns: Additional patterns to check beyond
                ``self.leak_markers``.

        Returns:
            LeakCheckResult with leaked markers and searchable text.
        """
        patterns = list(self.leak_markers)
        if extra_patterns:
            patterns.extend(extra_patterns)

        if not patterns or self.transcript is None:
            return LeakCheckResult(leaked=[], searched_patterns=patterns)

        text = self._build_transcript_text()
        leaked = [p for p in patterns if p in text]
        return LeakCheckResult(leaked=leaked, searched_patterns=patterns)

    def _build_transcript_text(self) -> str:
        """Build a searchable text blob from transcript data."""
        if self.transcript is None:
            return ""

        parts: list[str] = []
        for tc in self.transcript.tool_calls:
            if tc.output is not None:
                parts.append(str(tc.output))
            if tc.input:
                parts.append(str(tc.input))
        parts.extend(self.transcript.reasoning_steps)
        return "\n".join(parts)


@dataclass
class LeakCheckResult:
    """Result of a leak detection scan."""

    leaked: list[str]
    searched_patterns: list[str]

    @property
    def has_leaks(self) -> bool:
        return len(self.leaked) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_leaks": self.has_leaks,
            "leaked_markers": self.leaked,
            "total_patterns_checked": len(self.searched_patterns),
        }


def generate_leak_marker(prefix: str = "COMPASS_LEAK") -> str:
    """Generate a unique leak marker for embedding in grader/solution files.

    Example:
        marker = generate_leak_marker()
        # → "COMPASS_LEAK_a1b2c3d4e5f6"

        # Embed in a grader script:
        script_content = f"# {marker}\\nrun_tests()"

        # Configure detection in YAML:
        # leak_markers: ["{marker}"]
    """
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def normalize_tag(tag: str) -> str:
    """Normalize an observation tag to lowercase snake_case.

    Applied at the ``GradeResult`` boundary rather than trusted to each
    grader: tags are only useful if ``Correct Bicycle`` and
    ``correct_bicycle`` land in the same bucket when aggregated.
    """
    return re.sub(r"[^a-z0-9]+", "_", str(tag).lower()).strip("_")


def normalize_tags(tags: list[str]) -> list[str]:
    """Normalize, de-duplicate and sort a list of observation tags."""
    return sorted({t for t in (normalize_tag(x) for x in tags) if t})


def normalize_metrics(metrics: dict[str, Any]) -> dict[str, float | bool]:
    """Keep only what can actually be aggregated: numbers and booleans.

    A metric is a measurement to be averaged across runs. A string or a nested
    object cannot be, and silently carrying one into the aggregate would
    produce either a crash or a meaningless number — so it is dropped here and
    belongs in ``details`` instead.
    """
    kept: dict[str, float | bool] = {}
    for key, value in metrics.items():
        # bool is a subclass of int: check it first or every flag becomes 1/0
        # and loses its "report me as a rate" meaning.
        if isinstance(value, bool):
            kept[normalize_tag(key) or str(key)] = value
        elif isinstance(value, (int, float)):
            kept[normalize_tag(key) or str(key)] = float(value)
    return kept


@dataclass
class GradeResult:
    """Result from a grader.

    ``score=None`` means **not measured** — the grader could not produce a
    number (an LLM judge timed out, a required input was missing). That is not
    the same as ``score=0.0`` ("measured, and it is bad"), and the aggregate
    keeps them apart: an unscored grader is left out of the score denominator
    instead of silently dragging the mean toward zero.

    ``tags`` and ``metrics`` are the two halves of "what did we observe":
    tags are categorical (*what was seen*), metrics are quantitative (*how
    much*). Both are emitted whether the grader passed or failed —
    ``failure_tags`` only explain failures.
    Semantics are deliberately presence-only: an absent tag means "not
    observed", **not** "false". Reports aggregate them as counts and shares,
    which turns a pile of scores into a behavioural profile ("62% of answers
    cited a document", "9% showed no bicycle at all").
    """

    name: str = ""
    grader_type: GraderType = GraderType.MODEL
    grader_scope: GraderScope = GraderScope.OUTCOME
    # Version of the grader that produced this result (audit provenance).
    # Stamped by the runner from Grader.version; a score is only comparable
    # across runs when the verifier version matches.
    grader_version: str = ""
    passed: bool = False
    score: float | None = 0.0
    weight: float = 1.0

    # Detailed results
    details: dict[str, Any] = field(default_factory=dict)
    # Neutral observations, emitted pass or fail. Presence-only.
    tags: list[str] = field(default_factory=list)
    # Quantitative observations, aggregated across runs: numbers become
    # mean ± stderr, booleans become rates.
    metrics: dict[str, float | bool] = field(default_factory=dict)
    failure_tags: list[str] = field(default_factory=list)  # Structured failure labels
    reasoning: str = ""

    # Error handling
    error: str | None = None

    def __post_init__(self) -> None:
        # Normalize at the boundary so no grader can leak an un-normalized tag
        # into the aggregate.
        if self.tags:
            self.tags = normalize_tags(self.tags)
        if self.metrics:
            self.metrics = normalize_metrics(self.metrics)

    @property
    def scored(self) -> bool:
        """Whether this grader produced an actual measurement."""
        return self.score is not None

    @property
    def weighted_score(self) -> float:
        """Weighted score; 0.0 when unscored (callers should filter on ``scored``)."""
        if self.score is None:
            return 0.0
        return self.score * self.weight

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "grader_type": self.grader_type.value,
            "grader_scope": self.grader_scope.value,
            "grader_version": self.grader_version,
            "passed": self.passed,
            "score": self.score,
            "scored": self.scored,
            "weight": self.weight,
            "weighted_score": self.weighted_score,
            "details": self.details,
            "tags": self.tags,
            "metrics": self.metrics,
            "failure_tags": self.failure_tags,
            "reasoning": self.reasoning,
            "error": self.error,
        }


class Grader(ABC):
    """Abstract base class for all graders."""

    name: str = "base"
    grader_type: GraderType = GraderType.MODEL
    grader_scope: GraderScope = GraderScope.OUTCOME
    # Verifier version. Bump whenever scoring logic changes in a way that can
    # move scores (new thresholds, reworded judge prompt, changed rubric), so
    # historical results remain auditable: a score delta between two runs can
    # be attributed to the agent only if the grader_version is unchanged.
    version: str = "1.0"

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize grader with config."""
        self.config = config or {}

    @abstractmethod
    async def grade(self, context: GradeContext) -> GradeResult:
        """Grade based on the provided context.

        The context provides independent access to:
        - context.transcript  → execution trace (tool calls, reasoning, timing)
        - context.outcome     → final result (image, output data, blocked)
        - context.image       → convenience shortcut for outcome.image
        - context.tool_calls  → convenience shortcut for transcript.tool_calls

        Graders should declare their `grader_scope` to indicate which
        parts of the context they require.

        Args:
            context: GradeContext with transcript, outcome, and input info.

        Returns:
            GradeResult with score and details.
        """
        pass

    def validate_context(self, context: GradeContext) -> str | None:
        """Validate that context has required data for this grader's scope.

        Returns:
            Error message if invalid, None if valid.
        """
        if self.grader_scope == GraderScope.OUTCOME and not context.has_outcome:
            return f"Grader '{self.name}' requires outcome data but none provided"
        if self.grader_scope == GraderScope.TRANSCRIPT and not context.has_transcript:
            return f"Grader '{self.name}' requires transcript data but none provided"
        if self.grader_scope == GraderScope.BOTH:
            if not context.has_outcome:
                return f"Grader '{self.name}' requires outcome data but none provided"
            if not context.has_transcript:
                return f"Grader '{self.name}' requires transcript data but none provided"
        return None

    def validate_config(self) -> bool:
        """Validate grader configuration."""
        return True


class CodeGrader(Grader):
    """Base class for deterministic, code-based graders.

    Code graders are:
    - Fast and deterministic
    - Objective and repeatable
    - Good for: size checks, format validation, state assertions, cost budgets
    - Limitations: brittle to valid variations
    """

    grader_type: GraderType = GraderType.CODE


class ModelGrader(Grader):
    """Base class for model-based (LLM/VLM) graders.

    Model graders are:
    - Flexible and can understand nuance
    - Good for: semantic matching, quality assessment, complex criteria
    - Limitations: non-deterministic, requires calibration
    """

    grader_type: GraderType = GraderType.MODEL


class HumanGrader(Grader):
    """Base class for human evaluation graders.

    Human graders are:
    - Gold standard for quality
    - Good for: subjective quality, complex judgments, calibration
    - Limitations: expensive, slow, doesn't scale
    """

    grader_type: GraderType = GraderType.HUMAN

    async def grade(self, context: GradeContext) -> GradeResult:
        """Human grading - returns pending result for human input."""
        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=False,
            score=0.0,
            details={"status": "pending_human_review"},
            reasoning="Awaiting human evaluation",
        )

    async def submit_human_result(
        self,
        task_id: str,
        passed: bool,
        score: float,
        reasoning: str = "",
    ) -> GradeResult:
        """Submit human evaluation result."""
        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=passed,
            score=score,
            details={"status": "human_reviewed", "task_id": task_id},
            reasoning=reasoning,
        )
