"""Scenario definition and loading."""

from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from compass.core.sweep import SweepConfig


class GraderType(str, Enum):
    """Type of grader."""

    CODE = "code"
    MODEL = "model"
    HUMAN = "human"


class GraderConfig(BaseModel):
    """Configuration for a single grader.

    Gate graders are hard pass/fail checks that do NOT participate in
    score calculation.  When ``gate=True`` the grader must pass for the
    overall case to pass, but its score is excluded from the weighted
    average so it cannot drag the aggregate number down.

    ``required=True`` additionally **halts the chain**: graders declared after
    it are not run.  Graders execute in declared order over a shared
    workspace, so a failed prerequisite makes everything downstream
    meaningless — there is no point rendering an SVG that could not be
    extracted, or paying a VLM to judge an image that was never produced.
    """

    type: GraderType = GraderType.MODEL
    name: str
    # A name for *this instance*, when one case runs the same grader twice.
    # `name` is the registry key and cannot disambiguate them, so without a
    # label the two results are indistinguishable downstream: `breakdown` keys
    # on it and collapses, and `compass compare --on` cannot say which one it
    # was asked for. Cosmetic by construction — a label never moves a score,
    # which is why `case_grader_spec` excludes it from the fingerprint.
    label: str = ""
    weight: float = 1.0
    required: bool = False
    gate: bool = False  # Hard gate: must pass, excluded from score
    # Files this grader promises to write into the shared grade workspace.
    # The runner verifies they appeared; a grader that reports success without
    # producing what it promised is a silent failure, not a pass.
    creates: str | list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)

    @property
    def promised_files(self) -> list[str]:
        """``creates`` normalized to a list."""
        if isinstance(self.creates, str):
            return [self.creates] if self.creates else []
        return list(self.creates)

    @property
    def key(self) -> str:
        """How this grader instance is identified in results: label, else name."""
        return self.label or self.name


class ExpectedConfig(BaseModel):
    """Simplified expectation configuration for test cases.

    This provides a user-friendly way to define common expectations
    without manually configuring graders. Each field maps to a
    specific grader configuration.

    Example:
        expected:
          contains: ["hello", "world"]
          not_contains: ["error"]
          matches: "\\d{4}-\\d{2}-\\d{2}"
          schema:
            type: object
            properties:
              name: { type: string }
          similar_to: "A friendly greeting"
          similarity_threshold: 0.8

    Unknown keys are rejected rather than ignored. A typo'd ``contian:`` that
    silently expands to no grader at all is the worst kind of eval bug: the
    case reports a clean pass while checking nothing.
    """

    model_config = ConfigDict(extra="forbid")

    # Text content checks (maps to style_convention grader)
    contains: list[str] | None = None
    not_contains: list[str] | None = None
    matches: str | list[str] | None = None  # Regex pattern(s)
    not_matches: str | list[str] | None = None

    # Exact match (maps to exact_match grader)
    equals: str | None = None
    equals_json: dict[str, Any] | list[Any] | None = None

    # JSON Schema validation (maps to json_schema grader)
    json_schema: dict[str, Any] | None = None
    json_schema_strict: bool = True

    # Semantic similarity (maps to semantic grader)
    similar_to: str | None = None
    similarity_threshold: float = 0.7

    # Length constraints (maps to style_convention grader)
    min_length: int | None = None  # Characters
    max_length: int | None = None
    min_words: int | None = None
    max_words: int | None = None

    def is_empty(self) -> bool:
        """Check if no expectations are defined."""
        return all(
            getattr(self, field) is None
            for field in type(self).model_fields
            if field not in ("json_schema_strict", "similarity_threshold")
        )

    def to_graders(self) -> list["GraderConfig"]:
        """Convert expected config to list of grader configs.

        This is the core mapping from simplified expectations to
        full grader configurations.
        """
        graders: list[GraderConfig] = []

        # Style convention grader for text checks
        style_config: dict[str, Any] = {}
        if self.contains:
            style_config["required_phrases"] = self.contains
        if self.not_contains:
            style_config["forbidden_phrases"] = self.not_contains
        if self.matches:
            patterns = [self.matches] if isinstance(self.matches, str) else self.matches
            style_config["required_patterns"] = patterns
        if self.not_matches:
            patterns = [self.not_matches] if isinstance(self.not_matches, str) else self.not_matches
            style_config["forbidden_patterns"] = patterns
        if self.min_words is not None:
            style_config["min_words"] = self.min_words
        if self.max_words is not None:
            style_config["max_words"] = self.max_words
        if self.min_length is not None:
            style_config["min_chars"] = self.min_length
        if self.max_length is not None:
            style_config["max_chars"] = self.max_length

        if style_config:
            graders.append(
                GraderConfig(
                    type=GraderType.CODE,
                    name="style_convention",
                    config=style_config,
                )
            )

        # Exact match grader
        if self.equals is not None:
            graders.append(
                GraderConfig(
                    type=GraderType.CODE,
                    name="exact_match",
                    config={"expected": self.equals},
                )
            )

        if self.equals_json is not None:
            graders.append(
                GraderConfig(
                    type=GraderType.CODE,
                    name="exact_match",
                    config={"expected_json": self.equals_json},
                )
            )

        # JSON Schema grader
        if self.json_schema is not None:
            graders.append(
                GraderConfig(
                    type=GraderType.CODE,
                    name="json_schema",
                    config={
                        "schema": self.json_schema,
                        "strict": self.json_schema_strict,
                    },
                )
            )

        # Semantic similarity grader
        if self.similar_to is not None:
            graders.append(
                GraderConfig(
                    type=GraderType.MODEL,
                    name="semantic_match",
                    config={
                        "reference_text": self.similar_to,
                        "threshold": self.similarity_threshold,
                    },
                )
            )

        return graders


class ShortCircuitMode(str, Enum):
    """When to trigger short-circuit in Code-First mode.

    Modes:
    - DISABLED: Run all graders (no short-circuit)
    - CODE_FAIL: Skip Model Graders when any Code Grader fails
    - CODE_PASS: Skip Model Graders when all Code Graders pass (run Model only on failure)
    - REQUIRED_FAIL: Skip Model Graders when required Code Grader fails
    """

    DISABLED = "disabled"  # No short-circuit, run all graders
    CODE_FAIL = "code_fail"  # Short-circuit when any Code Grader fails
    CODE_PASS = "code_pass"  # Short-circuit when all Code Graders pass (Model runs only on failure)
    REQUIRED_FAIL = "required_fail"  # Short-circuit when required Code Grader fails


class AggregationConfig(BaseModel):
    """Configuration for score aggregation.

    Supports Code-First short-circuit mode to optimize evaluation:
    - Run Code Graders first (fast, cheap, deterministic)
    - If short-circuit condition is met, skip Model Graders (slow, expensive)
    - Save cost and provide faster feedback on obvious failures

    Example:
        aggregation:
          method: weighted_sum
          pass_threshold: 0.7
          short_circuit: required_fail  # Skip Model Graders if required Code fails
    """

    method: str = "weighted_sum"
    pass_threshold: float = 0.7
    required_graders: list[str] = Field(default_factory=list)

    # Code-First short-circuit mode
    short_circuit: ShortCircuitMode = ShortCircuitMode.DISABLED


class MetricsConfig(BaseModel):
    """Configuration for metrics reporting."""

    pass_at_k: list[int] = Field(default_factory=lambda: [1, 3])
    consistency: bool = True


class InputConfig(BaseModel):
    """Input configuration for a test case."""

    prompt: str
    negative_prompt: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    reference_images: dict[str, str] = Field(default_factory=dict)


class TestCase(BaseModel):
    """A single test case (task) definition.

    Test cases can define expectations in two ways:

    1. **Simple expectations** via `expected` field (recommended for common cases):
       ```yaml
       expected:
         contains: ["hello"]
         schema: { type: object }
       ```

    2. **Full grader configuration** via `graders` field (for complex cases):
       ```yaml
       graders:
         - name: json_schema
           config:
             schema: { ... }
             scoring_mode: partial
       ```

    Both can be used together - `expected` is converted to graders and
    merged with explicit `graders`.
    """

    id: str
    description: str = ""
    input: InputConfig

    # Expectation: "pass" for positive tests, "fail" for negative tests
    expect: Literal["pass", "fail"] = "pass"
    expect_reason: str = ""

    # Simplified expectations (auto-converted to graders)
    expected: ExpectedConfig | None = None

    # Trial configuration
    trials: int | None = None  # Override default trials

    # Graders (explicit configuration)
    graders: list[GraderConfig] = Field(default_factory=list)

    # Aggregation
    aggregation: AggregationConfig = Field(default_factory=AggregationConfig)

    # Metrics
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)

    # Metadata
    tags: list[str] = Field(default_factory=list)
    category: str = ""  # e.g., "backend", "fullstack", "gym"
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Leak detection markers (Stripe-style UUID strings)
    leak_markers: list[str] = Field(default_factory=list)

    # Stage (e.g., "smoke", "integration", "regression", "nightly")
    stage: str = ""

    # Parameter sweep
    sweep: SweepConfig | None = None

    @property
    def is_positive_test(self) -> bool:
        """Whether this is a positive test."""
        return self.expect == "pass"

    @property
    def is_negative_test(self) -> bool:
        """Whether this is a negative test."""
        return self.expect == "fail"

    def get_all_graders(self) -> list[GraderConfig]:
        """Get all graders including those generated from expected config.

        Returns graders in order:
        1. Graders generated from `expected` config
        2. Explicitly configured `graders`
        """
        all_graders: list[GraderConfig] = []

        # Convert expected to graders
        if self.expected is not None and not self.expected.is_empty():
            all_graders.extend(self.expected.to_graders())

        # Add explicit graders
        all_graders.extend(self.graders)

        return all_graders


class EnvironmentConfig(BaseModel):
    """Environment configuration."""

    isolation: bool = True
    clean_cache: bool = True
    timeout: int = 300


class DefaultsConfig(BaseModel):
    """Default configuration for all cases."""

    trials: int = 1
    timeout: int = 300
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)


class AgentConfig(BaseModel):
    """Agent configuration."""

    adapter: str
    endpoint: str = ""
    workflow: str = ""
    config: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def merge_top_level_into_config(self) -> "AgentConfig":
        """Merge top-level endpoint/workflow into config if not already set.

        This ensures that YAML like:
            agent:
              adapter: image
              endpoint: "http://localhost:8000"
              workflow: "workflow.json"

        Works the same as:
            agent:
              adapter: image
              config:
                endpoint: "http://localhost:8000"
                workflow: "workflow.json"
        """
        if self.endpoint and "endpoint" not in self.config:
            self.config["endpoint"] = self.endpoint
        if self.workflow and "workflow" not in self.config:
            self.config["workflow"] = self.workflow
        return self


class Scenario(BaseModel):
    """A test scenario containing multiple test cases."""

    name: str
    description: str = ""
    agent: AgentConfig

    # Defaults
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)

    # Default graders applied to all cases
    default_graders: list[GraderConfig] = Field(default_factory=list)

    # Default aggregation
    default_aggregation: AggregationConfig = Field(default_factory=AggregationConfig)

    # Test cases
    cases: list[TestCase] = Field(default_factory=list)

    # Parameter sweep (default for all cases)
    sweep: SweepConfig | None = None

    # Metadata
    tags: list[str] = Field(default_factory=list)
    category: str = ""  # Default category for all cases
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Leak detection markers (default for all cases)
    leak_markers: list[str] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Scenario":
        """Load scenario from YAML file."""
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scenario":
        """Create scenario from dictionary."""
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        """Save scenario to YAML file."""
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, allow_unicode=True, sort_keys=False)

    def get_case(self, case_id: str) -> TestCase | None:
        """Get a test case by ID."""
        for case in self.cases:
            if case.id == case_id:
                return case
        return None

    def get_trials_for_case(self, case: TestCase) -> int:
        """Get number of trials for a case (case override or default)."""
        return case.trials if case.trials is not None else self.defaults.trials

    def get_graders_for_case(self, case: TestCase) -> list[GraderConfig]:
        """Get graders for a case (default + expected + explicit graders).

        Graders are returned in order:
        1. Default graders from scenario
        2. Graders generated from case's `expected` config
        3. Explicitly configured case `graders`
        """
        # Start with default graders
        graders = list(self.default_graders)

        # Add case-specific graders (includes expected + explicit)
        graders.extend(case.get_all_graders())

        return graders

    def get_aggregation_for_case(self, case: TestCase) -> AggregationConfig:
        """Get aggregation config for a case via field-level merge.

        The scenario's ``default_aggregation`` is the base. Any field the case
        *explicitly* set in its own ``aggregation`` block overrides that base;
        unset fields fall through to the scenario default.

        This fixes two problems with the old "all-or-nothing" comparison:
        - A case overriding a single field (e.g. ``pass_threshold``) no longer
          silently drops the scenario's other defaults (e.g. ``short_circuit``).
        - A case can explicitly set a field back to a library-default value
          (e.g. ``pass_threshold: 0.7``) instead of being ignored because it
          "looks unset".

        Only fields present in ``model_fields_set`` count as explicit, so a case
        with no ``aggregation:`` block returns the scenario default unchanged.
        """
        overrides = {
            name: getattr(case.aggregation, name)
            for name in case.aggregation.model_fields_set
        }
        if not overrides:
            return self.default_aggregation
        return self.default_aggregation.model_copy(update=overrides)

    def get_category_for_case(self, case: TestCase) -> str:
        """Get effective category for a case (case override or scenario default)."""
        return case.category or self.category

    def get_leak_markers_for_case(self, case: TestCase) -> list[str]:
        """Get effective leak markers for a case (merged: scenario + case)."""
        markers = list(self.leak_markers)
        for m in case.leak_markers:
            if m not in markers:
                markers.append(m)
        return markers

    def filter_by_tags(self, tags: list[str]) -> list[TestCase]:
        """Filter test cases by tags."""
        return [case for case in self.cases if any(tag in case.tags for tag in tags)]

    def filter_by_category(self, categories: list[str]) -> list[TestCase]:
        """Filter test cases by category (case-level or scenario default)."""
        return [
            case for case in self.cases
            if self.get_category_for_case(case) in categories
        ]

    def filter_by_stage(self, stages: list[str]) -> list[TestCase]:
        """Filter test cases by stage."""
        return [case for case in self.cases if case.stage in stages]

    def filter_positive_tests(self) -> list[TestCase]:
        """Get all positive tests (expect=pass)."""
        return [case for case in self.cases if case.is_positive_test]

    def filter_negative_tests(self) -> list[TestCase]:
        """Get all negative tests (expect=fail)."""
        return [case for case in self.cases if case.is_negative_test]

    @property
    def positive_test_count(self) -> int:
        """Number of positive tests."""
        return len(self.filter_positive_tests())

    @property
    def negative_test_count(self) -> int:
        """Number of negative tests."""
        return len(self.filter_negative_tests())

    @classmethod
    def for_coding_eval(
        cls,
        name: str,
        cases: list[dict[str, Any]],
        *,
        description: str = "",
        adapter_config: dict[str, Any] | None = None,
        default_graders: list[dict[str, Any]] | None = None,
        sandbox_type: str = "local",
        run_command: str = "python main.py",
        timeout: int = 300,
    ) -> "Scenario":
        """Factory for creating coding evaluation scenarios.

        Each case dict should contain:
            id: str — unique case identifier.
            files: dict[str, str] — file path → content mapping.
            prompt: str (optional) — task prompt.
            description: str (optional) — case description.
            expected_files: list[dict] (optional) — reference files with
                ``path`` and ``content`` keys.
            graders: list[dict] (optional) — case-specific grader configs.
            tags: list[str] (optional) — tags for filtering.
            stage: str (optional) — test stage (e.g., "smoke", "integration").
            run_command: str (optional) — override the default run command.
        """
        grader_configs = default_graders or [{"name": "exit_code_check"}]
        parsed_graders = [GraderConfig.model_validate(g) for g in grader_configs]

        test_cases: list[TestCase] = []
        for case_data in cases:
            case_files_raw = case_data.get("files", {})
            # Normalize files to list[{path, content}] format
            # Accepts both dict[path→content] and list[{path, content}]
            if isinstance(case_files_raw, dict):
                case_files = [
                    {"path": path, "content": content}
                    for path, content in case_files_raw.items()
                ]
            else:
                case_files = case_files_raw
            case_run_cmd = case_data.get("run_command", run_command)

            params: dict[str, Any] = {
                "files": case_files,
                "run_command": case_run_cmd,
            }

            metadata: dict[str, Any] = {}
            expected_files = case_data.get("expected_files")
            if expected_files is not None:
                metadata["expected_files"] = expected_files

            case_graders_raw = case_data.get("graders")
            case_graders = (
                [GraderConfig.model_validate(g) for g in case_graders_raw]
                if case_graders_raw
                else []
            )

            test_cases.append(
                TestCase(
                    id=case_data["id"],
                    description=case_data.get("description", ""),
                    input=InputConfig(
                        prompt=case_data.get("prompt", ""),
                        params=params,
                    ),
                    graders=case_graders,
                    tags=case_data.get("tags", []),
                    stage=case_data.get("stage", ""),
                    metadata=metadata,
                )
            )

        return cls(
            name=name,
            description=description,
            agent=AgentConfig(
                adapter="coding",
                config=adapter_config or {},
            ),
            defaults=DefaultsConfig(timeout=timeout),
            default_graders=parsed_graders,
            cases=test_cases,
        )


# Backwards compatibility aliases
EvaluatorConfig = GraderConfig
