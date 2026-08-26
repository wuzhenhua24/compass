"""Tests for scenario loading."""

import tempfile

import pytest

from compass.core.scenario import (
    EvaluatorConfig,
    ExpectedConfig,
    GraderConfig,
    GraderType,
    Scenario,
    TestCase,
)


class TestScenario:
    """Tests for Scenario class."""

    def test_from_dict_minimal(self):
        """Test creating scenario from minimal dict."""
        data = {
            "name": "Test Scenario",
            "agent": {
                "adapter": "image",
                "endpoint": "http://localhost:8188",
            },
            "cases": [],
        }

        scenario = Scenario.from_dict(data)

        assert scenario.name == "Test Scenario"
        assert scenario.agent.adapter == "image"
        assert len(scenario.cases) == 0

    def test_from_dict_with_cases(self):
        """Test creating scenario with test cases."""
        data = {
            "name": "Test Scenario",
            "agent": {"adapter": "image"},
            "cases": [
                {
                    "id": "test_1",
                    "input": {
                        "prompt": "A cat",
                    },
                    "graders": [
                        {
                            "name": "semantic_match",
                            "weight": 0.5,
                        }
                    ],
                }
            ],
        }

        scenario = Scenario.from_dict(data)

        assert len(scenario.cases) == 1
        assert scenario.cases[0].id == "test_1"
        assert scenario.cases[0].input.prompt == "A cat"
        assert len(scenario.cases[0].graders) == 1

    def test_from_yaml(self):
        """Test loading scenario from YAML file."""
        yaml_content = """
name: YAML Test
agent:
  adapter: image
  endpoint: http://localhost:8188
cases:
  - id: case_1
    input:
      prompt: Test prompt
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            scenario = Scenario.from_yaml(f.name)

            assert scenario.name == "YAML Test"
            assert len(scenario.cases) == 1

    def test_get_case(self):
        """Test getting a case by ID."""
        scenario = Scenario.from_dict({
            "name": "Test",
            "agent": {"adapter": "test"},
            "cases": [
                {"id": "case_1", "input": {"prompt": "Test 1"}},
                {"id": "case_2", "input": {"prompt": "Test 2"}},
            ],
        })

        case = scenario.get_case("case_1")
        assert case is not None
        assert case.id == "case_1"

        assert scenario.get_case("nonexistent") is None

    def test_filter_by_tags(self):
        """Test filtering cases by tags."""
        scenario = Scenario.from_dict({
            "name": "Test",
            "agent": {"adapter": "test"},
            "cases": [
                {"id": "case_1", "input": {"prompt": "Test 1"}, "tags": ["landscape"]},
                {"id": "case_2", "input": {"prompt": "Test 2"}, "tags": ["portrait"]},
                {"id": "case_3", "input": {"prompt": "Test 3"}, "tags": ["landscape", "nature"]},
            ],
        })

        landscape_cases = scenario.filter_by_tags(["landscape"])
        assert len(landscape_cases) == 2

        portrait_cases = scenario.filter_by_tags(["portrait"])
        assert len(portrait_cases) == 1


class TestEvaluatorConfig:
    """Tests for EvaluatorConfig class."""

    def test_defaults(self):
        """Test default values."""
        config = EvaluatorConfig(name="test")

        assert config.name == "test"
        assert config.weight == 1.0
        assert config.required is False
        assert config.gate is False
        assert config.config == {}

    def test_with_config(self):
        """Test with custom config."""
        config = EvaluatorConfig(
            name="semantic_match",
            weight=0.5,
            required=True,
            config={"threshold": 0.3},
        )

        assert config.weight == 0.5
        assert config.required is True
        assert config.config["threshold"] == 0.3

    def test_gate_flag(self):
        """Test gate flag on GraderConfig."""
        config = EvaluatorConfig(name="text_rendering", gate=True)

        assert config.gate is True
        assert config.required is False  # gate and required are independent


class TestForCodingEval:
    """Tests for Scenario.for_coding_eval() factory method."""

    def test_for_coding_eval_valid(self):
        """Test creating a valid coding eval scenario."""
        scenario = Scenario.for_coding_eval(
            name="Coding Test",
            cases=[
                {
                    "id": "case_1",
                    "files": {"main.py": "print('hello')"},
                    "prompt": "Write hello world",
                },
            ],
        )
        assert isinstance(scenario, Scenario)
        assert scenario.name == "Coding Test"
        assert scenario.agent.adapter == "coding"
        assert len(scenario.cases) == 1

    def test_files_in_params(self):
        """Test that files are mapped to case.input.params as list format."""
        scenario = Scenario.for_coding_eval(
            name="Test",
            cases=[
                {
                    "id": "case_1",
                    "files": {"main.py": "x = 1", "utils.py": "y = 2"},
                },
            ],
        )
        case = scenario.cases[0]
        # Files should be normalized to list[{path, content}] format
        files = case.input.params["files"]
        assert isinstance(files, list)
        assert len(files) == 2
        paths = {f["path"] for f in files}
        assert paths == {"main.py", "utils.py"}
        content_map = {f["path"]: f["content"] for f in files}
        assert content_map["main.py"] == "x = 1"
        assert content_map["utils.py"] == "y = 2"
        assert "run_command" in case.input.params

    def test_expected_files_in_metadata(self):
        """Test that expected_files are mapped to case.metadata."""
        expected = [{"path": "main.py", "content": "x = 1\n"}]
        scenario = Scenario.for_coding_eval(
            name="Test",
            cases=[
                {
                    "id": "case_1",
                    "files": {"main.py": "x = 1"},
                    "expected_files": expected,
                },
            ],
        )
        case = scenario.cases[0]
        assert case.metadata["expected_files"] == expected

    def test_default_graders(self):
        """Test that default graders include exit_code_check."""
        scenario = Scenario.for_coding_eval(
            name="Test",
            cases=[{"id": "case_1", "files": {}}],
        )
        assert len(scenario.default_graders) == 1
        assert scenario.default_graders[0].name == "exit_code_check"

    def test_custom_graders(self):
        """Test custom default graders override the default."""
        scenario = Scenario.for_coding_eval(
            name="Test",
            cases=[{"id": "case_1", "files": {}}],
            default_graders=[
                {"name": "lint", "config": {"max_errors": 5}},
                {"name": "type_check"},
            ],
        )
        assert len(scenario.default_graders) == 2
        assert scenario.default_graders[0].name == "lint"
        assert scenario.default_graders[1].name == "type_check"


class TestExpectedConfig:
    """Tests for ExpectedConfig simplified expectations."""

    def test_is_empty(self):
        """Empty config should be detected."""
        config = ExpectedConfig()
        assert config.is_empty() is True

        config = ExpectedConfig(contains=["hello"])
        assert config.is_empty() is False

    def test_contains_to_graders(self):
        """Contains should map to style_convention grader."""
        config = ExpectedConfig(contains=["hello", "world"])
        graders = config.to_graders()

        assert len(graders) == 1
        assert graders[0].name == "style_convention"
        assert graders[0].config["required_phrases"] == ["hello", "world"]

    def test_not_contains_to_graders(self):
        """Not contains should map to forbidden_phrases."""
        config = ExpectedConfig(not_contains=["error", "failed"])
        graders = config.to_graders()

        assert len(graders) == 1
        assert graders[0].name == "style_convention"
        assert graders[0].config["forbidden_phrases"] == ["error", "failed"]

    def test_matches_single_pattern(self):
        """Single regex pattern should work."""
        config = ExpectedConfig(matches=r"\d{4}-\d{2}-\d{2}")
        graders = config.to_graders()

        assert graders[0].config["required_patterns"] == [r"\d{4}-\d{2}-\d{2}"]

    def test_matches_multiple_patterns(self):
        """Multiple regex patterns should work."""
        config = ExpectedConfig(matches=[r"\d+", r"[a-z]+"])
        graders = config.to_graders()

        assert graders[0].config["required_patterns"] == [r"\d+", r"[a-z]+"]

    def test_not_matches_to_graders(self):
        """Not matches should map to forbidden_patterns."""
        config = ExpectedConfig(not_matches=r"password:\s*\S+")
        graders = config.to_graders()

        assert graders[0].config["forbidden_patterns"] == [r"password:\s*\S+"]

    def test_length_constraints(self):
        """Length constraints should map to style_convention."""
        config = ExpectedConfig(min_words=10, max_words=100)
        graders = config.to_graders()

        assert graders[0].config["min_words"] == 10
        assert graders[0].config["max_words"] == 100

    def test_char_length_constraints(self):
        """Character length constraints should map correctly."""
        config = ExpectedConfig(min_length=50, max_length=500)
        graders = config.to_graders()

        assert graders[0].config["min_chars"] == 50
        assert graders[0].config["max_chars"] == 500

    def test_equals_to_graders(self):
        """Equals should map to exact_match grader."""
        config = ExpectedConfig(equals="expected output")
        graders = config.to_graders()

        assert len(graders) == 1
        assert graders[0].name == "exact_match"
        assert graders[0].config["expected"] == "expected output"

    def test_equals_json_to_graders(self):
        """Equals JSON should map to exact_match grader."""
        config = ExpectedConfig(equals_json={"key": "value"})
        graders = config.to_graders()

        assert len(graders) == 1
        assert graders[0].name == "exact_match"
        assert graders[0].config["expected_json"] == {"key": "value"}

    def test_schema_to_graders(self):
        """Schema should map to json_schema grader."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }
        config = ExpectedConfig(json_schema=schema)
        graders = config.to_graders()

        assert len(graders) == 1
        assert graders[0].name == "json_schema"
        assert graders[0].config["schema"] == schema
        assert graders[0].config["strict"] is True

    def test_schema_non_strict(self):
        """Schema with strict=False should pass through."""
        config = ExpectedConfig(
            json_schema={"type": "object"},
            json_schema_strict=False,
        )
        graders = config.to_graders()

        assert graders[0].config["strict"] is False

    def test_similar_to_graders(self):
        """Similar to should map to semantic_match grader."""
        config = ExpectedConfig(
            similar_to="A friendly greeting message",
            similarity_threshold=0.8,
        )
        graders = config.to_graders()

        assert len(graders) == 1
        assert graders[0].name == "semantic_match"
        assert graders[0].type == GraderType.MODEL
        assert graders[0].config["reference_text"] == "A friendly greeting message"
        assert graders[0].config["threshold"] == 0.8

    def test_unknown_expected_keys_are_rejected(self):
        """Silently ignoring a typo means checking nothing while reporting a pass."""
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            ExpectedConfig.model_validate({"contian": ["typo"]})

    def test_the_removed_assertions_field_is_gone(self):
        """It expanded to an unregistered grader, so it only ever failed cases."""
        import pydantic

        assert "assertions" not in ExpectedConfig.model_fields
        with pytest.raises(pydantic.ValidationError):
            ExpectedConfig.model_validate({"assertions": {"has_name": True}})

    def test_combined_expectations(self):
        """Multiple expectations should generate multiple graders."""
        config = ExpectedConfig(
            contains=["hello"],
            json_schema={"type": "object"},
            similar_to="greeting",
        )
        graders = config.to_graders()

        assert len(graders) == 3
        grader_names = [g.name for g in graders]
        assert "style_convention" in grader_names
        assert "json_schema" in grader_names
        assert "semantic_match" in grader_names

    def test_style_convention_combined(self):
        """Multiple style checks should be combined into one grader."""
        config = ExpectedConfig(
            contains=["hello"],
            not_contains=["error"],
            min_words=5,
            max_words=50,
        )
        graders = config.to_graders()

        # All should be in one style_convention grader
        assert len(graders) == 1
        assert graders[0].name == "style_convention"
        assert graders[0].config["required_phrases"] == ["hello"]
        assert graders[0].config["forbidden_phrases"] == ["error"]
        assert graders[0].config["min_words"] == 5
        assert graders[0].config["max_words"] == 50


class TestTestCaseExpected:
    """Tests for TestCase with expected field."""

    def test_get_all_graders_empty(self):
        """No expected or graders should return empty list."""
        case = TestCase(id="test", input={"prompt": "test"})
        assert case.get_all_graders() == []

    def test_get_all_graders_from_expected(self):
        """Expected should be converted to graders."""
        case = TestCase(
            id="test",
            input={"prompt": "test"},
            expected=ExpectedConfig(contains=["hello"]),
        )
        graders = case.get_all_graders()

        assert len(graders) == 1
        assert graders[0].name == "style_convention"

    def test_get_all_graders_from_explicit(self):
        """Explicit graders should be included."""
        case = TestCase(
            id="test",
            input={"prompt": "test"},
            graders=[GraderConfig(name="custom_grader")],
        )
        graders = case.get_all_graders()

        assert len(graders) == 1
        assert graders[0].name == "custom_grader"

    def test_get_all_graders_combined(self):
        """Expected and explicit graders should be combined."""
        case = TestCase(
            id="test",
            input={"prompt": "test"},
            expected=ExpectedConfig(contains=["hello"]),
            graders=[GraderConfig(name="custom_grader")],
        )
        graders = case.get_all_graders()

        assert len(graders) == 2
        # Expected comes first
        assert graders[0].name == "style_convention"
        # Explicit comes second
        assert graders[1].name == "custom_grader"

    def test_from_yaml_with_expected(self):
        """Test loading case with expected from YAML."""
        yaml_content = """
name: Expected Test
agent:
  adapter: test
cases:
  - id: simple_test
    input:
      prompt: "What is 2+2?"
    expected:
      contains: ["4", "four"]
      json_schema:
        type: object
        properties:
          answer: { type: integer }
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            scenario = Scenario.from_yaml(f.name)
            case = scenario.cases[0]

            assert case.expected is not None
            assert case.expected.contains == ["4", "four"]
            assert case.expected.json_schema is not None

            graders = case.get_all_graders()
            assert len(graders) == 2


class TestScenarioWithExpected:
    """Tests for Scenario.get_graders_for_case with expected."""

    def test_get_graders_includes_expected(self):
        """get_graders_for_case should include graders from expected."""
        scenario = Scenario.from_dict({
            "name": "Test",
            "agent": {"adapter": "test"},
            "default_graders": [{"name": "safety_check"}],
            "cases": [
                {
                    "id": "case_1",
                    "input": {"prompt": "Test"},
                    "expected": {
                        "contains": ["hello"],
                    },
                    "graders": [{"name": "custom"}],
                }
            ],
        })

        case = scenario.cases[0]
        graders = scenario.get_graders_for_case(case)

        # Order: default -> expected -> explicit
        assert len(graders) == 3
        assert graders[0].name == "safety_check"  # default
        assert graders[1].name == "style_convention"  # from expected
        assert graders[2].name == "custom"  # explicit

    def test_empty_expected_no_extra_graders(self):
        """Empty expected should not add graders."""
        scenario = Scenario.from_dict({
            "name": "Test",
            "agent": {"adapter": "test"},
            "cases": [
                {
                    "id": "case_1",
                    "input": {"prompt": "Test"},
                    "expected": {},  # Empty expected
                }
            ],
        })

        case = scenario.cases[0]
        graders = scenario.get_graders_for_case(case)

        assert len(graders) == 0


class TestGetAggregationForCase:
    """Tests for Scenario.get_aggregation_for_case."""

    def test_uses_default_when_case_has_defaults(self):
        """Should use scenario default when case aggregation is all defaults."""
        from compass.core.scenario import ShortCircuitMode

        scenario = Scenario.from_dict({
            "name": "Test",
            "agent": {"adapter": "test"},
            "default_aggregation": {
                "pass_threshold": 0.8,
                "method": "weighted_sum",
                "short_circuit": "code_fail",
            },
            "cases": [{"id": "case_1", "input": {"prompt": "Test"}}],
        })

        case = scenario.cases[0]
        agg = scenario.get_aggregation_for_case(case)

        # Should use scenario default
        assert agg.pass_threshold == 0.8
        assert agg.short_circuit == ShortCircuitMode.CODE_FAIL

    def test_uses_case_aggregation_when_threshold_differs(self):
        """Should use case aggregation when pass_threshold is set."""
        scenario = Scenario.from_dict({
            "name": "Test",
            "agent": {"adapter": "test"},
            "default_aggregation": {"pass_threshold": 0.8},
            "cases": [
                {
                    "id": "case_1",
                    "input": {"prompt": "Test"},
                    "aggregation": {"pass_threshold": 0.5},
                }
            ],
        })

        case = scenario.cases[0]
        agg = scenario.get_aggregation_for_case(case)

        assert agg.pass_threshold == 0.5

    def test_uses_case_aggregation_when_short_circuit_set(self):
        """Should use case aggregation when short_circuit is set."""
        from compass.core.scenario import ShortCircuitMode

        scenario = Scenario.from_dict({
            "name": "Test",
            "agent": {"adapter": "test"},
            "default_aggregation": {"short_circuit": "disabled"},
            "cases": [
                {
                    "id": "case_1",
                    "input": {"prompt": "Test"},
                    "aggregation": {"short_circuit": "code_pass"},
                }
            ],
        })

        case = scenario.cases[0]
        agg = scenario.get_aggregation_for_case(case)

        assert agg.short_circuit == ShortCircuitMode.CODE_PASS

    def test_uses_case_aggregation_when_method_differs(self):
        """Should use case aggregation when method is set."""
        scenario = Scenario.from_dict({
            "name": "Test",
            "agent": {"adapter": "test"},
            "default_aggregation": {"method": "weighted_sum"},
            "cases": [
                {
                    "id": "case_1",
                    "input": {"prompt": "Test"},
                    "aggregation": {"method": "min"},
                }
            ],
        })

        case = scenario.cases[0]
        agg = scenario.get_aggregation_for_case(case)

        assert agg.method == "min"

    def test_explicit_default_threshold_0_7_uses_case(self):
        """Explicit pass_threshold=0.7 should still use case aggregation (bug fix)."""
        from compass.core.scenario import ShortCircuitMode

        scenario = Scenario.from_dict({
            "name": "Test",
            "agent": {"adapter": "test"},
            "default_aggregation": {
                "pass_threshold": 0.8,
                "short_circuit": "code_fail",
            },
            "cases": [
                {
                    "id": "case_1",
                    "input": {"prompt": "Test"},
                    # User explicitly sets 0.7 threshold with different short_circuit
                    "aggregation": {
                        "pass_threshold": 0.7,
                        "short_circuit": "code_pass",
                    },
                }
            ],
        })

        case = scenario.cases[0]
        agg = scenario.get_aggregation_for_case(case)

        # Should use case aggregation because short_circuit differs from default
        assert agg.short_circuit == ShortCircuitMode.CODE_PASS

    def test_partial_override_preserves_scenario_defaults(self):
        """A case that overrides only one field must keep the scenario's other
        defaults (field-level merge), not silently reset them."""
        from compass.core.scenario import ShortCircuitMode

        scenario = Scenario.from_dict({
            "name": "Test",
            "agent": {"adapter": "test"},
            "default_aggregation": {
                "pass_threshold": 0.8,
                "short_circuit": "code_fail",
                "required_graders": ["safety_check"],
            },
            "cases": [
                {
                    "id": "case_1",
                    "input": {"prompt": "Test"},
                    # Only pass_threshold is overridden.
                    "aggregation": {"pass_threshold": 0.9},
                }
            ],
        })

        agg = scenario.get_aggregation_for_case(scenario.cases[0])

        assert agg.pass_threshold == 0.9  # overridden
        # Scenario defaults preserved (previously dropped to library defaults):
        assert agg.short_circuit == ShortCircuitMode.CODE_FAIL
        assert agg.required_graders == ["safety_check"]

    def test_case_can_restore_library_default_value(self):
        """A case can explicitly set a field back to a library-default value and
        have it respected, instead of being ignored as 'unset'."""
        scenario = Scenario.from_dict({
            "name": "Test",
            "agent": {"adapter": "test"},
            "default_aggregation": {"pass_threshold": 0.8},
            "cases": [
                {
                    "id": "case_1",
                    "input": {"prompt": "Test"},
                    # 0.7 happens to equal the library default; must still win.
                    "aggregation": {"pass_threshold": 0.7},
                }
            ],
        })

        agg = scenario.get_aggregation_for_case(scenario.cases[0])
        assert agg.pass_threshold == 0.7


class TestGateConfigParsing:
    """Tests for gate field parsing in GraderConfig."""

    def test_gate_from_yaml(self):
        """Gate flag should be correctly parsed from YAML."""
        yaml_content = """
name: Gate Parsing Test
agent:
  adapter: test
cases:
  - id: case_1
    input:
      prompt: Test
    graders:
      - name: text_rendering
        gate: true
        config: {}
      - name: aesthetic_score
        weight: 0.6
        config: {}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            scenario = Scenario.from_yaml(f.name)
            case = scenario.cases[0]

            assert case.graders[0].gate is True
            assert case.graders[0].name == "text_rendering"
            assert case.graders[1].gate is False
            assert case.graders[1].name == "aesthetic_score"

    def test_gate_from_dict(self):
        """Gate flag should be correctly parsed from dict."""
        scenario = Scenario.from_dict({
            "name": "Test",
            "agent": {"adapter": "test"},
            "cases": [
                {
                    "id": "case_1",
                    "input": {"prompt": "Test"},
                    "graders": [
                        {"name": "check_a", "gate": True},
                        {"name": "check_b"},
                    ],
                }
            ],
        })

        case = scenario.cases[0]
        assert case.graders[0].gate is True
        assert case.graders[1].gate is False


class TestFilterByStage:
    """Tests for stage field and filter_by_stage."""

    def test_stage_default_empty_string(self):
        """Default stage should be empty string."""
        case = TestCase(id="test", input={"prompt": "test"})
        assert case.stage == ""

    def test_stage_from_yaml(self):
        """Stage should be correctly parsed from YAML."""
        yaml_content = """
name: Stage Test
agent:
  adapter: test
cases:
  - id: case_1
    input:
      prompt: Test
    stage: smoke
  - id: case_2
    input:
      prompt: Test 2
    stage: integration
  - id: case_3
    input:
      prompt: Test 3
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            scenario = Scenario.from_yaml(f.name)

            assert scenario.cases[0].stage == "smoke"
            assert scenario.cases[1].stage == "integration"
            assert scenario.cases[2].stage == ""

    def test_filter_by_stage_single(self):
        """Single stage filter should return matching cases."""
        scenario = Scenario.from_dict({
            "name": "Test",
            "agent": {"adapter": "test"},
            "cases": [
                {"id": "case_1", "input": {"prompt": "T"}, "stage": "smoke"},
                {"id": "case_2", "input": {"prompt": "T"}, "stage": "integration"},
                {"id": "case_3", "input": {"prompt": "T"}, "stage": "smoke"},
            ],
        })

        result = scenario.filter_by_stage(["smoke"])
        assert len(result) == 2
        assert {c.id for c in result} == {"case_1", "case_3"}

    def test_filter_by_stage_multiple(self):
        """Multiple stages should use OR logic."""
        scenario = Scenario.from_dict({
            "name": "Test",
            "agent": {"adapter": "test"},
            "cases": [
                {"id": "case_1", "input": {"prompt": "T"}, "stage": "smoke"},
                {"id": "case_2", "input": {"prompt": "T"}, "stage": "integration"},
                {"id": "case_3", "input": {"prompt": "T"}, "stage": "nightly"},
            ],
        })

        result = scenario.filter_by_stage(["smoke", "integration"])
        assert len(result) == 2
        assert {c.id for c in result} == {"case_1", "case_2"}

    def test_filter_by_stage_no_match(self):
        """No matching stage should return empty list."""
        scenario = Scenario.from_dict({
            "name": "Test",
            "agent": {"adapter": "test"},
            "cases": [
                {"id": "case_1", "input": {"prompt": "T"}, "stage": "smoke"},
            ],
        })

        result = scenario.filter_by_stage(["regression"])
        assert len(result) == 0

    def test_filter_by_stage_excludes_empty_stage(self):
        """Cases with empty stage should not match non-empty stage filters."""
        scenario = Scenario.from_dict({
            "name": "Test",
            "agent": {"adapter": "test"},
            "cases": [
                {"id": "case_1", "input": {"prompt": "T"}, "stage": "smoke"},
                {"id": "case_2", "input": {"prompt": "T"}},  # no stage
            ],
        })

        result = scenario.filter_by_stage(["smoke"])
        assert len(result) == 1
        assert result[0].id == "case_1"

    def test_stage_from_dict(self):
        """Stage should be correctly parsed from dict."""
        scenario = Scenario.from_dict({
            "name": "Test",
            "agent": {"adapter": "test"},
            "cases": [
                {"id": "case_1", "input": {"prompt": "T"}, "stage": "regression"},
            ],
        })

        assert scenario.cases[0].stage == "regression"


# ===================================================================
# defaults 里的每个字段都必须有人读
# ===================================================================


class TestDefaultsAreLoadBearing:
    """A scenario field nobody reads is a promise the YAML cannot keep.

    `defaults` once carried `timeout` and an `environment` block
    (`isolation` / `clean_cache`). Both were documented as working and neither
    was read anywhere, so setting them got silence. This asserts that every
    field still on DefaultsConfig is consumed by something outside its own
    definition — the check that would have caught it.
    """

    @staticmethod
    def _source_outside_the_model() -> str:
        import pathlib

        scenario_py = pathlib.Path("src/compass/core/scenario.py")
        everything = "\n".join(
            p.read_text(encoding="utf-8")
            for p in pathlib.Path("src/compass").rglob("*.py")
        )
        return everything.replace(scenario_py.read_text(encoding="utf-8"), "")

    def test_every_defaults_field_is_read_somewhere(self):
        from compass.core.scenario import DefaultsConfig, Scenario

        scenario_py = __import__("pathlib").Path(
            "src/compass/core/scenario.py"
        ).read_text(encoding="utf-8")
        outside = self._source_outside_the_model()

        for field in DefaultsConfig.model_fields:
            reads = (
                f"defaults.{field}" in scenario_py
                or f"defaults.{field}" in outside
                or f'defaults["{field}"]' in outside
            )
            assert reads, (
                f"DefaultsConfig.{field} is never read — either wire it up or "
                f"drop it, but do not document it as configuration"
            )
        assert "trials" in DefaultsConfig.model_fields
        assert Scenario(
            name="n", agent={"adapter": "image"}, cases=[]
        ).defaults.trials == 1

    def test_an_old_scenario_carrying_the_dropped_keys_still_loads(self):
        """Pydantic ignores unknown keys, so removing them broke no YAML."""
        from compass.core.scenario import Scenario

        scenario = Scenario.model_validate(
            {
                "name": "legacy",
                "agent": {"adapter": "image"},
                "defaults": {
                    "trials": 3,
                    "timeout": 300,
                    "environment": {"isolation": True, "clean_cache": True},
                },
                "cases": [{"id": "c", "input": {"prompt": "p"}}],
            }
        )
        assert scenario.defaults.trials == 3

    def test_for_coding_eval_timeout_reaches_the_adapter(self):
        """It used to land on DefaultsConfig, where nothing read it."""
        from compass.core.scenario import Scenario

        scenario = Scenario.for_coding_eval(
            "n", [{"id": "a", "files": {}}], timeout=600
        )
        assert scenario.agent.config["timeout"] == 600

    def test_an_explicit_adapter_timeout_wins(self):
        from compass.core.scenario import Scenario

        scenario = Scenario.for_coding_eval(
            "n", [{"id": "a", "files": {}}], adapter_config={"timeout": 5}, timeout=600
        )
        assert scenario.agent.config["timeout"] == 5
