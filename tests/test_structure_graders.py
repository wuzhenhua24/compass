"""Tests for structured output validation graders."""

import pytest

from compass.core.transcript import Outcome
from compass.graders import get_grader
from compass.graders.base import GradeContext
from compass.graders.code.common.structure import JsonSchemaGrader, StructureCheckGrader

# =============================================================================
# JsonSchemaGrader Tests
# =============================================================================


class TestJsonSchemaGrader:
    """Tests for JsonSchemaGrader."""

    def test_registered(self):
        """Test grader is registered."""
        grader_cls = get_grader("json_schema")
        assert grader_cls is JsonSchemaGrader

    @pytest.mark.asyncio
    async def test_valid_json_passes(self):
        """Test valid JSON matching schema passes."""
        grader = JsonSchemaGrader({
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name", "age"],
            },
            "source": "output_data",
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={"name": "Alice", "age": 30}),
        )

        result = await grader.grade(context)
        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_invalid_json_fails(self):
        """Test invalid JSON fails validation."""
        grader = JsonSchemaGrader({
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name", "age"],
            },
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={"name": "Alice"}),  # Missing age
        )

        result = await grader.grade(context)
        assert result.passed is False
        assert result.score == 0.0
        assert "error" in result.details or result.error

    @pytest.mark.asyncio
    async def test_type_mismatch_fails(self):
        """Test type mismatch fails validation."""
        grader = JsonSchemaGrader({
            "schema": {
                "type": "object",
                "properties": {
                    "age": {"type": "integer"},
                },
            },
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={"age": "not a number"}),
        )

        result = await grader.grade(context)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_strict_mode(self):
        """Test strict mode rejects additional properties."""
        grader = JsonSchemaGrader({
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                },
            },
            "strict": True,
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={"name": "Alice", "extra": "field"}),
        )

        result = await grader.grade(context)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_extract_json_from_text(self):
        """Test extracting JSON from surrounding text."""
        grader = JsonSchemaGrader({
            "schema": {"type": "object"},
            "source": "output_data.response",
            "extract_json": True,
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(
                output_data={
                    "response": 'Here is the result:\n```json\n{"key": "value"}\n```'
                }
            ),
        )

        result = await grader.grade(context)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_no_content_fails(self):
        """Test missing content fails gracefully."""
        grader = JsonSchemaGrader({
            "schema": {"type": "object"},
            "source": "nonexistent",
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(),
        )

        result = await grader.grade(context)
        assert result.passed is False
        assert "No content found" in result.error

    # =========================================================================
    # Partial Scoring Mode Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_partial_scoring_mode_basic(self):
        """Test partial scoring mode gives partial credit."""
        grader = JsonSchemaGrader({
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "email": {"type": "string"},
                },
                "required": ["name", "age", "email"],
            },
            "scoring_mode": "partial",
            "pass_threshold": 0.5,
        })

        # 2 out of 3 fields correct
        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={
                "name": "Alice",
                "age": 30,
                # email missing
            }),
        )

        result = await grader.grade(context)
        # Should get partial credit
        assert 0 < result.score < 1.0
        assert "field_compliance" in result.details
        assert result.details["field_compliance"]["passed"] == 2
        assert result.details["field_compliance"]["failed"] == 1

    @pytest.mark.asyncio
    async def test_partial_scoring_type_error(self):
        """Test partial scoring with type errors."""
        grader = JsonSchemaGrader({
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "active": {"type": "boolean"},
                },
                "required": ["name"],
            },
            "scoring_mode": "partial",
            "pass_threshold": 0.6,
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={
                "name": "Alice",
                "age": "not a number",  # type error
                "active": True,
            }),
        )

        result = await grader.grade(context)
        # 2 out of 3 correct (name and active)
        assert 0 < result.score < 1.0
        # Check error categorization
        assert "error_summary" in result.details
        assert len(result.details["error_summary"]["type_error"]) > 0

    @pytest.mark.asyncio
    async def test_partial_scoring_with_pass_threshold(self):
        """Test pass threshold in partial mode."""
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"},
                "c": {"type": "string"},
                "d": {"type": "string"},
            },
            "required": ["a", "b", "c", "d"],
        }

        # High threshold - 3/4 fields correct should fail
        grader_high = JsonSchemaGrader({
            "schema": schema,
            "scoring_mode": "partial",
            "pass_threshold": 0.9,
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={
                "a": "1", "b": "2", "c": "3",
                # d missing
            }),
        )

        result_high = await grader_high.grade(context)
        assert result_high.passed is False  # 75% < 90%

        # Low threshold - 3/4 fields correct should pass
        grader_low = JsonSchemaGrader({
            "schema": schema,
            "scoring_mode": "partial",
            "pass_threshold": 0.6,
        })

        result_low = await grader_low.grade(context)
        assert result_low.passed is True  # 75% >= 60%

    @pytest.mark.asyncio
    async def test_weighted_scoring_mode(self):
        """Test weighted scoring with field_weights."""
        grader = JsonSchemaGrader({
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},      # weight: 2.0
                    "email": {"type": "string"},     # weight: 1.5
                    "phone": {"type": "string"},     # weight: 1.0
                },
                "required": ["name"],
            },
            "scoring_mode": "weighted",
            "field_weights": {
                "name": 2.0,
                "email": 1.5,
            },
            "required_weight": 1.0,  # Don't multiply required
            "pass_threshold": 0.5,
        })

        # Only phone is wrong
        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={
                "name": "Alice",
                "email": "alice@example.com",
                "phone": 12345,  # should be string
            }),
        )

        result = await grader.grade(context)
        # name (2.0) + email (1.5) = 3.5 earned
        # total = 2.0 + 1.5 + 1.0 = 4.5
        # score should be around 3.5/4.5 ≈ 0.78
        assert result.score > 0.7
        assert result.passed is True
        assert "score_breakdown" in result.details["field_compliance"]

    @pytest.mark.asyncio
    async def test_required_weight_multiplier(self):
        """Test required fields get weight multiplier."""
        grader = JsonSchemaGrader({
            "schema": {
                "type": "object",
                "properties": {
                    "required_field": {"type": "string"},
                    "optional_field": {"type": "string"},
                },
                "required": ["required_field"],
            },
            "scoring_mode": "partial",
            "required_weight": 3.0,  # required fields count 3x
        })

        # Missing required field
        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={
                "optional_field": "value",
                # required_field missing
            }),
        )

        result = await grader.grade(context)
        # optional (1.0) earned, required (3.0) failed
        # score = 1.0 / 4.0 = 0.25
        assert result.score < 0.3

    @pytest.mark.asyncio
    async def test_nested_field_scoring(self):
        """Test nested fields are included in scoring."""
        grader = JsonSchemaGrader({
            "schema": {
                "type": "object",
                "properties": {
                    "user": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "email": {"type": "string"},
                        },
                        "required": ["name"],
                    },
                },
                "required": ["user"],
            },
            "scoring_mode": "partial",
            "count_nested": True,
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={
                "user": {
                    "name": "Alice",
                    "email": 12345,  # wrong type
                },
            }),
        )

        result = await grader.grade(context)
        # Check nested fields are in results
        field_paths = [f["path"] for f in result.details.get("field_results", [])]
        assert "user" in field_paths or any("user" in p for p in field_paths)

    @pytest.mark.asyncio
    async def test_strict_mode_still_works(self):
        """Test strict scoring mode (original behavior) still works."""
        grader = JsonSchemaGrader({
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name", "age"],
            },
            "scoring_mode": "strict",  # default
        })

        # One field wrong
        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={
                "name": "Alice",
                "age": "not a number",
            }),
        )

        result = await grader.grade(context)
        # Strict mode: any error = score 0
        assert result.score == 0.0
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_field_level_diagnostics(self):
        """Test detailed field-level error reporting."""
        grader = JsonSchemaGrader({
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer", "minimum": 0},
                    "email": {"type": "string", "format": "email"},
                },
                "required": ["name", "age"],
            },
            "scoring_mode": "partial",
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={
                "name": "Alice",
                "age": -5,  # constraint violation
                # email missing but not required
            }),
        )

        result = await grader.grade(context)
        # Check error categorization
        assert "error_summary" in result.details
        error_summary = result.details["error_summary"]
        # Should have constraint error for age
        assert len(error_summary["constraint"]) > 0

    @pytest.mark.asyncio
    async def test_error_categorization(self):
        """Test errors are properly categorized."""
        grader = JsonSchemaGrader({
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "count": {"type": "integer"},
                },
                "required": ["name", "count", "missing_field"],
            },
            "scoring_mode": "partial",
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={
                "name": "Alice",
                "count": "not a number",  # type error
                # missing_field is missing
            }),
        )

        result = await grader.grade(context)
        error_summary = result.details["error_summary"]

        # Should have both missing and type errors
        assert len(error_summary["missing"]) > 0
        assert len(error_summary["type_error"]) > 0

    @pytest.mark.asyncio
    async def test_perfect_score_in_partial_mode(self):
        """Test perfect compliance gives score 1.0 in partial mode."""
        grader = JsonSchemaGrader({
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name", "age"],
            },
            "scoring_mode": "partial",
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={
                "name": "Alice",
                "age": 30,
            }),
        )

        result = await grader.grade(context)
        assert result.score == 1.0
        assert result.passed is True
        assert result.details["field_compliance"]["failed"] == 0

    @pytest.mark.asyncio
    async def test_reasoning_includes_compliance_info(self):
        """Test reasoning string includes compliance information."""
        grader = JsonSchemaGrader({
            "schema": {
                "type": "object",
                "properties": {
                    "a": {"type": "string"},
                    "b": {"type": "string"},
                },
                "required": ["a", "b"],
            },
            "scoring_mode": "partial",
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={"a": "value"}),  # b missing
        )

        result = await grader.grade(context)
        assert "1/2" in result.reasoning or "partial" in result.reasoning.lower()


# =============================================================================
# StructureCheckGrader Tests
# =============================================================================


class TestStructureCheckGrader:
    """Tests for StructureCheckGrader."""

    def test_registered(self):
        """Test grader is registered."""
        grader_cls = get_grader("structure_check")
        assert grader_cls is StructureCheckGrader

    @pytest.mark.asyncio
    async def test_valid_json(self):
        """Test valid JSON passes."""
        grader = StructureCheckGrader({
            "format": "json",
            "source": "output_data.content",
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={"content": '{"key": "value"}'}),
        )

        result = await grader.grade(context)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_invalid_json(self):
        """Test invalid JSON fails."""
        grader = StructureCheckGrader({
            "format": "json",
            "source": "output_data.content",
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={"content": "{invalid json}"}),
        )

        result = await grader.grade(context)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_valid_yaml(self):
        """Test valid YAML passes."""
        grader = StructureCheckGrader({
            "format": "yaml",
            "source": "output_data.content",
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={"content": "name: test\nversion: 1.0"}),
        )

        result = await grader.grade(context)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_required_keys(self):
        """Test required_keys validation."""
        grader = StructureCheckGrader({
            "format": "json",
            "source": "output_data.content",
            "required_keys": ["name", "version"],
        })

        # Missing version
        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={"content": '{"name": "test"}'}),
        )

        result = await grader.grade(context)
        assert result.passed is False
        assert "Missing required keys" in str(result.error)

    @pytest.mark.asyncio
    async def test_required_keys_present(self):
        """Test all required keys present passes."""
        grader = StructureCheckGrader({
            "format": "json",
            "source": "output_data.content",
            "required_keys": ["name", "version"],
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(
                output_data={"content": '{"name": "test", "version": "1.0"}'}
            ),
        )

        result = await grader.grade(context)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_valid_xml(self):
        """Test valid XML passes."""
        grader = StructureCheckGrader({
            "format": "xml",
            "source": "output_data.content",
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(
                output_data={"content": "<root><item>value</item></root>"}
            ),
        )

        result = await grader.grade(context)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_invalid_xml(self):
        """Test invalid XML fails."""
        grader = StructureCheckGrader({
            "format": "xml",
            "source": "output_data.content",
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={"content": "<root><unclosed>"}),
        )

        result = await grader.grade(context)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_xml_required_elements(self):
        """Test XML required_elements validation."""
        grader = StructureCheckGrader({
            "format": "xml",
            "source": "output_data.content",
            "required_elements": ["name", "version"],
        })

        # Missing version element
        context = GradeContext(
            prompt="test",
            outcome=Outcome(
                output_data={"content": "<root><name>test</name></root>"}
            ),
        )

        result = await grader.grade(context)
        assert result.passed is False
        assert "Missing required elements" in str(result.error)

    @pytest.mark.asyncio
    async def test_schema_validation(self):
        """Test schema validation for JSON."""
        grader = StructureCheckGrader({
            "format": "json",
            "source": "output_data.content",
            "schema": {
                "type": "object",
                "properties": {
                    "count": {"type": "integer"},
                },
                "required": ["count"],
            },
        })

        # Wrong type for count
        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={"content": '{"count": "not a number"}'}),
        )

        result = await grader.grade(context)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_unsupported_format_raises(self):
        """Test unsupported format raises error."""
        with pytest.raises(ValueError, match="Unsupported format"):
            StructureCheckGrader({"format": "unsupported"})

    @pytest.mark.asyncio
    async def test_extract_from_code_block(self):
        """Test extraction from code block."""
        grader = StructureCheckGrader({
            "format": "json",
            "source": "output_data.response",
            "extract_from_text": True,
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(
                output_data={
                    "response": 'Result:\n```json\n{"valid": true}\n```'
                }
            ),
        )

        result = await grader.grade(context)
        assert result.passed is True


# =============================================================================
# Artifact Source Tests (text_artifact / code_artifact)
# =============================================================================


class TestArtifactSourceExtraction:
    """Tests for extracting content from text_artifact and code_artifact sources."""

    @pytest.mark.asyncio
    async def test_text_artifact_source(self):
        """Test extraction from text_artifact source."""
        from compass.core.artifacts import TextArtifact

        grader = StructureCheckGrader({
            "format": "json",
            "source": "text_artifact",
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(
                artifacts=[TextArtifact(content='{"name": "test", "valid": true}')],
            ),
        )

        result = await grader.grade(context)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_text_artifact_source_alias(self):
        """Test 'text' as alias for 'text_artifact'."""
        from compass.core.artifacts import TextArtifact

        grader = StructureCheckGrader({
            "format": "json",
            "source": "text",
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(
                artifacts=[TextArtifact(content='{"status": "ok"}')],
            ),
        )

        result = await grader.grade(context)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_code_artifact_source(self):
        """Test extraction from code_artifact source."""
        from compass.core.artifacts import CodeArtifact, GeneratedFile

        grader = StructureCheckGrader({
            "format": "json",
            "source": "code_artifact",
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(
                artifacts=[
                    CodeArtifact(files=[
                        GeneratedFile(path="data.json", content='{"key": "value"}'),
                    ])
                ],
            ),
        )

        result = await grader.grade(context)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_code_artifact_source_alias(self):
        """Test 'code' as alias for 'code_artifact'."""
        from compass.core.artifacts import CodeArtifact, GeneratedFile

        grader = StructureCheckGrader({
            "format": "json",
            "source": "code",
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(
                artifacts=[
                    CodeArtifact(files=[
                        GeneratedFile(path="config.json", content='{"enabled": true}'),
                    ])
                ],
            ),
        )

        result = await grader.grade(context)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_code_artifact_multiple_files(self):
        """Test code_artifact with multiple files concatenates content."""
        from compass.core.artifacts import CodeArtifact, GeneratedFile

        grader = StructureCheckGrader({
            "format": "yaml",
            "source": "code_artifact",
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(
                artifacts=[
                    CodeArtifact(files=[
                        GeneratedFile(path="a.yaml", content="name: test"),
                        GeneratedFile(path="b.yaml", content="value: 123"),
                    ])
                ],
            ),
        )

        result = await grader.grade(context)
        # Concatenated content should be valid YAML
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_missing_text_artifact_fails(self):
        """Test missing text_artifact fails gracefully."""
        grader = StructureCheckGrader({
            "format": "json",
            "source": "text_artifact",
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={"some": "data"}),  # No text artifact
        )

        result = await grader.grade(context)
        assert result.passed is False
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_missing_code_artifact_fails(self):
        """Test missing code_artifact fails gracefully."""
        grader = StructureCheckGrader({
            "format": "json",
            "source": "code_artifact",
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={"some": "data"}),  # No code artifact
        )

        result = await grader.grade(context)
        assert result.passed is False
        assert result.score == 0.0
