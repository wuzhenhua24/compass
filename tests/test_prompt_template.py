"""Tests for RubricPromptTemplate."""

from __future__ import annotations

from compass.graders.model.prompt_template import RubricPromptTemplate


class TestRubricPromptTemplate:
    """Tests for the RubricPromptTemplate dataclass."""

    def test_from_config_empty(self):
        """Empty config produces a template with no structured sections."""
        template = RubricPromptTemplate.from_config({})

        assert template.role == ""
        assert template.scope_constraints == []
        assert template.verdict_rules == []
        assert template.has_structured_sections is False

    def test_from_config_full(self):
        """Full config populates all fields correctly."""
        config = {
            "role": "Expert evaluator",
            "scope_constraints": ["Only evaluate text", "Ignore style"],
            "verdict_rules": ["Garbled text means score < 0.5"],
        }
        template = RubricPromptTemplate.from_config(config)

        assert template.role == "Expert evaluator"
        assert template.scope_constraints == ["Only evaluate text", "Ignore style"]
        assert template.verdict_rules == ["Garbled text means score < 0.5"]
        assert template.has_structured_sections is True

    def test_from_config_partial_role_only(self):
        """Only role provided still counts as structured."""
        template = RubricPromptTemplate.from_config({"role": "Expert"})

        assert template.has_structured_sections is True
        assert template.role == "Expert"
        assert template.scope_constraints == []
        assert template.verdict_rules == []

    def test_from_config_partial_constraints_only(self):
        """Only scope_constraints provided still counts as structured."""
        template = RubricPromptTemplate.from_config(
            {"scope_constraints": ["Only evaluate text"]}
        )
        assert template.has_structured_sections is True

    def test_from_config_partial_verdict_rules_only(self):
        """Only verdict_rules provided still counts as structured."""
        template = RubricPromptTemplate.from_config(
            {"verdict_rules": ["Score must be 0 if missing"]}
        )
        assert template.has_structured_sections is True

    def test_from_config_ignores_unrelated_keys(self):
        """Unrelated config keys are ignored."""
        template = RubricPromptTemplate.from_config({
            "model": "gpt-4o",
            "criteria": [{"name": "test"}],
            "role": "Expert",
        })
        assert template.role == "Expert"
        assert template.has_structured_sections is True

    def test_render_role_section(self):
        """Rendered output includes <role> tag."""
        template = RubricPromptTemplate(role="Expert text evaluator")
        output = template.render(criteria_text="- accuracy", content="test content")

        assert "<role>" in output
        assert "Expert text evaluator" in output
        assert "</role>" in output

    def test_render_scope_constraints_section(self):
        """Rendered output includes <scope_constraints> with list items."""
        template = RubricPromptTemplate(
            scope_constraints=["Only evaluate text", "Ignore aesthetics"]
        )
        output = template.render(criteria_text="- accuracy", content="test content")

        assert "<scope_constraints>" in output
        assert "- Only evaluate text" in output
        assert "- Ignore aesthetics" in output
        assert "</scope_constraints>" in output

    def test_render_verdict_rules_section(self):
        """Rendered output includes <verdict_rules> with list items."""
        template = RubricPromptTemplate(
            verdict_rules=["Garbled text means < 0.5", "Correct text means >= 0.8"]
        )
        output = template.render(criteria_text="- accuracy", content="test content")

        assert "<verdict_rules>" in output
        assert "- Garbled text means < 0.5" in output
        assert "- Correct text means >= 0.8" in output
        assert "</verdict_rules>" in output

    def test_render_empty_sections_omitted(self):
        """Empty sections do not appear in the rendered output."""
        template = RubricPromptTemplate(role="Expert")
        output = template.render(criteria_text="- accuracy", content="test content")

        assert "<role>" in output
        assert "<scope_constraints>" not in output
        assert "<verdict_rules>" not in output

    def test_render_full_prompt(self):
        """Full template renders all sections in correct order."""
        template = RubricPromptTemplate(
            role="Expert evaluator",
            scope_constraints=["Only text accuracy"],
            verdict_rules=["Score 0 if garbled"],
        )
        output = template.render(
            criteria_text="- text_accuracy (weight: 3.0)",
            content="Hello world",
            context_info="Original prompt: render hello world",
        )

        # All sections present
        assert "<role>" in output
        assert "<scope_constraints>" in output
        assert "<metrics_and_scoring>" in output
        assert "<verdict_rules>" in output
        assert "<context>" in output
        assert "<content>" in output
        assert "<instructions>" in output
        assert "<output_schema>" in output

        # Correct ordering: role before scope_constraints before metrics
        role_pos = output.index("<role>")
        scope_pos = output.index("<scope_constraints>")
        metrics_pos = output.index("<metrics_and_scoring>")
        verdict_pos = output.index("<verdict_rules>")
        content_pos = output.index("<content>")
        instructions_pos = output.index("<instructions>")
        output_schema_pos = output.index("<output_schema>")

        assert role_pos < scope_pos < metrics_pos < verdict_pos < content_pos
        assert content_pos < instructions_pos < output_schema_pos

    def test_render_includes_context_when_provided(self):
        """Context info appears in output when provided."""
        template = RubricPromptTemplate(role="Expert")
        output = template.render(
            criteria_text="- accuracy",
            content="test",
            context_info="Some context",
        )
        assert "<context>" in output
        assert "Some context" in output

    def test_render_omits_context_when_empty(self):
        """Context section is omitted when context_info is empty."""
        template = RubricPromptTemplate(role="Expert")
        output = template.render(criteria_text="- accuracy", content="test")
        assert "<context>" not in output

    def test_render_system_prompt_with_template(self):
        """Structured system prompt includes XML tags."""
        template = RubricPromptTemplate(
            role="Expert image evaluator",
            scope_constraints=["Only evaluate composition"],
            verdict_rules=["Missing subject means 0.0"],
        )
        output = template.render_system_prompt()

        assert "<role>" in output
        assert "Expert image evaluator" in output
        assert "<scope_constraints>" in output
        assert "- Only evaluate composition" in output
        assert "<verdict_rules>" in output
        assert "- Missing subject means 0.0" in output

    def test_render_system_prompt_default(self):
        """Empty template returns default system prompt text."""
        template = RubricPromptTemplate()
        output = template.render_system_prompt()

        assert "<role>" not in output
        assert "image quality evaluator" in output

    def test_render_user_prompt(self):
        """User prompt contains criteria and output format."""
        template = RubricPromptTemplate(role="Expert")
        output = template.render_user_prompt(
            criteria_text="- Clear focal point\n- Good composition",
            prompt="A sunset photo",
        )

        assert "<context>" in output
        assert "A sunset photo" in output
        assert "<metrics_and_scoring>" in output
        assert "- Clear focal point" in output
        assert "<output_schema>" in output
        assert "JSON format" in output

    def test_render_user_prompt_no_prompt(self):
        """User prompt omits context section when no prompt provided."""
        template = RubricPromptTemplate(role="Expert")
        output = template.render_user_prompt(
            criteria_text="- accuracy",
        )

        assert "<context>" not in output
        assert "<metrics_and_scoring>" in output
