"""Tests for VLMJudgeGrader structured prompt integration."""

from __future__ import annotations

from typing import Any

from compass.graders.domains.image.vlm import VLMJudgeGrader


class TestVLMStructuredPrompt:
    """Tests for VLMJudgeGrader with RubricPromptTemplate."""

    def test_init_with_structured_config(self):
        """Template is created when structured fields are provided."""
        config: dict[str, Any] = {
            "role": "Expert image composition evaluator",
            "scope_constraints": ["Only evaluate composition, NOT text"],
            "verdict_rules": ["Missing subject means score 0.0"],
            "criteria": ["Clear focal point", "Good composition"],
        }
        grader = VLMJudgeGrader(config)

        assert grader._template.has_structured_sections is True
        assert grader._template.role == "Expert image composition evaluator"
        assert len(grader._template.scope_constraints) == 1
        assert len(grader._template.verdict_rules) == 1

    def test_init_without_structured_config(self):
        """Template without structured fields has no sections."""
        grader = VLMJudgeGrader({
            "criteria": ["Image is clear"],
        })
        assert grader._template.has_structured_sections is False

    def test_vlm_system_prompt_uses_template(self):
        """When structured config is set, render_system_prompt produces XML tags."""
        config: dict[str, Any] = {
            "role": "Expert image evaluator",
            "scope_constraints": ["Only composition"],
            "verdict_rules": ["Missing subject = 0.0"],
        }
        grader = VLMJudgeGrader(config)

        system_prompt = grader._template.render_system_prompt()

        assert "<role>" in system_prompt
        assert "Expert image evaluator" in system_prompt
        assert "<scope_constraints>" in system_prompt
        assert "<verdict_rules>" in system_prompt

    def test_vlm_backward_compatible(self):
        """Without new fields, behavior is unchanged."""
        grader = VLMJudgeGrader({
            "criteria": ["Image is clear"],
            "threshold": 0.8,
        })

        assert grader._template.has_structured_sections is False
        assert grader.system_prompt == VLMJudgeGrader.DEFAULT_SYSTEM_PROMPT
        assert grader.threshold == 0.8

    def test_vlm_custom_system_prompt_preserved(self):
        """User-provided system_prompt takes priority over template."""
        config: dict[str, Any] = {
            "system_prompt": "My custom system prompt",
            "role": "Expert evaluator",  # Also has structured fields
            "criteria": ["Image is clear"],
        }
        grader = VLMJudgeGrader(config)

        # Template has structured sections
        assert grader._template.has_structured_sections is True
        # But user's custom system_prompt is stored
        assert grader.system_prompt == "My custom system prompt"
        # The custom system_prompt != DEFAULT_SYSTEM_PROMPT, so the
        # _call_vlm logic should use the custom one, not the template.
        # We verify this by checking the condition that would be used:
        assert grader.system_prompt != VLMJudgeGrader.DEFAULT_SYSTEM_PROMPT

    def test_vlm_default_system_prompt_unchanged(self):
        """Default system prompt text is preserved."""
        grader = VLMJudgeGrader({})
        assert "image quality evaluator" in grader.system_prompt
        assert "PASS" in grader.system_prompt
        assert "FAIL" in grader.system_prompt
