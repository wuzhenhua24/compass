"""Tests for style convention graders."""

from __future__ import annotations

import pytest

from compass.core.transcript import Outcome, Transcript
from compass.graders.base import GradeContext
from compass.graders.code.common.style import StyleConventionGrader


class TestStyleConventionGrader:
    """Tests for StyleConventionGrader."""

    @pytest.fixture
    def make_context(self):
        """Factory to create GradeContext with text content."""

        def _make(content: str) -> GradeContext:
            transcript = Transcript(task_id="test-task", trial_id="test-trial")
            outcome = Outcome(output_data=content)
            return GradeContext(
                transcript=transcript,
                outcome=outcome,
            )

        return _make

    # ===================================================================
    # Basic tests
    # ===================================================================

    @pytest.mark.asyncio
    async def test_no_checks_configured_passes(self, make_context):
        """When no checks are configured, should pass by default."""
        context = make_context("Any content here")

        grader = StyleConventionGrader({})
        result = await grader.grade(context)

        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_missing_content_fails(self):
        """When content is missing, should fail."""
        context = GradeContext(
            transcript=Transcript(task_id="test", trial_id="test"),
            outcome=Outcome(),
        )

        grader = StyleConventionGrader({"required_phrases": ["hello"]})
        result = await grader.grade(context)

        assert result.passed is False
        assert "No content found" in result.error

    # ===================================================================
    # Section checks
    # ===================================================================

    @pytest.mark.asyncio
    async def test_required_sections_present(self, make_context):
        """Required sections present should pass."""
        content = """## Summary
This is the summary.

## Details
These are the details.

## Recommendations
- Item 1
- Item 2
"""
        context = make_context(content)

        grader = StyleConventionGrader({
            "required_sections": ["Summary", "Details", "Recommendations"],
        })
        result = await grader.grade(context)

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_required_sections_missing(self, make_context):
        """Missing required sections should fail."""
        content = """## Summary
This is the summary.

## Details
These are the details.
"""
        context = make_context(content)

        grader = StyleConventionGrader({
            "required_sections": ["Summary", "Details", "Recommendations"],
        })
        result = await grader.grade(context)

        assert result.passed is False
        assert any(v["type"] == "missing_section" for v in result.details["violations"])

    @pytest.mark.asyncio
    async def test_section_order_enforced(self, make_context):
        """Sections in wrong order should fail when section_order=True."""
        content = """## Details
These are the details.

## Summary
This is the summary.
"""
        context = make_context(content)

        grader = StyleConventionGrader({
            "required_sections": ["Summary", "Details"],
            "section_order": True,
        })
        result = await grader.grade(context)

        assert result.passed is False
        assert any(v["type"] == "section_order" for v in result.details["violations"])

    @pytest.mark.asyncio
    async def test_section_order_not_enforced(self, make_context):
        """Sections in wrong order should pass when section_order=False."""
        content = """## Details
These are the details.

## Summary
This is the summary.
"""
        context = make_context(content)

        grader = StyleConventionGrader({
            "required_sections": ["Summary", "Details"],
            "section_order": False,
        })
        result = await grader.grade(context)

        assert result.passed is True

    # ===================================================================
    # Phrase checks
    # ===================================================================

    @pytest.mark.asyncio
    async def test_required_phrases_present(self, make_context):
        """Required phrases present should pass."""
        content = "Based on the analysis, we recommend taking action immediately."
        context = make_context(content)

        grader = StyleConventionGrader({
            "required_phrases": ["Based on the analysis", "recommend"],
        })
        result = await grader.grade(context)

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_required_phrases_missing(self, make_context):
        """Missing required phrases should fail."""
        content = "Here is some random content without the required text."
        context = make_context(content)

        grader = StyleConventionGrader({
            "required_phrases": ["Based on the analysis"],
        })
        result = await grader.grade(context)

        assert result.passed is False
        assert any(v["type"] == "missing_phrase" for v in result.details["violations"])

    @pytest.mark.asyncio
    async def test_forbidden_phrases_absent(self, make_context):
        """Forbidden phrases absent should pass."""
        content = "Here is a confident and clear response."
        context = make_context(content)

        grader = StyleConventionGrader({
            "forbidden_phrases": ["I don't know", "I'm not sure"],
        })
        result = await grader.grade(context)

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_forbidden_phrases_present(self, make_context):
        """Forbidden phrases present should fail."""
        content = "I'm not sure about this, but I think it might work."
        context = make_context(content)

        grader = StyleConventionGrader({
            "forbidden_phrases": ["I don't know", "I'm not sure"],
        })
        result = await grader.grade(context)

        assert result.passed is False
        assert any(v["type"] == "forbidden_phrase" for v in result.details["violations"])

    @pytest.mark.asyncio
    async def test_case_insensitive_phrase_matching(self, make_context):
        """Case insensitive matching should work."""
        content = "BASED ON THE ANALYSIS, we proceed."
        context = make_context(content)

        grader = StyleConventionGrader({
            "required_phrases": ["based on the analysis"],
            "case_sensitive": False,
        })
        result = await grader.grade(context)

        assert result.passed is True

    # ===================================================================
    # Pattern checks
    # ===================================================================

    @pytest.mark.asyncio
    async def test_required_patterns_match(self, make_context):
        """Required patterns matching should pass."""
        content = "Error code: ERR-12345\nStatus: completed"
        context = make_context(content)

        grader = StyleConventionGrader({
            "required_patterns": [r"ERR-\d{5}", r"Status:\s+\w+"],
        })
        result = await grader.grade(context)

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_required_patterns_not_match(self, make_context):
        """Required patterns not matching should fail."""
        content = "Error code: invalid\nStatus: completed"
        context = make_context(content)

        grader = StyleConventionGrader({
            "required_patterns": [r"ERR-\d{5}"],
        })
        result = await grader.grade(context)

        assert result.passed is False
        assert any(v["type"] == "missing_pattern" for v in result.details["violations"])

    @pytest.mark.asyncio
    async def test_forbidden_patterns_absent(self, make_context):
        """Forbidden patterns absent should pass."""
        content = "Clean content without sensitive data."
        context = make_context(content)

        grader = StyleConventionGrader({
            "forbidden_patterns": [r"\b\d{3}-\d{2}-\d{4}\b"],  # SSN pattern
        })
        result = await grader.grade(context)

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_forbidden_patterns_present(self, make_context):
        """Forbidden patterns present should fail."""
        content = "User SSN: 123-45-6789"
        context = make_context(content)

        grader = StyleConventionGrader({
            "forbidden_patterns": [r"\b\d{3}-\d{2}-\d{4}\b"],
        })
        result = await grader.grade(context)

        assert result.passed is False
        assert any(v["type"] == "forbidden_pattern" for v in result.details["violations"])

    # ===================================================================
    # Length constraint checks
    # ===================================================================

    @pytest.mark.asyncio
    async def test_length_within_constraints(self, make_context):
        """Content within length constraints should pass."""
        content = "This is a response with exactly the right amount of words."
        context = make_context(content)

        grader = StyleConventionGrader({
            "min_words": 5,
            "max_words": 20,
        })
        result = await grader.grade(context)

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_too_short(self, make_context):
        """Content too short should fail."""
        content = "Short."
        context = make_context(content)

        grader = StyleConventionGrader({
            "min_words": 10,
        })
        result = await grader.grade(context)

        assert result.passed is False
        assert any(v["type"] == "too_few_words" for v in result.details["violations"])

    @pytest.mark.asyncio
    async def test_too_long(self, make_context):
        """Content too long should fail."""
        content = " ".join(["word"] * 100)
        context = make_context(content)

        grader = StyleConventionGrader({
            "max_words": 50,
        })
        result = await grader.grade(context)

        assert result.passed is False
        assert any(v["type"] == "too_many_words" for v in result.details["violations"])

    @pytest.mark.asyncio
    async def test_min_chars_alias(self, make_context):
        """min_chars should work as alias for min_length (used by expected config)."""
        content = "Short"  # 5 chars
        context = make_context(content)

        grader = StyleConventionGrader({
            "min_chars": 10,  # Alias used by scenario.py expected config
        })
        result = await grader.grade(context)

        assert result.passed is False
        assert any(v["type"] == "too_short" for v in result.details["violations"])

    @pytest.mark.asyncio
    async def test_max_chars_alias(self, make_context):
        """max_chars should work as alias for max_length (used by expected config)."""
        content = "This is a longer response"  # 25 chars
        context = make_context(content)

        grader = StyleConventionGrader({
            "max_chars": 10,  # Alias used by scenario.py expected config
        })
        result = await grader.grade(context)

        assert result.passed is False
        assert any(v["type"] == "too_long" for v in result.details["violations"])

    @pytest.mark.asyncio
    async def test_line_count_constraints(self, make_context):
        """Line count constraints should be enforced."""
        content = "Line 1\nLine 2\nLine 3"
        context = make_context(content)

        # Within limits
        grader = StyleConventionGrader({"min_lines": 2, "max_lines": 5})
        result = await grader.grade(context)
        assert result.passed is True

        # Too few lines
        grader = StyleConventionGrader({"min_lines": 5})
        result = await grader.grade(context)
        assert result.passed is False

    # ===================================================================
    # Markdown checks
    # ===================================================================

    @pytest.mark.asyncio
    async def test_markdown_code_blocks_required(self, make_context):
        """Required code blocks should be enforced."""
        content_with_code = """Here is some code:
```python
print("hello")
```
"""
        content_without_code = "Here is some text without code blocks."

        grader = StyleConventionGrader({
            "markdown_checks": {"require_code_blocks": True},
        })

        # With code block
        context = make_context(content_with_code)
        result = await grader.grade(context)
        assert result.passed is True

        # Without code block
        context = make_context(content_without_code)
        result = await grader.grade(context)
        assert result.passed is False
        assert any(v["type"] == "missing_code_block" for v in result.details["violations"])

    @pytest.mark.asyncio
    async def test_markdown_heading_level_limit(self, make_context):
        """Heading level limit should be enforced."""
        content = """# Title
## Section
### Subsection
#### Too Deep
"""
        context = make_context(content)

        grader = StyleConventionGrader({
            "markdown_checks": {"max_heading_level": 3},
        })
        result = await grader.grade(context)

        assert result.passed is False
        assert any(v["type"] == "heading_too_deep" for v in result.details["violations"])

    @pytest.mark.asyncio
    async def test_markdown_lists_required(self, make_context):
        """Required lists should be enforced."""
        content_with_list = """Summary:
- Item 1
- Item 2
"""
        content_without_list = "Summary: Just plain text."

        grader = StyleConventionGrader({
            "markdown_checks": {"require_lists": True},
        })

        # With list
        context = make_context(content_with_list)
        result = await grader.grade(context)
        assert result.passed is True

        # Without list
        context = make_context(content_without_list)
        result = await grader.grade(context)
        assert result.passed is False

    # ===================================================================
    # Naming convention checks
    # ===================================================================

    @pytest.mark.asyncio
    async def test_function_naming_snake_case(self, make_context):
        """Function naming convention should be enforced."""
        good_code = """
def calculate_total():
    pass

def get_user_data():
    pass
"""
        bad_code = """
def calculateTotal():
    pass

def GetUserData():
    pass
"""
        grader = StyleConventionGrader({
            "naming_conventions": {"functions": "snake_case"},
        })

        # Good naming
        context = make_context(good_code)
        result = await grader.grade(context)
        assert result.passed is True

        # Bad naming
        context = make_context(bad_code)
        result = await grader.grade(context)
        assert result.passed is False
        assert any(v["type"] == "function_naming" for v in result.details["violations"])

    @pytest.mark.asyncio
    async def test_class_naming_pascal_case(self, make_context):
        """Class naming convention should be enforced."""
        good_code = """
class UserManager:
    pass

class DataProcessor:
    pass
"""
        bad_code = """
class user_manager:
    pass

class dataProcessor:
    pass
"""
        grader = StyleConventionGrader({
            "naming_conventions": {"classes": "PascalCase"},
        })

        # Good naming
        context = make_context(good_code)
        result = await grader.grade(context)
        assert result.passed is True

        # Bad naming
        context = make_context(bad_code)
        result = await grader.grade(context)
        assert result.passed is False
        assert any(v["type"] == "class_naming" for v in result.details["violations"])

    # ===================================================================
    # Combined checks
    # ===================================================================

    @pytest.mark.asyncio
    async def test_multiple_checks_all_pass(self, make_context):
        """Multiple checks all passing should pass."""
        content = """## Summary
Based on the analysis, here are the findings.

## Details
- Finding 1
- Finding 2

## Recommendations
We recommend taking action.
"""
        context = make_context(content)

        grader = StyleConventionGrader({
            "required_sections": ["Summary", "Details", "Recommendations"],
            "required_phrases": ["Based on the analysis"],
            "forbidden_phrases": ["I don't know"],
            "min_words": 10,
            "markdown_checks": {"require_lists": True},
        })
        result = await grader.grade(context)

        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_multiple_checks_partial_pass(self, make_context):
        """Multiple checks with some failures should have partial score."""
        content = """## Summary
I'm not sure, but here are some thoughts.
"""
        context = make_context(content)

        grader = StyleConventionGrader({
            "required_sections": ["Summary"],  # Pass
            "forbidden_phrases": ["I'm not sure"],  # Fail
            "min_words": 5,  # Pass
        })
        result = await grader.grade(context)

        assert result.passed is False
        assert 0 < result.score < 1.0  # Partial score

    @pytest.mark.asyncio
    async def test_content_stats_in_details(self, make_context):
        """Content statistics should be included in details."""
        content = "Word one two three four.\nSecond line."
        context = make_context(content)

        grader = StyleConventionGrader({})
        result = await grader.grade(context)

        assert "content_stats" in result.details
        # Word one two three four. Second line.
        assert result.details["content_stats"]["words"] == 7
        assert result.details["content_stats"]["lines"] == 2
