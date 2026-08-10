"""Style convention graders for evaluating output formatting and conventions.

Provides graders for validating that agent outputs follow specified style
conventions, templates, naming rules, and formatting requirements.
"""

from __future__ import annotations

import re
from typing import Any

from compass.graders.base import CodeGrader, GradeContext, GradeResult, GraderScope, GraderType
from compass.graders.registry import register_grader


def _extract_text_content(context: GradeContext, source: str) -> str | None:
    """Extract text content from context based on source specification.

    Args:
        context: Grade context.
        source: Source specification:
            - "output_data" or "output": from outcome.output_data
            - "text" or "text_artifact": from TextArtifact.content
            - "code" or "code_artifact": from CodeArtifact files
            - "metadata.<key>": from outcome.metadata
            - "output_data.<key>": from specific field in output_data

    Returns:
        Extracted content as string, or None if not found.
    """
    if not context.outcome:
        return None

    if source in ("output_data", "output"):
        data = context.outcome.output_data
        if isinstance(data, str):
            return data if data else None
        elif isinstance(data, dict):
            if not data:  # Empty dict
                return None
            # Try common text fields
            for key in ("text", "content", "response", "output", "message"):
                if key in data and isinstance(data[key], str):
                    return data[key]
            # Return JSON string representation
            import json
            return json.dumps(data, ensure_ascii=False, indent=2)
        return str(data) if data else None

    if source in ("text", "text_artifact"):
        artifact = context.text_artifact
        if artifact and hasattr(artifact, "content"):
            return artifact.content
        return None

    if source in ("code", "code_artifact"):
        artifact = context.code_artifact
        if artifact and hasattr(artifact, "files"):
            contents = []
            for f in artifact.files:
                if hasattr(f, "content"):
                    contents.append(f.content)
            return "\n".join(contents)
        return None

    if source.startswith("metadata."):
        key = source[9:]
        value = context.outcome.metadata.get(key)
        return str(value) if value is not None else None

    if source.startswith("output_data."):
        key = source[12:]
        data = context.outcome.output_data
        if isinstance(data, dict):
            value = data.get(key)
            return str(value) if value is not None else None

    return None


@register_grader("style_convention")
class StyleConventionGrader(CodeGrader):
    """Validate that output follows specified style conventions.

    Scope: OUTCOME - evaluates the format and style of agent output.

    Checks multiple style dimensions:
    1. Template sections - required sections in specific order
    2. Required/forbidden phrases - must/must-not contain
    3. Custom regex patterns - flexible pattern matching
    4. Length constraints - character/word/line limits
    5. Markdown formatting - headings, lists, code blocks
    6. Naming conventions - for code identifiers

    Config:
        source: str — Where to get content from (default "output_data").

        # Template checks
        required_sections: list[str] — Section headers that must appear.
        section_order: bool — Whether sections must appear in order (default True).
        section_pattern: str — Regex pattern for section headers (default "^##?#?\\s+").

        # Phrase checks
        required_phrases: list[str] — Phrases that must appear in output.
        forbidden_phrases: list[str] — Phrases that must NOT appear.
        case_sensitive: bool — Case sensitivity for phrase matching (default False).

        # Pattern checks
        required_patterns: list[str] — Regex patterns that must match.
        forbidden_patterns: list[str] — Regex patterns that must NOT match.

        # Length constraints
        min_length: int — Minimum character count (default 0).
        max_length: int — Maximum character count (default unlimited).
        min_words: int — Minimum word count (default 0).
        max_words: int — Maximum word count (default unlimited).
        min_lines: int — Minimum line count (default 0).
        max_lines: int — Maximum line count (default unlimited).

        # Markdown checks
        markdown_checks: dict — Markdown formatting checks:
            - require_code_blocks: bool — Must contain code blocks
            - max_heading_level: int — Deepest heading level allowed (1-6)
            - require_lists: bool — Must contain bullet/numbered lists
            - no_bare_urls: bool — URLs must be in markdown link format

        # Naming convention checks (for code)
        naming_conventions: dict — Naming convention checks:
            - functions: str — "snake_case", "camelCase", "PascalCase"
            - variables: str — "snake_case", "camelCase"
            - classes: str — "PascalCase"
            - constants: str — "UPPER_SNAKE_CASE"

    Example YAML:
        graders:
          - name: style_convention
            config:
              source: output_data
              required_sections:
                - "Summary"
                - "Details"
                - "Recommendations"
              section_order: true
              required_phrases:
                - "Based on the analysis"
              forbidden_phrases:
                - "I don't know"
                - "I'm not sure"
              min_words: 50
              max_words: 500
              markdown_checks:
                require_code_blocks: true
                max_heading_level: 3
    """

    name = "style_convention"
    grader_type = GraderType.CODE
    grader_scope = GraderScope.OUTCOME

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.source = self.config.get("source", "output_data")

        # Template checks
        self.required_sections = self.config.get("required_sections", [])
        self.section_order = self.config.get("section_order", True)
        self.section_pattern = self.config.get("section_pattern", r"^##?#?\s+(.+)$")

        # Phrase checks
        self.required_phrases = self.config.get("required_phrases", [])
        self.forbidden_phrases = self.config.get("forbidden_phrases", [])
        self.case_sensitive = self.config.get("case_sensitive", False)

        # Pattern checks
        self.required_patterns = self.config.get("required_patterns", [])
        self.forbidden_patterns = self.config.get("forbidden_patterns", [])

        # Length constraints (support both min_length/max_length and min_chars/max_chars)
        self.min_length = self.config.get("min_length") or self.config.get("min_chars", 0)
        # 0 = unlimited
        self.max_length = self.config.get("max_length") or self.config.get("max_chars", 0)
        self.min_words = self.config.get("min_words", 0)
        self.max_words = self.config.get("max_words", 0)
        self.min_lines = self.config.get("min_lines", 0)
        self.max_lines = self.config.get("max_lines", 0)

        # Markdown checks
        self.markdown_checks = self.config.get("markdown_checks", {})

        # Naming convention checks
        self.naming_conventions = self.config.get("naming_conventions", {})

    async def grade(self, context: GradeContext) -> GradeResult:
        """Validate style conventions."""
        # Extract content
        content = _extract_text_content(context, self.source)
        if content is None:
            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=False,
                score=0.0,
                error=f"No content found at source: {self.source}",
            )

        violations = []
        checks_passed = []
        total_checks = 0

        # 1. Template section checks
        if self.required_sections:
            section_result = self._check_sections(content)
            total_checks += 1
            if section_result["passed"]:
                checks_passed.append("sections")
            else:
                violations.extend(section_result["violations"])

        # 2. Required phrase checks
        if self.required_phrases:
            phrase_result = self._check_required_phrases(content)
            total_checks += 1
            if phrase_result["passed"]:
                checks_passed.append("required_phrases")
            else:
                violations.extend(phrase_result["violations"])

        # 3. Forbidden phrase checks
        if self.forbidden_phrases:
            forbidden_result = self._check_forbidden_phrases(content)
            total_checks += 1
            if forbidden_result["passed"]:
                checks_passed.append("forbidden_phrases")
            else:
                violations.extend(forbidden_result["violations"])

        # 4. Required pattern checks
        if self.required_patterns:
            pattern_result = self._check_required_patterns(content)
            total_checks += 1
            if pattern_result["passed"]:
                checks_passed.append("required_patterns")
            else:
                violations.extend(pattern_result["violations"])

        # 5. Forbidden pattern checks
        if self.forbidden_patterns:
            forbidden_pattern_result = self._check_forbidden_patterns(content)
            total_checks += 1
            if forbidden_pattern_result["passed"]:
                checks_passed.append("forbidden_patterns")
            else:
                violations.extend(forbidden_pattern_result["violations"])

        # 6. Length constraint checks
        length_result = self._check_length_constraints(content)
        if length_result["checks_performed"] > 0:
            total_checks += 1
            if length_result["passed"]:
                checks_passed.append("length")
            else:
                violations.extend(length_result["violations"])

        # 7. Markdown checks
        if self.markdown_checks:
            markdown_result = self._check_markdown(content)
            total_checks += 1
            if markdown_result["passed"]:
                checks_passed.append("markdown")
            else:
                violations.extend(markdown_result["violations"])

        # 8. Naming convention checks
        if self.naming_conventions:
            naming_result = self._check_naming_conventions(content)
            total_checks += 1
            if naming_result["passed"]:
                checks_passed.append("naming")
            else:
                violations.extend(naming_result["violations"])

        # Calculate score
        if total_checks == 0:
            # No checks configured, pass by default
            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=True,
                score=1.0,
                details={
                    "message": "No style checks configured",
                    "content_stats": {
                        "length": len(content),
                        "words": len(content.split()),
                        "lines": len(content.splitlines()),
                    },
                },
            )

        score = len(checks_passed) / total_checks
        passed = len(violations) == 0

        # Extract failure tags from violation types
        failure_tags = list({v["type"] for v in violations}) if violations else []

        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=passed,
            score=score,
            details={
                "total_checks": total_checks,
                "checks_passed": checks_passed,
                "violation_count": len(violations),
                "violations": violations[:20],  # Limit to 20
                "content_stats": {
                    "length": len(content),
                    "words": len(content.split()),
                    "lines": len(content.splitlines()),
                },
            },
            failure_tags=failure_tags,
            reasoning=self._build_reasoning(violations, checks_passed, total_checks),
        )

    def _check_sections(self, content: str) -> dict[str, Any]:
        """Check for required sections."""
        violations = []

        # Find all sections in content
        section_regex = re.compile(self.section_pattern, re.MULTILINE)
        found_sections = section_regex.findall(content)

        # Normalize section names for comparison
        def normalize(s: str) -> str:
            return s.strip().lower() if not self.case_sensitive else s.strip()

        found_normalized = [normalize(s) for s in found_sections]

        # Check each required section exists
        missing_sections = []
        found_indices = []
        for req in self.required_sections:
            req_norm = normalize(req)
            if req_norm in found_normalized:
                found_indices.append(found_normalized.index(req_norm))
            else:
                missing_sections.append(req)

        if missing_sections:
            violations.append({
                "type": "missing_section",
                "message": f"Missing required sections: {missing_sections}",
                "severity": "high",
            })

        # Check section order if required
        if self.section_order and len(found_indices) > 1:
            if found_indices != sorted(found_indices):
                violations.append({
                    "type": "section_order",
                    "message": (
                        "Sections not in required order. "
                        f"Expected: {self.required_sections}"
                    ),
                    "severity": "medium",
                })

        return {
            "passed": len(violations) == 0,
            "violations": violations,
            "found_sections": found_sections,
        }

    def _check_required_phrases(self, content: str) -> dict[str, Any]:
        """Check for required phrases."""
        violations = []
        check_content = content if self.case_sensitive else content.lower()

        missing = []
        for phrase in self.required_phrases:
            check_phrase = phrase if self.case_sensitive else phrase.lower()
            if check_phrase not in check_content:
                missing.append(phrase)

        if missing:
            violations.append({
                "type": "missing_phrase",
                "message": f"Missing required phrases: {missing}",
                "severity": "medium",
            })

        return {"passed": len(violations) == 0, "violations": violations}

    def _check_forbidden_phrases(self, content: str) -> dict[str, Any]:
        """Check for forbidden phrases."""
        violations = []
        check_content = content if self.case_sensitive else content.lower()

        found = []
        for phrase in self.forbidden_phrases:
            check_phrase = phrase if self.case_sensitive else phrase.lower()
            if check_phrase in check_content:
                found.append(phrase)

        if found:
            violations.append({
                "type": "forbidden_phrase",
                "message": f"Contains forbidden phrases: {found}",
                "severity": "high",
            })

        return {"passed": len(violations) == 0, "violations": violations}

    def _check_required_patterns(self, content: str) -> dict[str, Any]:
        """Check for required regex patterns."""
        violations = []

        missing = []
        for pattern in self.required_patterns:
            try:
                if not re.search(pattern, content, re.MULTILINE):
                    missing.append(pattern)
            except re.error as e:
                violations.append({
                    "type": "invalid_pattern",
                    "message": f"Invalid regex pattern '{pattern}': {e}",
                    "severity": "low",
                })

        if missing:
            violations.append({
                "type": "missing_pattern",
                "message": f"Missing required patterns: {missing}",
                "severity": "medium",
            })

        return {"passed": len(violations) == 0, "violations": violations}

    def _check_forbidden_patterns(self, content: str) -> dict[str, Any]:
        """Check for forbidden regex patterns."""
        violations = []

        found = []
        for pattern in self.forbidden_patterns:
            try:
                match = re.search(pattern, content, re.MULTILINE)
                if match:
                    found.append({
                        "pattern": pattern,
                        "match": match.group()[:50],  # Truncate match
                    })
            except re.error:
                pass  # Skip invalid patterns

        if found:
            violations.append({
                "type": "forbidden_pattern",
                "message": f"Contains forbidden patterns: {[f['pattern'] for f in found]}",
                "details": found,
                "severity": "high",
            })

        return {"passed": len(violations) == 0, "violations": violations}

    def _check_length_constraints(self, content: str) -> dict[str, Any]:
        """Check length constraints."""
        violations = []
        checks_performed = 0

        char_count = len(content)
        word_count = len(content.split())
        line_count = len(content.splitlines())

        # Character length
        if self.min_length > 0:
            checks_performed += 1
            if char_count < self.min_length:
                violations.append({
                    "type": "too_short",
                    "message": f"Content too short: {char_count} chars < {self.min_length} min",
                    "severity": "medium",
                })

        if self.max_length > 0:
            checks_performed += 1
            if char_count > self.max_length:
                violations.append({
                    "type": "too_long",
                    "message": f"Content too long: {char_count} chars > {self.max_length} max",
                    "severity": "medium",
                })

        # Word count
        if self.min_words > 0:
            checks_performed += 1
            if word_count < self.min_words:
                violations.append({
                    "type": "too_few_words",
                    "message": f"Too few words: {word_count} < {self.min_words} min",
                    "severity": "medium",
                })

        if self.max_words > 0:
            checks_performed += 1
            if word_count > self.max_words:
                violations.append({
                    "type": "too_many_words",
                    "message": f"Too many words: {word_count} > {self.max_words} max",
                    "severity": "medium",
                })

        # Line count
        if self.min_lines > 0:
            checks_performed += 1
            if line_count < self.min_lines:
                violations.append({
                    "type": "too_few_lines",
                    "message": f"Too few lines: {line_count} < {self.min_lines} min",
                    "severity": "low",
                })

        if self.max_lines > 0:
            checks_performed += 1
            if line_count > self.max_lines:
                violations.append({
                    "type": "too_many_lines",
                    "message": f"Too many lines: {line_count} > {self.max_lines} max",
                    "severity": "low",
                })

        return {
            "passed": len(violations) == 0,
            "violations": violations,
            "checks_performed": checks_performed,
        }

    def _check_markdown(self, content: str) -> dict[str, Any]:
        """Check markdown formatting conventions."""
        violations = []

        # Check for code blocks
        if self.markdown_checks.get("require_code_blocks", False):
            if not re.search(r"```[\s\S]*?```", content):
                violations.append({
                    "type": "missing_code_block",
                    "message": "Output must contain code blocks",
                    "severity": "medium",
                })

        # Check heading levels
        max_level = self.markdown_checks.get("max_heading_level", 0)
        if max_level > 0:
            # Find all headings and their levels
            headings = re.findall(r"^(#{1,6})\s", content, re.MULTILINE)
            for h in headings:
                level = len(h)
                if level > max_level:
                    violations.append({
                        "type": "heading_too_deep",
                        "message": f"Heading level {level} exceeds max level {max_level}",
                        "severity": "low",
                    })
                    break  # Only report once

        # Check for lists
        if self.markdown_checks.get("require_lists", False):
            has_bullet = re.search(r"^\s*[-*+]\s", content, re.MULTILINE)
            has_numbered = re.search(r"^\s*\d+\.\s", content, re.MULTILINE)
            if not has_bullet and not has_numbered:
                violations.append({
                    "type": "missing_list",
                    "message": "Output must contain bullet or numbered lists",
                    "severity": "medium",
                })

        # Check for bare URLs
        if self.markdown_checks.get("no_bare_urls", False):
            # Find URLs not in markdown link format
            # This is a simplified check - URLs that are not preceded by ]( or wrapped in <>
            bare_url_pattern = r"(?<!\]\()(?<![<])(https?://[^\s\)>\]]+)(?![>\]])"
            bare_urls = re.findall(bare_url_pattern, content)
            if bare_urls:
                violations.append({
                    "type": "bare_url",
                    "message": (
                        f"Found {len(bare_urls)} bare URL(s) - "
                        "should use markdown link format"
                    ),
                    "severity": "low",
                })

        # Check for consistent heading style (ATX vs Setext)
        if self.markdown_checks.get("consistent_headings", False):
            has_atx = bool(re.search(r"^#+\s", content, re.MULTILINE))
            has_setext = bool(re.search(r"^[=-]+\s*$", content, re.MULTILINE))
            if has_atx and has_setext:
                violations.append({
                    "type": "inconsistent_headings",
                    "message": "Mixed ATX (#) and Setext (===) heading styles",
                    "severity": "low",
                })

        return {"passed": len(violations) == 0, "violations": violations}

    def _check_naming_conventions(self, content: str) -> dict[str, Any]:
        """Check naming conventions in code."""
        violations = []

        # Define naming patterns
        patterns = {
            "snake_case": r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$",
            "camelCase": r"^[a-z][a-zA-Z0-9]*$",
            "PascalCase": r"^[A-Z][a-zA-Z0-9]*$",
            "UPPER_SNAKE_CASE": r"^[A-Z][A-Z0-9]*(_[A-Z0-9]+)*$",
        }

        # Check function naming
        if "functions" in self.naming_conventions:
            expected = self.naming_conventions["functions"]
            if expected in patterns:
                # Find function definitions (Python style)
                func_names = re.findall(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", content)
                # Also check JavaScript/TypeScript style
                func_names.extend(re.findall(r"function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", content))

                bad_names = [
                    name for name in func_names
                    if not re.match(patterns[expected], name) and not name.startswith("_")
                ]
                if bad_names:
                    violations.append({
                        "type": "function_naming",
                        "message": f"Functions should use {expected}: {bad_names[:5]}",
                        "severity": "low",
                    })

        # Check variable naming
        if "variables" in self.naming_conventions:
            expected = self.naming_conventions["variables"]
            if expected in patterns:
                # Find variable assignments (simplified)
                var_names = re.findall(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*=", content, re.MULTILINE)
                # Filter out likely constants (all caps) and private vars
                var_names = [
                    v for v in var_names
                    if not v.isupper() and not v.startswith("_")
                ]

                bad_names = [
                    name for name in var_names
                    if not re.match(patterns[expected], name)
                ]
                if bad_names:
                    violations.append({
                        "type": "variable_naming",
                        "message": f"Variables should use {expected}: {bad_names[:5]}",
                        "severity": "low",
                    })

        # Check class naming
        if "classes" in self.naming_conventions:
            expected = self.naming_conventions["classes"]
            if expected in patterns:
                # Find class definitions
                class_names = re.findall(r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)", content)

                bad_names = [
                    name for name in class_names
                    if not re.match(patterns[expected], name)
                ]
                if bad_names:
                    violations.append({
                        "type": "class_naming",
                        "message": f"Classes should use {expected}: {bad_names[:5]}",
                        "severity": "low",
                    })

        # Check constant naming
        if "constants" in self.naming_conventions:
            expected = self.naming_conventions["constants"]
            if expected in patterns:
                # Find likely constants (all caps assignments)
                const_names = re.findall(r"^([A-Z][A-Z0-9_]*)\s*=", content, re.MULTILINE)

                bad_names = [
                    name for name in const_names
                    if not re.match(patterns[expected], name)
                ]
                if bad_names:
                    violations.append({
                        "type": "constant_naming",
                        "message": f"Constants should use {expected}: {bad_names[:5]}",
                        "severity": "low",
                    })

        return {"passed": len(violations) == 0, "violations": violations}

    def _build_reasoning(
        self, violations: list[dict[str, Any]], checks_passed: list[str], total_checks: int
    ) -> str:
        """Build human-readable reasoning string."""
        if not violations:
            return f"All {total_checks} style checks passed: {', '.join(checks_passed)}"

        parts = [f"Style violations found ({len(violations)}/{total_checks} checks failed):"]
        for v in violations[:3]:  # Limit to 3 in reasoning
            parts.append(f"- {v['type']}: {v['message'][:60]}")

        return " ".join(parts)
