"""Structured output validation graders.

Provides graders for validating structured outputs like JSON Schema,
SQL syntax, and general format validation (JSON/YAML/XML/TOML).
"""

from __future__ import annotations

import json
import re
from typing import Any

from compass.graders.base import CodeGrader, GradeContext, GradeResult, GraderScope, GraderType
from compass.graders.registry import register_grader


def _extract_content(context: GradeContext, source: str) -> str | dict[str, Any] | None:
    """Extract content from context based on source specification.

    Args:
        context: Grade context.
        source: Source specification:
            - "output_data" or "output": from outcome.output_data
            - "text" or "text_artifact": from TextArtifact.content
            - "code" or "code_artifact": from CodeArtifact files
            - "metadata.<key>": from outcome.metadata
            - "output_data.<key>": from specific field in output_data

    Returns:
        Extracted content as string or dict.
    """
    if not context.outcome:
        return None

    if source in ("output_data", "output"):
        return context.outcome.output_data

    if source in ("text", "text_artifact"):
        artifact = context.text_artifact
        if artifact and hasattr(artifact, "content"):
            return artifact.content
        return None

    if source in ("code", "code_artifact"):
        artifact = context.code_artifact
        if artifact and hasattr(artifact, "files"):
            # Return concatenated file contents
            contents = []
            for f in artifact.files:
                if hasattr(f, "content"):
                    contents.append(f.content)
            return "\n".join(contents)
        return None

    if source.startswith("metadata."):
        key = source[9:]
        return context.outcome.metadata.get(key)

    if source.startswith("output_data."):
        key = source[12:]
        data = context.outcome.output_data
        if isinstance(data, dict):
            return data.get(key)

    return None


# =============================================================================
# JSON Schema Grader
# =============================================================================


@register_grader("json_schema")
class JsonSchemaGrader(CodeGrader):
    """Validates JSON output against a JSON Schema with partial compliance scoring.

    Config:
        schema: dict | str — JSON Schema definition (dict) or file path (str).
        source: str — Where to get JSON from (default "output_data").
            Options: "output_data", "text_artifact", "metadata.<key>", etc.
        strict: bool — If True, disallow additional properties (default False).
        extract_json: bool — If True, extract JSON from text (default False).

        # Scoring mode (NEW)
        scoring_mode: str — How to calculate score (default "strict"):
            - "strict": score=0 on any validation error (original behavior)
            - "partial": score based on percentage of valid fields
            - "weighted": use field_weights for custom scoring

        # Partial scoring config (NEW)
        field_weights: dict — Custom weights for specific fields.
            Example: {"name": 2.0, "email": 1.5}
        required_weight: float — Weight multiplier for required fields (default 2.0).
        count_nested: bool — Include nested fields in scoring (default True).
        pass_threshold: float — Minimum score to pass (default 1.0 for strict, 0.7 for partial).

    Example YAML (strict mode - original behavior):
        graders:
          - name: json_schema
            config:
              schema:
                type: object
                properties:
                  name: { type: string }
                  age: { type: integer, minimum: 0 }
                required: [name, age]
              strict: true

    Example YAML (partial scoring mode):
        graders:
          - name: json_schema
            config:
              scoring_mode: partial
              pass_threshold: 0.8
              schema:
                type: object
                properties:
                  name: { type: string }
                  email: { type: string, format: email }
                  age: { type: integer }
                  address: { type: object }
                required: [name, email]
              field_weights:
                name: 2.0      # name is more important
                email: 1.5     # email is important
              required_weight: 2.0  # required fields count double
    """

    grader_type = GraderType.CODE
    grader_scope = GraderScope.OUTCOME

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.schema = self.config.get("schema", {})
        self.source = self.config.get("source", "output_data")
        self.strict = self.config.get("strict", False)
        self.extract_json = self.config.get("extract_json", False)

        # Scoring configuration
        self.scoring_mode = self.config.get("scoring_mode", "strict")
        self.field_weights = self.config.get("field_weights", {})
        self.required_weight = self.config.get("required_weight", 2.0)
        self.count_nested = self.config.get("count_nested", True)

        # Pass threshold depends on scoring mode
        default_threshold = 1.0 if self.scoring_mode == "strict" else 0.7
        self.pass_threshold = self.config.get("pass_threshold", default_threshold)

        # Load schema from file if string path provided
        if isinstance(self.schema, str):
            import os
            if os.path.exists(self.schema):
                with open(self.schema, encoding="utf-8") as f:
                    self.schema = json.load(f)
            else:
                # Treat as inline JSON string
                self.schema = json.loads(self.schema)

        # Apply strict mode
        if self.strict and isinstance(self.schema, dict):
            self._apply_strict_mode(self.schema)

    def _apply_strict_mode(self, schema: dict[str, Any]) -> None:
        """Recursively set additionalProperties: false for objects."""
        if schema.get("type") == "object":
            schema.setdefault("additionalProperties", False)
            props = schema.get("properties", {})
            for prop_schema in props.values():
                if isinstance(prop_schema, dict):
                    self._apply_strict_mode(prop_schema)

        # Handle nested schemas in arrays
        if schema.get("type") == "array":
            items = schema.get("items")
            if isinstance(items, dict):
                self._apply_strict_mode(items)

    def _extract_json_from_text(self, text: str) -> str:
        """Extract JSON from text that may contain other content."""
        # Try to find JSON in code blocks
        code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if code_block_match:
            return code_block_match.group(1).strip()

        # Try to find JSON object or array
        json_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
        if json_match:
            return json_match.group(1)

        return text

    def _get_schema_fields(
        self, schema: dict[str, Any], prefix: str = "", required_fields: set[Any] | None = None
    ) -> list[dict[str, Any]]:
        """Extract all fields from schema with their metadata.

        Returns list of field info dicts with keys:
            - path: field path (e.g., "user.name")
            - required: whether field is required
            - type: expected type
            - weight: field weight for scoring
        """
        if required_fields is None:
            required_fields = set(schema.get("required", []))

        fields = []
        properties = schema.get("properties", {})

        for name, prop_schema in properties.items():
            field_path = f"{prefix}{name}" if prefix else name
            is_required = name in required_fields

            # Calculate weight
            base_weight = self.field_weights.get(field_path, 1.0)
            if is_required:
                base_weight *= self.required_weight

            field_info = {
                "path": field_path,
                "required": is_required,
                "type": prop_schema.get("type", "any"),
                "weight": base_weight,
            }
            fields.append(field_info)

            # Recursively get nested fields
            if self.count_nested and prop_schema.get("type") == "object":
                nested_required = set(prop_schema.get("required", []))
                nested_fields = self._get_schema_fields(
                    prop_schema, prefix=f"{field_path}.", required_fields=nested_required
                )
                fields.extend(nested_fields)

        return fields

    def _categorize_errors(self, errors: list[Any], data: Any) -> dict[str, Any]:
        """Categorize validation errors by type and field.

        Returns:
            {
                "by_field": {"field_path": [errors]},
                "by_type": {"missing": [...], "type_error": [...],
                            "constraint": [...], "other": [...]},
                "failed_fields": set of field paths that failed,
            }
        """
        by_field: dict[str, list[Any]] = {}
        by_type: dict[str, list[Any]] = {
            "missing": [],      # Required field missing
            "type_error": [],   # Wrong type
            "constraint": [],   # Constraint violation (min, max, pattern, etc.)
            "format": [],       # Format error (email, uri, etc.)
            "additional": [],   # Additional properties not allowed
            "other": [],        # Other errors
        }
        failed_fields: set[str] = set()

        for error in errors:
            path = "/".join(str(p) for p in error.absolute_path) if error.absolute_path else ""
            error_info = {
                "path": path,
                "message": error.message,
                "validator": error.validator,
                "schema_path": list(error.schema_path),
            }

            # Categorize by field
            if path:
                by_field.setdefault(path, []).append(error_info)
                failed_fields.add(path)

            # Categorize by error type
            validator = error.validator
            if validator == "required":
                by_type["missing"].append(error_info)
                # Extract missing field names from message
                if "is a required property" in error.message:
                    missing_field = error.message.split("'")[1]
                    parent_path = path + "/" if path else ""
                    failed_fields.add(f"{parent_path}{missing_field}")
            elif validator == "type":
                by_type["type_error"].append(error_info)
            elif validator in ("minimum", "maximum", "minLength", "maxLength", "pattern", "enum"):
                by_type["constraint"].append(error_info)
            elif validator == "format":
                by_type["format"].append(error_info)
            elif validator == "additionalProperties":
                by_type["additional"].append(error_info)
            else:
                by_type["other"].append(error_info)

        return {
            "by_field": by_field,
            "by_type": by_type,
            "failed_fields": failed_fields,
        }

    def _calculate_partial_score(
        self, schema_fields: list[dict[str, Any]], failed_fields: set[str], data: Any
    ) -> tuple[float, dict[str, Any]]:
        """Calculate partial compliance score.

        Returns:
            (score, field_results)
        """
        field_results = []
        total_weight = 0.0
        earned_weight = 0.0

        for field in schema_fields:
            path = field["path"]
            weight = field["weight"]
            total_weight += weight

            # Check if field passed
            passed = path not in failed_fields

            # Also check if required field exists in data
            if field["required"] and passed:
                # Verify the field actually exists
                value = self._get_nested_value(data, path)
                if value is None and path not in str(data):
                    passed = False

            if passed:
                earned_weight += weight

            field_results.append({
                "path": path,
                "required": field["required"],
                "type": field["type"],
                "weight": weight,
                "passed": passed,
            })

        score = earned_weight / total_weight if total_weight > 0 else 1.0
        return score, {
            "fields": field_results,
            "total_weight": total_weight,
            "earned_weight": earned_weight,
        }

    def _get_nested_value(self, data: Any, path: str) -> Any:
        """Get nested value from data using dot-separated path."""
        if not isinstance(data, dict):
            return None

        parts = path.split(".")
        current = data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    async def grade(self, context: GradeContext) -> GradeResult:
        """Validate JSON against schema with partial compliance scoring."""
        try:
            from jsonschema import Draft7Validator
        except ImportError:
            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                score=0.0,
                passed=False,
                error="jsonschema library not installed. Run: pip install jsonschema",
            )

        # Extract content
        content = _extract_content(context, self.source)
        if content is None:
            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                score=0.0,
                passed=False,
                error=f"No content found at source: {self.source}",
            )

        # Parse JSON
        try:
            if isinstance(content, str):
                if self.extract_json:
                    content = self._extract_json_from_text(content)
                data = json.loads(content)
            elif isinstance(content, (dict, list)):
                data = content
            else:
                return GradeResult(
                    name=self.name,
                    grader_type=self.grader_type,
                    grader_scope=self.grader_scope,
                    score=0.0,
                    passed=False,
                    error=f"Unexpected content type: {type(content).__name__}",
                )
        except json.JSONDecodeError as e:
            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                score=0.0,
                passed=False,
                details={"parse_error": str(e), "content_preview": str(content)[:200]},
                error=f"Invalid JSON: {e}",
            )

        # Validate against schema
        validator = Draft7Validator(self.schema)
        errors = list(validator.iter_errors(data))

        # No errors - perfect score
        if not errors:
            schema_fields = (
                self._get_schema_fields(self.schema)
                if isinstance(self.schema, dict) else []
            )
            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                score=1.0,
                passed=True,
                details={
                    "scoring_mode": self.scoring_mode,
                    "total_fields": len(schema_fields),
                    "valid_fields": len(schema_fields),
                    "validated_keys": list(data.keys()) if isinstance(data, dict) else len(data),
                    "field_compliance": {
                        "total": len(schema_fields),
                        "passed": len(schema_fields),
                        "failed": 0,
                    },
                },
                reasoning="All schema validations passed",
            )

        # Categorize errors for detailed diagnostics
        error_analysis = self._categorize_errors(errors, data)

        # Calculate score based on scoring mode
        if self.scoring_mode == "strict":
            # Original behavior: any error = score 0
            score = 0.0
            field_details = None
        else:
            # Partial or weighted scoring
            schema_fields = (
                self._get_schema_fields(self.schema)
                if isinstance(self.schema, dict) else []
            )
            score, field_details = self._calculate_partial_score(
                schema_fields, error_analysis["failed_fields"], data
            )

        passed = score >= self.pass_threshold

        # Build error summary
        error_summary = []
        for err_type, err_list in error_analysis["by_type"].items():
            if err_list:
                error_summary.append(f"{err_type}: {len(err_list)}")

        # Build detailed error list (limit to 15)
        error_details = [
            {
                "path": "/".join(str(p) for p in e.absolute_path) if e.absolute_path else "(root)",
                "message": e.message,
                "validator": e.validator,
            }
            for e in errors[:15]
        ]

        # Build field compliance summary
        if field_details:
            passed_count = sum(1 for f in field_details["fields"] if f["passed"])
            failed_count = len(field_details["fields"]) - passed_count
            field_compliance = {
                "total": len(field_details["fields"]),
                "passed": passed_count,
                "failed": failed_count,
                "score_breakdown": {
                    "total_weight": round(field_details["total_weight"], 2),
                    "earned_weight": round(field_details["earned_weight"], 2),
                },
            }
            # Include per-field results
            field_results = [
                {
                    "path": f["path"],
                    "required": f["required"],
                    "passed": f["passed"],
                    "weight": f["weight"],
                }
                for f in field_details["fields"]
            ]
        else:
            field_compliance = {
                "total": len(error_analysis["by_field"]) + 1,  # Approximate
                "passed": 0,
                "failed": len(errors),
            }
            field_results = None

        details = {
            "scoring_mode": self.scoring_mode,
            "pass_threshold": self.pass_threshold,
            "error_count": len(errors),
            "error_summary": dict(error_analysis["by_type"]),
            "errors": error_details,
            "field_compliance": field_compliance,
        }
        if field_results:
            details["field_results"] = field_results

        # Build reasoning
        if self.scoring_mode == "strict":
            reasoning = f"Schema validation failed with {len(errors)} error(s)"
        else:
            reasoning = (
                f"Partial compliance: {field_compliance['passed']}"
                f"/{field_compliance['total']} fields valid "
                f"(score: {score:.2f}, threshold: {self.pass_threshold})"
            )

        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            score=round(score, 4),
            passed=passed,
            details=details,
            reasoning=reasoning,
            error=f"Schema validation: {', '.join(error_summary)}" if not passed else None,
        )


# =============================================================================
# SQL Syntax Grader
# =============================================================================


@register_grader("sql_syntax")
class SqlSyntaxGrader(CodeGrader):
    """Validates SQL syntax correctness.

    Config:
        source: str — Where to get SQL from (default "text_artifact").
        dialect: str — SQL dialect for validation (default "generic").
            Options: "generic", "mysql", "postgresql", "sqlite", "tsql".
        allowed_statements: list[str] — Allowed statement types (default all).
            Options: "SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", etc.
        forbidden_keywords: list[str] — Keywords that should not appear.
        require_semicolon: bool — Require statements to end with semicolon.
        max_statements: int — Maximum number of statements allowed (0 = unlimited).

    Example YAML:
        graders:
          - name: sql_syntax
            config:
              source: text_artifact
              dialect: postgresql
              allowed_statements: [SELECT]
              forbidden_keywords: [DROP, TRUNCATE, DELETE]
    """

    grader_type = GraderType.CODE
    grader_scope = GraderScope.OUTCOME

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.source = self.config.get("source", "text_artifact")
        self.dialect = self.config.get("dialect", "generic")
        self.allowed_statements = self.config.get("allowed_statements", [])
        self.forbidden_keywords = [k.upper() for k in self.config.get("forbidden_keywords", [])]
        self.require_semicolon = self.config.get("require_semicolon", False)
        self.max_statements = self.config.get("max_statements", 0)

    async def grade(self, context: GradeContext) -> GradeResult:
        """Validate SQL syntax."""
        try:
            import sqlparse
        except ImportError:
            return GradeResult(
                score=0.0,
                passed=False,
                error="sqlparse library not installed. Run: pip install sqlparse",
            )

        # Extract content
        content = _extract_content(context, self.source)
        if content is None:
            return GradeResult(
                score=0.0,
                passed=False,
                error=f"No content found at source: {self.source}",
            )

        if not isinstance(content, str):
            content = str(content)

        # Extract SQL from code blocks if present
        code_block_match = re.search(r"```(?:sql)?\s*([\s\S]*?)```", content)
        if code_block_match:
            content = code_block_match.group(1).strip()

        if not content.strip():
            return GradeResult(
                score=0.0,
                passed=False,
                error="Empty SQL content",
            )

        # Parse SQL
        try:
            statements = sqlparse.parse(content)
        except Exception as e:
            return GradeResult(
                score=0.0,
                passed=False,
                error=f"SQL parse error: {e}",
            )

        if not statements:
            return GradeResult(
                score=0.0,
                passed=False,
                error="No valid SQL statements found",
            )

        # Filter out empty statements
        statements = [s for s in statements if s.get_type() != "UNKNOWN" or str(s).strip()]

        errors = []
        warnings = []
        statement_types = []

        # Check max statements
        if self.max_statements > 0 and len(statements) > self.max_statements:
            errors.append(f"Too many statements: {len(statements)} > {self.max_statements}")

        for i, stmt in enumerate(statements):
            stmt_str = str(stmt).strip()
            if not stmt_str:
                continue

            stmt_type = stmt.get_type()
            statement_types.append(stmt_type)

            # Check allowed statements
            if self.allowed_statements:
                allowed_upper = [s.upper() for s in self.allowed_statements]
                if stmt_type.upper() not in allowed_upper and stmt_type != "UNKNOWN":
                    errors.append(f"Statement {i+1}: {stmt_type} not in allowed types")

            # Check forbidden keywords
            stmt_upper = stmt_str.upper()
            for keyword in self.forbidden_keywords:
                if re.search(rf"\b{keyword}\b", stmt_upper):
                    errors.append(f"Statement {i+1}: contains forbidden keyword '{keyword}'")

            # Check semicolon
            if self.require_semicolon and not stmt_str.rstrip().endswith(";"):
                warnings.append(f"Statement {i+1}: missing semicolon")

            # Basic syntax validation
            if stmt_type == "UNKNOWN" and not self._is_valid_statement(stmt_str):
                errors.append(f"Statement {i+1}: unrecognized or invalid syntax")

        # Calculate score
        if errors:
            score = 0.0
            passed = False
        elif warnings:
            score = 0.8
            passed = True
        else:
            score = 1.0
            passed = True

        return GradeResult(
            score=score,
            passed=passed,
            details={
                "statement_count": len(statements),
                "statement_types": statement_types,
                "errors": errors,
                "warnings": warnings,
            },
            error="; ".join(errors) if errors else None,
        )

    def _is_valid_statement(self, sql: str) -> bool:
        """Basic check if SQL looks valid."""
        sql = sql.strip().upper()
        # Check if it starts with a known keyword
        known_starts = [
            "SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP",
            "TRUNCATE", "GRANT", "REVOKE", "BEGIN", "COMMIT", "ROLLBACK",
            "WITH", "EXPLAIN", "SHOW", "DESCRIBE", "USE", "SET",
        ]
        return any(sql.startswith(k) for k in known_starts)


# =============================================================================
# Structure Check Grader (Generic Format Validation)
# =============================================================================


@register_grader("structure_check")
class StructureCheckGrader(CodeGrader):
    """Validates structured format (JSON/YAML/XML/TOML).

    Config:
        format: str — Format to validate: "json", "yaml", "xml", "toml".
        source: str — Where to get content from (default "text_artifact").
        schema: dict — Optional JSON Schema for JSON/YAML validation.
        required_keys: list[str] — Keys that must be present (JSON/YAML/TOML).
        required_elements: list[str] — Elements that must be present (XML).
        extract_from_text: bool — Extract format from surrounding text.

    Example YAML:
        graders:
          - name: structure_check
            config:
              format: yaml
              source: text_artifact
              required_keys: [name, version, dependencies]
    """

    grader_type = GraderType.CODE
    grader_scope = GraderScope.OUTCOME

    SUPPORTED_FORMATS = ["json", "yaml", "xml", "toml"]

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.format = self.config.get("format", "json").lower()
        self.source = self.config.get("source", "text_artifact")
        self.schema = self.config.get("schema")
        self.required_keys = self.config.get("required_keys", [])
        self.required_elements = self.config.get("required_elements", [])
        self.extract_from_text = self.config.get("extract_from_text", True)

        if self.format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {self.format}. Use: {self.SUPPORTED_FORMATS}")

    def _extract_format_content(self, content: str) -> str:
        """Extract formatted content from text."""
        if not self.extract_from_text:
            return content

        # Try code blocks first
        pattern = rf"```(?:{self.format})?\s*([\s\S]*?)```"
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        return content

    async def grade(self, context: GradeContext) -> GradeResult:
        """Validate structured format."""
        # Extract content
        content = _extract_content(context, self.source)
        if content is None:
            return GradeResult(
                score=0.0,
                passed=False,
                error=f"No content found at source: {self.source}",
            )

        if isinstance(content, dict):
            # Already parsed
            data = content
        elif isinstance(content, str):
            content = self._extract_format_content(content)
            if not content.strip():
                return GradeResult(
                    score=0.0,
                    passed=False,
                    error="Empty content",
                )

            # Parse based on format
            try:
                data = await self._parse_content(content)
            except Exception as e:
                return GradeResult(
                    score=0.0,
                    passed=False,
                    details={"content_preview": content[:200]},
                    error=f"Parse error ({self.format}): {e}",
                )
        else:
            return GradeResult(
                score=0.0,
                passed=False,
                error=f"Unexpected content type: {type(content).__name__}",
            )

        # Validate structure
        errors = []

        # Check required keys (for dict-like formats)
        if self.required_keys and isinstance(data, dict):
            missing_keys = [k for k in self.required_keys if k not in data]
            if missing_keys:
                errors.append(f"Missing required keys: {missing_keys}")

        # Check required elements (for XML)
        if self.required_elements and self.format == "xml":
            missing = self._check_xml_elements(data, self.required_elements)
            if missing:
                errors.append(f"Missing required elements: {missing}")

        # Validate against schema if provided (JSON/YAML)
        if self.schema and self.format in ("json", "yaml"):
            schema_errors = await self._validate_schema(data)
            errors.extend(schema_errors)

        if errors:
            return GradeResult(
                score=0.0,
                passed=False,
                details={"errors": errors},
                error="; ".join(errors),
            )

        return GradeResult(
            score=1.0,
            passed=True,
            details={
                "format": self.format,
                "keys": list(data.keys()) if isinstance(data, dict) else None,
            },
        )

    async def _parse_content(self, content: str) -> Any:
        """Parse content based on format."""
        if self.format == "json":
            return json.loads(content)

        elif self.format == "yaml":
            try:
                import yaml
            except ImportError as exc:
                raise ImportError("pyyaml not installed. Run: pip install pyyaml") from exc
            return yaml.safe_load(content)

        elif self.format == "toml":
            try:
                import tomllib  # Python 3.11+
            except ImportError:
                try:
                    import tomli as tomllib  # Fallback
                except ImportError as exc:
                    raise ImportError("tomli not installed. Run: pip install tomli") from exc
            return tomllib.loads(content)

        elif self.format == "xml":
            try:
                import xml.etree.ElementTree as ET
            except ImportError as exc:
                raise ImportError("xml.etree not available") from exc
            return ET.fromstring(content)

        raise ValueError(f"Unknown format: {self.format}")

    def _check_xml_elements(self, root: Any, required: list[str]) -> list[str]:
        """Check for required XML elements."""
        missing = []
        for elem in required:
            if root.find(f".//{elem}") is None:
                missing.append(elem)
        return missing

    async def _validate_schema(self, data: Any) -> list[str]:
        """Validate data against JSON Schema."""
        if not self.schema:
            return []

        try:
            from jsonschema import Draft7Validator
        except ImportError:
            return ["jsonschema not installed for schema validation"]

        validator = Draft7Validator(self.schema)
        errors = list(validator.iter_errors(data))

        return [
            f"Schema error at {'/'.join(str(p) for p in e.absolute_path)}: {e.message}"
            for e in errors[:5]
        ]
