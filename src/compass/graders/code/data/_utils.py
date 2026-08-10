"""Shared utilities for Data Agent graders."""

from __future__ import annotations

import re
from typing import Any


def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison (remove comments, extra whitespace)."""
    # Remove SQL comments
    sql = re.sub(r"--.*$", "", sql, flags=re.MULTILINE)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    # Normalize whitespace
    sql = " ".join(sql.split())
    # Normalize case for keywords
    return sql.strip()


def extract_sql_from_text(text: str) -> str | None:
    """Extract SQL from text that may contain markdown code blocks."""
    # Try to find SQL in code blocks
    match = re.search(r"```(?:sql)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Try to detect raw SQL (starts with common keywords)
    sql_pattern = r"^\s*(SELECT|INSERT|UPDATE|DELETE|WITH|CREATE|ALTER)\b"
    if re.match(sql_pattern, text.strip(), re.IGNORECASE):
        return text.strip()

    return None


def compare_values(actual: Any, expected: Any, tolerance: float = 1e-6) -> bool:
    """Compare two values with type-aware logic."""
    if actual is None and expected is None:
        return True
    if actual is None or expected is None:
        return False

    # Numeric comparison with tolerance
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        if expected == 0:
            return abs(actual) < tolerance
        return abs(actual - expected) / max(abs(expected), tolerance) < tolerance

    # String comparison (case-insensitive option could be added)
    if isinstance(actual, str) and isinstance(expected, str):
        return actual.strip() == expected.strip()

    return actual == expected


def compare_rows(
    actual_rows: list[dict[str, Any]],
    expected_rows: list[dict[str, Any]],
    key_columns: list[str] | None = None,
    tolerance: float = 1e-6,
    ignore_order: bool = True,
    ignore_extra_columns: bool = True,
) -> tuple[bool, list[str]]:
    """Compare two result sets.

    Args:
        actual_rows: Actual query results.
        expected_rows: Expected query results.
        key_columns: Columns to use for matching rows (if ignore_order=True).
        tolerance: Numeric comparison tolerance.
        ignore_order: If True, row order doesn't matter.
        ignore_extra_columns: If True, extra columns in actual are ignored.

    Returns:
        Tuple of (match, list of differences).
    """
    differences = []

    if len(actual_rows) != len(expected_rows):
        differences.append(
            f"Row count mismatch: actual={len(actual_rows)}, expected={len(expected_rows)}"
        )
        # Continue to find more specific differences

    if not expected_rows:
        return len(actual_rows) == 0, differences

    # Get columns to compare
    expected_columns = set(expected_rows[0].keys())

    if ignore_order and key_columns:
        # Match rows by key columns
        def get_key(row: dict[str, Any]) -> tuple[Any, ...]:
            return tuple(row.get(k) for k in key_columns)

        actual_by_key = {get_key(r): r for r in actual_rows}
        expected_by_key = {get_key(r): r for r in expected_rows}

        missing_keys = set(expected_by_key.keys()) - set(actual_by_key.keys())
        extra_keys = set(actual_by_key.keys()) - set(expected_by_key.keys())

        if missing_keys:
            differences.append(f"Missing rows with keys: {list(missing_keys)[:5]}")
        if extra_keys:
            differences.append(f"Extra rows with keys: {list(extra_keys)[:5]}")

        # Compare matching rows
        for key in set(expected_by_key.keys()) & set(actual_by_key.keys()):
            actual_row = actual_by_key[key]
            expected_row = expected_by_key[key]

            for col in expected_columns:
                if not ignore_extra_columns or col in actual_row:
                    actual_val = actual_row.get(col)
                    expected_val = expected_row.get(col)
                    if not compare_values(actual_val, expected_val, tolerance):
                        differences.append(
                            f"Row {key}: column '{col}' mismatch: "
                            f"actual={actual_val}, expected={expected_val}"
                        )

    elif ignore_order:
        # Sort and compare (for simple cases without key columns)
        def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
            return tuple(sorted(str(v) for v in row.values()))

        sorted_actual = sorted(actual_rows, key=sort_key)
        sorted_expected = sorted(expected_rows, key=sort_key)

        # strict=False: a row-count mismatch is already recorded as a difference
        # above, and comparison deliberately continues to surface more of them.
        for i, (actual_row, expected_row) in enumerate(
            zip(sorted_actual, sorted_expected, strict=False)
        ):
            for col in expected_columns:
                if not ignore_extra_columns or col in actual_row:
                    actual_val = actual_row.get(col)
                    expected_val = expected_row.get(col)
                    if not compare_values(actual_val, expected_val, tolerance):
                        differences.append(
                            f"Row {i}: column '{col}' mismatch: "
                            f"actual={actual_val}, expected={expected_val}"
                        )
    else:
        # Compare in order
        for i, (actual_row, expected_row) in enumerate(
            zip(actual_rows, expected_rows, strict=False)
        ):
            for col in expected_columns:
                if not ignore_extra_columns or col in actual_row:
                    actual_val = actual_row.get(col)
                    expected_val = expected_row.get(col)
                    if not compare_values(actual_val, expected_val, tolerance):
                        differences.append(
                            f"Row {i}: column '{col}' mismatch: "
                            f"actual={actual_val}, expected={expected_val}"
                        )

    return len(differences) == 0, differences
