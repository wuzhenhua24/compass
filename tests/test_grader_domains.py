"""The framework/domain split, and the laziness that makes it worth having.

The split is only real if importing Compass does not import the domains. That
is one eager import away from silently reverting, and nothing about the tree
would look wrong afterwards — so it is asserted here rather than trusted.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from compass.graders import list_graders
from compass.graders.domains import DOMAINS, domain_of, load, load_all


def _in_a_fresh_interpreter(code: str) -> str:
    """Run *code* in a new process — module state does not survive it."""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


class TestLaziness:
    def test_importing_compass_does_not_import_any_domain(self):
        loaded = _in_a_fresh_interpreter(
            "import sys, compass.graders;"
            "print([m for m in sys.modules if m.startswith('compass.graders.domains.')])"
        )
        assert loaded == "[]", f"a domain is being imported eagerly: {loaded}"

    def test_importing_compass_does_not_pull_in_a_domain_backend(self):
        """The extras exist so a user evaluating none of this pays for none of it."""
        loaded = _in_a_fresh_interpreter(
            "import sys, compass.graders;"
            "print(sorted({'sqlparse', 'duckdb', 'torch', 'open_clip'} & set(sys.modules)))"
        )
        assert loaded == "[]", f"an optional backend is imported eagerly: {loaded}"

    def test_naming_a_domain_grader_loads_its_domain(self):
        """No declaration, no config — the registry falls back on a miss."""
        loaded = _in_a_fresh_interpreter(
            "import sys;"
            "from compass.graders import get_grader;"
            "g = get_grader('sql_equivalence');"
            "print(g.__module__)"
        )
        assert loaded == "compass.graders.domains.data.sql_equivalence"


class TestTheSplit:
    def test_every_grader_is_on_exactly_one_side(self):
        assert {domain_of(n) for n in list_graders()} <= {None, *DOMAINS}

    def test_the_framework_half_owns_no_domain_package(self):
        framework = [n for n in list_graders() if domain_of(n) is None]
        for name in framework:
            from compass.graders.registry import get_grader

            module = get_grader(name).__module__
            assert not module.startswith("compass.graders.domains."), name

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_each_shipped_domain_registers_something(self, domain):
        load(domain)
        assert [n for n in list_graders() if domain_of(n) == domain]

    def test_process_graders_stayed_in_the_framework(self):
        """The reusable spine — these must never end up behind an extra."""
        for name in ("cost_budget", "latency_budget", "loop_detection", "tool_usage",
                     "dangerous_operations", "state_delta", "turn_count", "leak_detection"):
            assert domain_of(name) is None, f"{name} moved into a domain"

    def test_correctness_graders_are_in_a_domain(self):
        for name, expected in (("sql_equivalence", "data"), ("sql_syntax", "data"),
                               ("lint", "coding"), ("integration_test", "coding"),
                               ("semantic_match", "image"), ("safety_check", "image")):
            assert domain_of(name) == expected


class TestLoad:
    def test_load_is_idempotent(self):
        load("data")
        load("data")

    def test_load_all_loads_all(self):
        load_all()
        assert {domain_of(n) for n in list_graders()} >= set(DOMAINS)

    def test_an_unknown_domain_says_so(self):
        with pytest.raises(ValueError, match="Unknown grader domain"):
            load("javascript")
