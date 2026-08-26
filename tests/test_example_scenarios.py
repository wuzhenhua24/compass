"""Every scenario YAML under examples/ must actually be runnable.

These files are the first thing someone copies, so a name that no longer
resolves or a config key nothing reads is worse here than in the docs — it
looks like it works. All of this is checked against the code rather than
maintained by hand.
"""

from __future__ import annotations

import inspect
import pathlib

import pytest
import yaml

from compass.adapters import list_adapters
from compass.adapters.registry import get_adapter
from compass.core.scenario import Scenario
from compass.graders.registry import get_grader

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"


def _scenario_files() -> list[pathlib.Path]:
    """Example YAMLs that are Scenarios, not datasets a harness parses itself."""
    found = []
    for path in sorted(EXAMPLES.rglob("*.y*ml")):
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            found.append(path)  # let the load test report it
            continue
        if isinstance(raw, dict) and "agent" in raw:
            found.append(path)
    return found


SCENARIOS = _scenario_files()


def _config_keys_read_by(cls) -> str:
    """Source of the class and its Compass bases, for a substring check."""
    parts = []
    for base in cls.__mro__:
        if base.__module__.startswith("compass"):
            try:
                parts.append(inspect.getsource(base))
            except OSError:
                pass
    return "\n".join(parts)


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: p.name)
class TestExampleScenarios:
    def test_it_loads(self, path: pathlib.Path):
        Scenario.from_yaml(str(path))

    def test_its_adapter_is_registered(self, path: pathlib.Path):
        adapter = Scenario.from_yaml(str(path)).agent.adapter
        assert adapter in set(list_adapters()), f"unknown adapter {adapter!r}"

    def test_every_grader_it_names_resolves(self, path: pathlib.Path):
        scenario = Scenario.from_yaml(str(path))
        names = {g.name for g in scenario.default_graders}
        for case in scenario.cases:
            names |= {g.name for g in case.graders}
        assert names, f"{path.name} names no graders"
        for name in sorted(names):
            get_grader(name)  # raises KeyError if the example invented one

    def test_its_adapter_config_keys_are_read(self, path: pathlib.Path):
        """A key the adapter never reads is a setting that silently does nothing."""
        scenario = Scenario.from_yaml(str(path))
        source = _config_keys_read_by(get_adapter(scenario.agent.adapter))
        unread = [
            key
            for key in (scenario.agent.config or {})
            if f'"{key}"' not in source and f"'{key}'" not in source
        ]
        assert unread == [], (
            f"{scenario.agent.adapter} never reads {unread} — "
            f"either wire them up or drop them from the example"
        )


def test_there_are_example_scenarios_to_check():
    """A silent zero here would make every parametrized test vacuous."""
    assert len(SCENARIOS) >= 5
