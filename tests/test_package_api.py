"""What ``import compass`` costs, and what it promises.

Only one name in the root package drives an agent, and only that one needs the
adapter registry behind it. Everything else — grading a recorded run, importing
a trajectory, publishing a site — is control plane and should not pay for the
drivers. That is one eager import away from silently reverting, and nothing
about the tree would look wrong afterwards, so it is asserted here rather than
trusted.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

import compass


def _in_a_fresh_interpreter(code: str) -> str:
    """Run *code* in a new process — module state does not survive it."""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _adapters_loaded_after(statement: str) -> str:
    return _in_a_fresh_interpreter(
        f"import sys; {statement}; print('compass.adapters' in sys.modules)"
    )


class TestLaziness:
    def test_importing_compass_does_not_import_the_adapter_registry(self):
        assert _adapters_loaded_after("import compass") == "False"

    @pytest.mark.parametrize(
        "statement",
        [
            "import compass.llm",
            "import compass.integrations",
            "import compass.graders",
            "from compass import Scenario, EvalResult",
        ],
    )
    def test_the_scoring_path_does_not_import_the_adapter_registry(self, statement):
        """Pricing a trajectory, scoring one, or reading the result model are all
        things you can do without a driver — and without importing one."""
        assert _adapters_loaded_after(statement) == "False", statement

    def test_asking_for_Compass_gets_the_real_class(self):
        """Deferred, not removed: the fallback loads the runner on first touch."""
        assert (
            _in_a_fresh_interpreter(
                "from compass import Compass; print(Compass.__module__)"
            )
            == "compass.core.runner"
        )

    def test_asking_for_Compass_is_what_pulls_the_registry_in(self):
        """The other half of the claim — the cost was deferred to the one export
        that genuinely needs it, not made to vanish."""
        assert _adapters_loaded_after("from compass import Compass") == "True"


class TestTheSurface:
    def test_every_name_in_all_resolves(self):
        for name in compass.__all__:
            assert getattr(compass, name) is not None, name

    def test_dir_lists_the_lazy_export(self):
        """Otherwise it is invisible to tab-completion and to `help(compass)`."""
        assert "Compass" in dir(compass)

    def test_an_unknown_attribute_still_raises(self):
        """A module __getattr__ that answers everything turns typos into imports
        that fail somewhere else."""
        with pytest.raises(AttributeError, match="no attribute 'NotAThing'"):
            _ = compass.NotAThing
