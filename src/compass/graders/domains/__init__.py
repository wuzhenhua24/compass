"""Domain correctness graders — the part of Compass that is not the framework.

Compass's spine is domain-agnostic: the Transcript/Outcome model, the ToolCall
protocol, and the *process* graders that read them (cost, latency, loops, tool
usage, dangerous-op execution). Whether an answer, an image or a patch is
*correct* is a question about a domain, and the answer belongs to whoever owns
that domain — the way ``tests/`` belongs to a project, not to pytest.

Three domains ship anyway, because they are what Compass was built against and
because a framework with no worked examples teaches nothing: ``coding``,
``data`` and ``image``. They live here rather than under ``graders/code`` and
``graders/model`` so the boundary is visible in the tree, and they are imported
lazily so that ``import compass`` does not drag in ``sqlparse``, ``duckdb`` or
a CLIP model for a user who is evaluating none of that.

Loading is automatic: name ``sql_equivalence`` in a scenario and the registry
loads ``data`` before it gives up. Nothing to declare, nothing to configure.
Their optional dependencies are the extras ``compass[data]`` and
``compass[image]``; a grader whose dependency is missing says so in its result
rather than raising at import.
"""

from __future__ import annotations

import importlib

#: The domains that ship with Compass, in load order.
DOMAINS = ("coding", "data", "image")

_loaded: set[str] = set()


def load(domain: str) -> None:
    """Import one domain package, registering its graders.

    Idempotent, and a no-op for a domain already loaded.

    Raises:
        ValueError: If *domain* is not one Compass ships.
    """
    if domain in _loaded:
        return
    if domain not in DOMAINS:
        raise ValueError(f"Unknown grader domain {domain!r}. Available: {list(DOMAINS)}")
    importlib.import_module(f"compass.graders.domains.{domain}")
    _loaded.add(domain)


def load_all() -> None:
    """Import every shipped domain. What the registry falls back to on a miss."""
    for domain in DOMAINS:
        load(domain)


def domain_of(name: str) -> str | None:
    """Which domain registered the grader *name*, or ``None`` if the framework did.

    The answer is read off the grader's own module rather than kept in a second
    list, so a grader moved between packages cannot end up filed under the one
    it left.
    """
    from compass.graders.registry import get_grader

    module = get_grader(name).__module__
    prefix = f"{__name__}."
    if not module.startswith(prefix):
        return None
    return module[len(prefix):].split(".", 1)[0]
