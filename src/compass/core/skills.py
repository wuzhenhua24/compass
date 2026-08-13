"""Reading a skill directory's own identity.

A *skill* here is the convention Claude Code and Codex share: a directory with
a ``SKILL.md`` whose YAML frontmatter carries ``name`` and ``description``, plus
optional ``scripts/`` , ``references/`` and ``assets/`` beside it.

Two distant parts of Compass need the ``name`` out of that frontmatter, and they
have to agree:

- :class:`~compass.adapters.cli_agent.CliAgentAdapter` installs the directory
  under that name, so ``report-writer-v1`` and ``report-writer-v2`` both reach
  the agent as ``report-writer`` — otherwise a v1/v2 comparison silently
  renames the thing it is measuring.
- ``skill_trigger`` has to recognise the same name in a trace to answer "did
  the agent actually load it".

If those two ever disagreed, the grader would report an untriggered skill for
every run of a skill that worked fine. Hence one rule, in one place.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def read_skill_name(skill_dir: Path) -> str:
    """The ``name:`` from ``skill_dir/SKILL.md``'s frontmatter, or "".

    Deliberately hand-rolled rather than a ``yaml.safe_load`` of the file: the
    frontmatter of a skill written for another tool may carry keys PyYAML
    chokes on, and the only thing needed here is one scalar. A skill whose
    frontmatter cannot be read falls back to its directory name — which is the
    convention anyway — rather than failing the run.
    """
    try:
        text = (skill_dir / "SKILL.md").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if not text.startswith("---"):
        return ""
    _, _, rest = text.partition("\n")
    for line in rest.splitlines():
        if line.strip() == "---":
            break
        key, sep, value = line.partition(":")
        if sep and key.strip() == "name":
            return value.strip().strip("'\"")
    return ""


def resolve_skill_name(reference: str) -> str:
    """A configured skill reference — a name *or* a directory — as its name.

    Scenarios name the skill under test twice: once on the adapter (as a path,
    because something has to be installed) and once on the grader (as a name,
    because something has to be recognised). Accepting either spelling in
    either place removes a whole class of comparison that runs clean and
    measures nothing.
    """
    ref = reference.strip()
    if not ref:
        return ""
    path = Path(ref).expanduser()
    if path.is_dir():
        return read_skill_name(path) or path.name
    return ref


def installed_skills(output_data: Any) -> list[dict[str, Any]]:
    """The install records an adapter left on an agent's output.

    ``CliAgentAdapter`` records what it installed in two places — the
    transcript's metadata, where transcript-scope graders find it, and the
    agent output's, which is what survives into a *result*. Only the second
    reaches a results file: a trace is opt-in and separate, so without this the
    numbers outlive the one fact that says which skill produced them.

    Each record is ``{"name", "source", "path", "digest", "files"}``; ``digest``
    is the hash of the installed tree, and it is the reason this is worth
    carrying — six months on, a path has been reused and the content has not.

    Reads defensively: an adapter that installs nothing, an imported trace, and
    a hand-written results file all legitimately have nothing here.
    """
    if not isinstance(output_data, Mapping):
        return []
    metadata = output_data.get("metadata")
    if not isinstance(metadata, Mapping):
        return []
    entries = metadata.get("skills")
    if not isinstance(entries, list):
        return []
    return [dict(e) for e in entries if isinstance(e, Mapping) and e.get("name")]
