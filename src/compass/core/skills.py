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

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

#: Version of the :func:`tree_digest` algorithm. A digest is an identity that
#: outlives the run that produced it — it is published in ``run.json``, it is
#: what a trend line over a skill's versions is keyed on, and an external tool
#: has to be able to recompute it to line its own report up with a Compass run.
#: So the algorithm is a **contract**, in the same sense as
#: ``TOOLCALL_PROTOCOL_VERSION`` and ``compass.run/1``: changing it invalidates
#: every stored digest and every join anyone built on one, and the change must
#: be a deliberate, visible bump rather than a refactor that happens to pass.
SKILL_DIGEST_VERSION = "1"

#: Hex characters kept from the SHA-256. Part of the contract — a caller that
#: does not truncate identically produces a different string for the same tree.
SKILL_DIGEST_CHARS = 16


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


def tree_digest(root: Path) -> tuple[str, int]:
    """``(digest, file_count)`` identifying the exact content of a directory.

    **This is a published contract** (``SKILL_DIGEST_VERSION``), not an
    implementation detail. The value it returns is stored in results files and
    in ``run.json``, so it has to mean the same thing across Compass versions,
    and an external tool that scans the *source* of a skill has to be able to
    recompute it over that source and match what Compass recorded for the
    *installed* copy. Installation is a whole-tree copy, so identical trees
    give identical digests — as long as the algorithm is identical too.

    The algorithm, exactly:

    1. Every file under *root*, recursively. Directories are not visited as
       entries of their own; an empty directory does not affect the digest.
    2. Sorted by path — filesystem walk order is not stable, and an unsorted
       digest is not even reproducible against itself.
    3. For each file, feed SHA-256 three things in order: the path relative to
       *root* as UTF-8, a single ``NUL`` byte, then the file's bytes.
    4. A file that cannot be read contributes the literal ``<unreadable>``
       instead of its bytes — the *fact* is hashed rather than skipped, so an
       unreadable file is not silently the same as an absent one.
    5. Truncate the hex digest to ``SKILL_DIGEST_CHARS`` characters.

    Two consequences worth stating, because both are deliberate:

    - **Paths are hashed alongside bytes**, so renaming a reference file makes
      a different skill even when every byte of content is unchanged.
    - **The ``NUL`` separator is load-bearing.** Without it a path and the
      content that follows it concatenate ambiguously, and two different trees
      can collide.

    Each of those steps changes the output if a reimplementation gets it wrong,
    and — this is the reason the contract is written down — **it changes it
    silently**. Nothing raises; the two sides simply never line up again, and
    a product that never lines up invites someone to "fix" it by matching on
    the skill's name instead. Names are exactly what cannot distinguish
    versions here: ``report-writer-v1`` and ``report-writer-v2`` both install
    as ``report-writer``, on purpose, because renaming the thing under test
    would defeat the comparison. The digest is the only thing that says which.
    """
    h = hashlib.sha256()
    count = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(str(path.relative_to(root)).encode("utf-8"))
        h.update(b"\0")
        try:
            h.update(path.read_bytes())
        except OSError:  # unreadable file: hash the fact, not the content
            h.update(b"<unreadable>")
        count += 1
    return h.hexdigest()[:SKILL_DIGEST_CHARS], count


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
