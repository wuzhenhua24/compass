"""Tests for ``tree_digest`` — the identity a skill is joined on.

A digest outlives the run that produced it. It is written into results files
and into a published ``run.json``, it is what a trend line over a skill's
versions is keyed on, and an external tool that scanned the *source* of a skill
has to be able to recompute it and match what Compass recorded for the
*installed* copy. That makes the algorithm a contract rather than an
implementation detail, and a contract needs a test that fails loudly when it
changes — because the way this breaks in the wild is silence. Nothing raises;
the two sides simply stop lining up.

So the golden value below is the point of this file. Every other test here
explains one clause of the contract; that one enforces all of them at once.
"""

from __future__ import annotations

import os
import stat

import pytest

from compass.core.skills import (
    SKILL_DIGEST_CHARS,
    SKILL_DIGEST_VERSION,
    tree_digest,
)


def _skill(root, files: dict[str, str]):
    """Write ``{relative path: content}`` under *root* and return it."""
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


FIXTURE = {
    "SKILL.md": "---\nname: fixture\n---\nbody\n",
    "scripts/run.py": "print(1)\n",
}


# ---------------------------------------------------------------------------
# The golden value
# ---------------------------------------------------------------------------


def test_the_digest_of_a_known_tree_is_this_exact_string(tmp_path):
    """Changing the algorithm invalidates every digest already stored, and
    every join an external tool built on one. If this test fails and the change
    was deliberate, bump ``SKILL_DIGEST_VERSION`` and say so in docs/skills.md
    — do not update the constant to match the new output."""
    assert tree_digest(_skill(tmp_path, FIXTURE)) == ("fa0b45886d9f932b", 2)


def test_the_contract_version_is_declared():
    assert SKILL_DIGEST_VERSION == "1"
    assert SKILL_DIGEST_CHARS == 16


# ---------------------------------------------------------------------------
# One test per clause, so a failure says which clause broke
# ---------------------------------------------------------------------------


def test_identical_trees_in_different_places_agree(tmp_path):
    """The whole point: a scanner reads the source, Compass hashes the
    installed copy, and the two have to meet."""
    source = _skill(tmp_path / "source", FIXTURE)
    installed = _skill(tmp_path / "elsewhere" / ".claude" / "skills" / "x", FIXTURE)
    assert tree_digest(source) == tree_digest(installed)


def test_renaming_a_file_makes_a_different_skill(tmp_path):
    """Paths are hashed alongside bytes, so this differs even though every
    byte of content is unchanged."""
    a = _skill(tmp_path / "a", FIXTURE)
    renamed = {"SKILL.md": FIXTURE["SKILL.md"], "scripts/go.py": FIXTURE["scripts/run.py"]}
    b = _skill(tmp_path / "b", renamed)
    assert tree_digest(a)[0] != tree_digest(b)[0]


def test_one_changed_byte_makes_a_different_skill(tmp_path):
    a = _skill(tmp_path / "a", FIXTURE)
    b = _skill(tmp_path / "b", {**FIXTURE, "scripts/run.py": "print(2)\n"})
    assert tree_digest(a)[0] != tree_digest(b)[0]


def test_the_nul_separator_keeps_a_boundary_shift_from_colliding(tmp_path):
    """Without the separator, path+content concatenate ambiguously: moving one
    character across the boundary would feed the hash the same bytes."""
    a = _skill(tmp_path / "a", {"ab": "c"})
    b = _skill(tmp_path / "b", {"a": "bc"})
    assert tree_digest(a)[0] != tree_digest(b)[0]


def test_the_digest_is_truncated_to_the_declared_width(tmp_path):
    digest, _ = tree_digest(_skill(tmp_path, FIXTURE))
    assert len(digest) == SKILL_DIGEST_CHARS
    assert all(c in "0123456789abcdef" for c in digest)


def test_creation_order_does_not_matter(tmp_path):
    """Files are sorted by path, so a filesystem that walks them differently
    still produces the same digest."""
    a = _skill(tmp_path / "a", {"z.md": "1", "a.md": "2"})
    b = _skill(tmp_path / "b", {"a.md": "2", "z.md": "1"})
    assert tree_digest(a) == tree_digest(b)


def test_an_empty_directory_does_not_change_the_digest(tmp_path):
    """Only files are hashed. Copying a tree need not preserve empty dirs for
    the two sides to agree."""
    plain = _skill(tmp_path / "a", FIXTURE)
    with_empty = _skill(tmp_path / "b", FIXTURE)
    (with_empty / "assets").mkdir()
    assert tree_digest(plain) == tree_digest(with_empty)


def test_the_file_count_is_files_not_entries(tmp_path):
    root = _skill(tmp_path, FIXTURE)
    (root / "references").mkdir()
    assert tree_digest(root)[1] == 2


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads unreadable files anyway")
def test_an_unreadable_file_is_hashed_as_the_fact_not_skipped(tmp_path):
    """An unreadable file must not be silently equal to an absent one."""
    absent = _skill(tmp_path / "a", {"SKILL.md": FIXTURE["SKILL.md"]})
    blocked = _skill(tmp_path / "b", {**FIXTURE, "SKILL.md": FIXTURE["SKILL.md"]})
    (blocked / "scripts" / "run.py").chmod(0o000)
    try:
        assert tree_digest(blocked)[1] == 2
        assert tree_digest(blocked)[0] != tree_digest(absent)[0]
    finally:
        (blocked / "scripts" / "run.py").chmod(stat.S_IRUSR | stat.S_IWUSR)


def test_an_empty_tree_still_produces_a_digest(tmp_path):
    (tmp_path / "empty").mkdir()
    digest, count = tree_digest(tmp_path / "empty")
    assert count == 0
    assert len(digest) == SKILL_DIGEST_CHARS


# ---------------------------------------------------------------------------
# The value that actually ships
# ---------------------------------------------------------------------------


def test_the_adapter_records_the_same_digest_this_function_computes():
    """``cli_agent`` used to carry its own copy. If the two ever diverge, every
    published digest stops being recomputable from the source."""
    from compass.adapters import cli_agent

    assert cli_agent.tree_digest is tree_digest
