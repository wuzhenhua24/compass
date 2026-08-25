"""Best-effort secret scrubbing for anything Compass publishes.

Publishing is the boundary this module guards. A results file on your own disk
is yours; a built site is a page you hand to someone, and a trace published
alongside it carries every tool call's arguments verbatim — which is the whole
point of publishing evidence, and also how an ``Authorization: Bearer …`` header
or an ``export OPENAI_API_KEY=…`` ends up on a shareable URL.

So the scrub runs on the way out, unconditionally, and reports what it hit:

    published, hits = redact_json(doc)
    if hits:
        print(f"redacted {sum(hits.values())} secret(s): {', '.join(hits)}")

**Unconditional on purpose.** ``--include-details`` means "show me the
evidence", not "show me the credentials", so there is no flag that turns this
off. The patterns are written narrowly enough that this is a safe trade rather
than a hopeful one: every prefix form requires a token boundary before it, so
``sk-`` does not fire inside ``task-granularity``; the glued form additionally
requires a 20+ character run mixing lower, upper *and* digits, which excludes
ordinary words, lowercase hex ids, and content hashes. Compass's own skill
digests are lowercase hex and never match.

**Structure-aware, so a JSON document cannot match across fields.**
:func:`redact_json` walks the parsed document and scrubs each *string value* on
its own, so `"total_tokens": 1234567890` is never read as a `token = …`
assignment. Only :func:`redact_text`, used on files that are not JSON, sees a
serialized document — and the assignment pattern there still requires the value
to contain a letter, so a long number is never mistaken for a credential.

**Known values beat known shapes.** :func:`env_secret_values` collects the
*values* of environment variables whose names look like credentials, and any
string equal to or containing one is redacted whatever its shape. That is what
catches an in-house token this module has never heard of.

What is *not* covered, and cannot be: a secret with no recognizable prefix, no
credential-shaped name, and no matching environment variable. Read a clean
scrub as "nothing recognized", not as "nothing there" — the same way an empty
``state_delta`` means "not recorded", never "nothing changed".
"""

from __future__ import annotations

import os
import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

#: What replaces the secret part of a match.
PLACEHOLDER = "[REDACTED]"

# How deep into a nested document to walk. Deeper than a trace ever nests.
_MAX_DEPTH = 24

# Shortest environment value worth treating as a secret. Below this, a literal
# match is far more likely to be a coincidence than a credential.
_MIN_ENV_SECRET_CHARS = 8

# Not preceded by a token character. Without it, `sk-` matches inside
# "task-granularity" and quietly mangles ordinary prose.
_EDGE = r"(?<![A-Za-z0-9_-])"

# The signature of a generated key: 20+ alphanumerics mixing lower, upper and
# digits. Used for the glued form (`xsk-Ab1…`), where the token boundary above
# cannot help — it excludes dictionary words, lowercase hex, and short tokens.
_KEY_BODY = (
    r"(?=[A-Za-z0-9]*[a-z])(?=[A-Za-z0-9]*[A-Z])(?=[A-Za-z0-9]*[0-9])[A-Za-z0-9]{20,}"
)

# A value that is a reference rather than a literal — `$TOKEN`, `${TOKEN}`,
# `<your-key>` — is left alone: it is the safe spelling, and redacting it would
# destroy the one thing a reader needs to see. `[` additionally stops a second
# pattern from re-matching a `[REDACTED]` an earlier one already wrote.
_NOT_A_LITERAL = r"(?![$<{\[\s])"

# A credential-shaped name: `api_key`, `MY_SERVICE_TOKEN`, `db-password`.
_SECRET_NAME = (
    r"[A-Za-z0-9_-]*(?:api[_-]?key|apikey|secret|token|password|passwd|credential)"
    r"[A-Za-z0-9_-]*"
)

# A literal value: at least eight characters, one of them a letter, and not one
# an earlier pattern already scrubbed. The letter requirement is what keeps
# `"total_tokens": 1234567890123` out of the assignment patterns; the
# placeholder guard is what stops `KEY=sk-[REDACTED]` being counted a second
# time and losing the `sk-` that says which credential to go rotate.
_SECRET_VALUE = (
    rf"{_NOT_A_LITERAL}(?=[^\s\"',;]*[A-Za-z])(?![^\s\"',;]*\[REDACTED\])[^\s\"',;]{{8,}}"
)

# name -> (pattern, replacement). Ordered most specific first; each is applied
# to the whole string, so an overlapping later pattern simply finds nothing.
_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "openai_key",
        re.compile(rf"{_EDGE}sk-[A-Za-z0-9_-]{{16,}}|sk-{_KEY_BODY}"),
        f"sk-{PLACEHOLDER}",
    ),
    (
        "nvidia_key",
        re.compile(rf"{_EDGE}nvapi-[A-Za-z0-9_-]{{16,}}|nvapi-{_KEY_BODY}"),
        f"nvapi-{PLACEHOLDER}",
    ),
    (
        "github_token",
        re.compile(rf"{_EDGE}(?:gh[pousr]_[A-Za-z0-9]{{20,}}|github_pat_[A-Za-z0-9_]{{20,}})"),
        PLACEHOLDER,
    ),
    (
        "aws_access_key",
        re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA|AROA|AIDA|AIPA|ANPA|ANVA|ABIA|ACCA|AGPA)[A-Z0-9]{16}(?![A-Z0-9])"),
        PLACEHOLDER,
    ),
    (
        "google_key",
        re.compile(rf"{_EDGE}AIza[A-Za-z0-9_-]{{35}}"),
        PLACEHOLDER,
    ),
    (
        "slack_token",
        re.compile(rf"{_EDGE}xox[abporsu]-[A-Za-z0-9-]{{10,}}"),
        PLACEHOLDER,
    ),
    (
        # A JWT is three base64url segments; the header always starts `eyJ`.
        "jwt",
        re.compile(rf"{_EDGE}eyJ[A-Za-z0-9_-]{{10,}}\.[A-Za-z0-9_-]{{10,}}\.[A-Za-z0-9_-]{{5,}}"),
        PLACEHOLDER,
    ),
    (
        "openshift_token",
        re.compile(rf"{_EDGE}sha256~[A-Za-z0-9._~-]{{16,}}"),
        PLACEHOLDER,
    ),
    (
        # Provider-agnostic: keeps the scheme word, drops the credential.
        "bearer_token",
        re.compile(rf"\b(?P<scheme>[Bb]earer\s+){_NOT_A_LITERAL}[A-Za-z0-9._~+/=-]{{20,}}"),
        rf"\g<scheme>{PLACEHOLDER}",
    ),
    (
        # `export MY_SERVICE_TOKEN=…`. An `=` after a credential-shaped name is
        # unambiguous wherever it appears, so this form needs no anchoring.
        "secret_assignment",
        re.compile(rf"(?P<name>{_SECRET_NAME}\s*=\s*[\"']?){_SECRET_VALUE}", re.IGNORECASE),
        rf"\g<name>{PLACEHOLDER}",
    ),
    (
        # `"api_key": "…"`, or a YAML `api_key: …`. The colon form only counts
        # in *field position* — start of a line, or just after a quote, brace or
        # comma. Without that anchor an ordinary sentence ("check the token:
        # authentication is handled upstream") reads as an assignment.
        "secret_field",
        re.compile(
            rf"(?m)(?P<lead>(?:^[ \t-]*|[\"'{{,]\s*)[\"']?{_SECRET_NAME}[\"']?\s*:\s*[\"']?)"
            rf"{_SECRET_VALUE}",
            re.IGNORECASE,
        ),
        rf"\g<lead>{PLACEHOLDER}",
    ),
)


def env_secret_values(environ: dict[str, str] | None = None) -> tuple[str, ...]:
    """Values of environment variables whose names look like credentials.

    Read at publish time. This is the half of the scrub that does not depend on
    recognizing a shape: whatever your provider's token looks like, if it is in
    the environment that produced the run, its literal text comes out.

    Values shorter than 8 characters are ignored — at that length a literal
    match in a transcript is far more likely to be a coincidence than a leak.
    """
    env = os.environ if environ is None else environ
    name = re.compile(
        r"(?:API[_-]?KEY|APIKEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL)", re.IGNORECASE
    )
    values = {
        value
        for key, value in env.items()
        if name.search(key) and len(value.strip()) >= _MIN_ENV_SECRET_CHARS
    }
    # Longest first, so a value that contains another is replaced whole.
    return tuple(sorted(values, key=len, reverse=True))


def redact_text(
    text: str, *, extra_values: Iterable[str] = (), counts: Counter[str] | None = None
) -> str:
    """Scrub one string. *counts* accumulates hits by pattern name if given."""
    if not text:
        return text

    for value in extra_values:
        if value and value in text:
            text = text.replace(value, PLACEHOLDER)
            if counts is not None:
                counts["known_value"] += 1

    for label, pattern, replacement in _PATTERNS:
        text, hits = pattern.subn(replacement, text)
        if hits and counts is not None:
            counts[label] += hits
    return text


def redact_json(
    value: Any, *, extra_values: Iterable[str] | None = None
) -> tuple[Any, Counter[str]]:
    """A scrubbed copy of a parsed document, and what was hit, by pattern name.

    Only *values* are scrubbed. Keys are structure — rewriting one would change
    what the document means, and a key is not where a credential lives.

    ``extra_values`` defaults to :func:`env_secret_values`; pass ``()`` to scrub
    by shape alone.
    """
    known = tuple(env_secret_values() if extra_values is None else extra_values)
    counts: Counter[str] = Counter()

    def walk(node: Any, depth: int) -> Any:
        if depth > _MAX_DEPTH:
            return node
        if isinstance(node, str):
            return redact_text(node, extra_values=known, counts=counts)
        if isinstance(node, dict):
            return {k: walk(v, depth + 1) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(item, depth + 1) for item in node]
        if isinstance(node, tuple):
            return tuple(walk(item, depth + 1) for item in node)
        return node

    return walk(value, 0), counts


def summarize(counts: Counter[str]) -> str:
    """A one-line account of a scrub, or ``""`` when it found nothing."""
    total = sum(counts.values())
    if not total:
        return ""
    kinds = ", ".join(f"{label}×{n}" for label, n in counts.most_common())
    noun = "secret" if total == 1 else "secrets"
    return f"redacted {total} {noun} ({kinds})"
