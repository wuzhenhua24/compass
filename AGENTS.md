# AGENTS.md

**Read [CLAUDE.md](CLAUDE.md) first** — it is the same guidance for every agent:
the positioning rule (what belongs to the framework and what to a domain), the
commands, and the three traps (`ruff format`, the mypy ratchet, the lazy
`compass.adapters` import). This file adds only what is not there.

## Tests

- One file per feature, `tests/test_<feature>.py`.
- `asyncio_mode = "auto"` — async tests need no `@pytest.mark.asyncio`. Older
  files still carry it; don't copy it into new ones.
- Deterministic and isolated: fixtures and fakes over network or a real
  filesystem. No coverage threshold is configured; new behavior gets tests for
  the success path and the failure path.

## Commits

Conventional Commits, working directly on `main`. The subject says what changed
about the system, not which files were touched — read `git log` before writing
one.

## Secrets

API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, …) live in the environment,
never in the tree. When touching sandbox logic, verify sensitive env vars stay
blocked — `tests/test_sandbox.py`.
