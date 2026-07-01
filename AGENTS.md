# Repository Guidelines

## Project Structure & Module Organization
- `src/compass/` contains production code, organized by domain:
  - `cli/` for Click commands
  - `core/` for scenario models, runner, trials, and metrics
  - `adapters/`, `graders/`, `eval/`, `sandbox/`, `report/`, `trace/` for integrations and evaluation pipeline
- `tests/` contains pytest suites (`test_*.py`) mirroring package behavior.
- `examples/scenarios/` and `examples/results/` provide runnable sample inputs/outputs.
- `protocols/` and `docs/` store reference material and design notes; keep executable logic in `src/`.

## Build, Test, and Development Commands
- `uv sync --extra dev`: install project + development dependencies.
- `uv run pytest`: run the full test suite.
- `uv run pytest tests/test_cli.py -k init`: run a focused test subset.
- `uv run ruff check src tests`: run lint/import-order checks.
- `uv run mypy src`: run strict static type checks.
- `uv run compass test examples/scenarios/image_gen_test.yaml`: execute a sample scenario through the CLI.

## Coding Style & Naming Conventions
- Target Python `>=3.10`; use 4-space indentation and explicit type hints.
- Follow Ruff defaults configured in `pyproject.toml` (line length `100`, rules `E/W/F/I/B/UP`).
- Mypy is configured as `strict = true`; avoid untyped defs and unchecked `Any`.
- Naming: `snake_case` for modules/functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.

## Testing Guidelines
- Use `pytest` (+ `pytest-asyncio`); async tests should use `@pytest.mark.asyncio`.
- Keep tests deterministic and isolated; prefer fixtures/mocks over external network or filesystem dependencies.
- Place regression tests next to related behavior in `tests/test_<feature>.py`.
- No explicit coverage threshold is configured; new changes should include meaningful tests for success and failure paths.

## Commit & Pull Request Guidelines
- This workspace snapshot does not include `.git` history, so no local commit pattern can be inferred.
- Use Conventional Commits going forward (for example, `feat(cli): add trace format flag`, `fix(sandbox): block leaked env vars`).
- PRs should include:
  - clear problem/solution summary
  - linked issue (if available)
  - test evidence (`uv run pytest`, plus lint/type checks when relevant)
  - sample CLI output or report screenshots for user-facing changes.

## Security & Configuration Tips
- Keep secrets in environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`); never commit them.
- When touching sandbox logic, verify sensitive env vars remain blocked and add/adjust tests in `tests/test_sandbox.py`.
