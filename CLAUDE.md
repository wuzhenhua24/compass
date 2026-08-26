# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## Positioning — the line to hold

Compass is a **substrate**: the framework owns the reusable spine
(Transcript/Outcome, the ToolCall protocol, trace ingestion, domain-agnostic
*process* graders, reliability metrics), while *correctness* graders and datasets
are user code that plugs in, the way `tests/` plugs into pytest. So every
addition picks a side — process/reliability (TRANSCRIPT scope: cost, latency,
loops, tool usage, dangerous ops) goes in `graders/code/common/`; correctness
(OUTCOME scope: is the answer/image/code right) goes in
`graders/domains/{coding,data,image}/`; fitting none of the three is the signal
it is user code, not a core addition. Two corollaries: domain fields
(`expected_doc`, `key_facts`, …) live in grader config or a user harness, never
in core `Scenario`/`GradeContext`, and domain backends are extras
(`compass[data]`, `compass[image]`) imported in a try/except so a missing one
degrades the grader instead of breaking the import — with `Outcome.image` the
one deliberate exception, which is why Pillow is a hard dependency and image is
the one domain core knows by name.

## Commands

Everything runs through `uv run` — a source checkout is not on PATH. `uv sync`
alone gives a working dev environment (the `dev` group carries test/lint/type
tooling and the `data` backends). The `image` backends are an extra
(`uv sync --extra image`); without them the image graders degrade and the suite
is still green.

```bash
uv run pytest tests/       # keep green
uv run ruff check .        # keep clean
uv run mypy src/compass    # green via a ratchet, see below
```

CLI usage: `uv run compass --help`. Compass's own docs: `uv run compass docs <topic>`.

## Three things that bite

- **Never run `ruff format .`** — this codebase has never been formatter-managed;
  it would reformat ~12k lines and bury the real change. Match surrounding style
  by hand. Adopting the formatter is a separate, deliberate commit.
- **mypy is a ratchet**, not a clean bill of health: `pyproject.toml` quarantines
  modules with `ignore_errors` that still carry real findings. New code and edits
  to clean modules must type-check. **Never add a module to that list to silence
  an error** — fix it, or say so. Removing entries is always welcome.
- **`import compass` must not import `compass.adapters`** — `Compass` comes from a
  PEP 562 `__getattr__` in `compass/__init__.py`; an eager import of
  `compass.core.runner` there silently undoes it. `tests/test_package_api.py`
  asserts this in a subprocess.

## Two placements the tree doesn't explain

`compass/llm/` (pricing, usage, structured completions) and `compass/sandbox/`
(isolated workdir + exec + collect) do **not** belong under `adapters/`: the
former is needed by trace importers and LLM judges, the latter has six callers
(two adapters plus four `domains/coding/` graders). Don't relocate them by
association.

## Metrics

`pass@k` = P(at least 1 success in k) — exploration. `pass^k` = P(all k succeed)
— reliability.

## 文档

- 每次增加新功能特性，更新对应专题文档和 `docs/interview.md`。
- README.md 只留骨架（定位/概念/架构/CLI/快速开始/导航），细节进 `docs/` 专题文档：
  cheatsheet / core-design / graders / scenario-config / analysis / integrations /
  skills / roadmap。新特性写进专题文档，README 只在导航表加一句。
- **新增专题文档要三处同步**：`src/compass/docs_index.py` 的 TOPICS、pyproject 的
  force-include、README 导航表——漏一处，装出来的包就读不到它。
- **改 grader 数量要同步**：README 导航表两处 + 安装校验那行、docs/graders.md 分组
  表格、docs/cheatsheet.md。数量以这条命令为准，不要靠人数：
  ```bash
  uv run python -c "from compass.graders import list_graders; from compass.graders.domains import domain_of; import collections; print(collections.Counter(domain_of(g) for g in list_graders()))"
  ```
- docs/ 下的 *.html、*_files/、interview.md、todos.md、idea.md 是本地参考资料
  （gitignore），专题 *.md 文档是版本库的一部分。
