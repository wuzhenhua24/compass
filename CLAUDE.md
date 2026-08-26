# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Compass is a **substrate for Agent evaluation** — not a "evals everything out of the box" framework. It provides the reusable spine (a standard Transcript/Outcome model + ToolCall protocol, trace ingestion, domain-agnostic *process* graders, and reliability metrics); domain *correctness* graders and datasets are user code that plugs in (like `tests/` using pytest). Design inspired by [Anthropic: Demystifying Evals for AI Agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents).

Positioning discipline (keep this in mind when extending): the core stays small. Process/reliability checks (TRANSCRIPT scope: cost, latency, loops, tool usage, dangerous-op execution) are reusable and belong in the framework; correctness checks (OUTCOME scope: is the answer/image/code right) are domain-specific and belong in user graders. Domain-specific fields (e.g. `expected_doc`, `key_facts`) go in grader config or a user harness, NOT the core Scenario/GradeContext models — resist adding them to core. See README "定位：是什么 / 不是什么".

**Where that line falls in the tree.** `compass/graders/` holds the framework's 20 (`code/common/`, `model/`, `human/`); `compass/graders/domains/` holds the 23 that ship as worked examples of user code, one package per domain (`coding`, `data`, `image`), imported lazily on a registry miss so `import compass` costs nothing for a domain nobody is evaluating. A new *process* grader goes in `code/common/`. A new *correctness* grader goes in a domain — and if it does not fit one of the three, that is the signal it is user code, not a core addition. Optional backends are extras named after the domain (`compass[data]`, `compass[image]`), imported inside a try/except so a missing one degrades the grader instead of breaking the import.

One field already sits on the wrong side of this line and is not being moved: `Outcome.image` is a first-class field of the core transcript model, so Pillow is a hard dependency and image is the one domain the core knows about by name. Pulling it out means generalizing `Outcome` to artifacts across seven core modules — a deliberate, separate change, not something to do in passing.

## Common Commands

Everything runs through `uv run` — Compass is not installed on PATH in a source
checkout, `uv sync` only puts it in `.venv`.

`uv sync` gives a working dev environment: the test/lint/type tooling and the
`data` domain's backends live in the PEP 735 `dev` group, which uv installs by
default. The `image` domain's backends (torch, open-clip) are the `image`
*extra* and are not installed — the image graders degrade without them and the
suite is fully green either way. Working on that domain: `uv sync --extra image`.

```bash
# Install dependencies
uv sync

# Run all tests (2257 as of now; keep them green)
uv run pytest tests/

# Run a single test file
uv run pytest tests/test_scenario.py

# Run a specific test
uv run pytest tests/test_scenario.py::test_function_name -v

# Lint — clean; keep it that way
uv run ruff check .
uv run ruff check . --fix     # safe autofixes only

# Type check — green via a ratchet (see below)
uv run mypy src/compass
```

**Do not run `ruff format .`** — the codebase has never been formatter-managed,
so it would reformat 168 of the 252 tracked `.py` files (~12k lines) and bury
real changes. Match the surrounding style by hand instead. Adopting the
formatter is a deliberate, separate commit if it ever happens.

**The mypy ratchet.** `uv run mypy src/compass` passes, but that is a floor, not
a clean bill of health: 30 modules are quarantined by `ignore_errors` in
`pyproject.toml` and still carry 87 findings (measured by deleting the block and
re-running). Everything else among the 114 checked files is gated —
**new code and edits to clean modules must type-check**. Never add a module to
that list to make an error go away; fix the annotation, or say so explicitly.
Removing an entry (and fixing what mypy then reports) is always welcome.

```bash
# CLI usage
uv run compass test <scenario.yaml>          # Run tests
uv run compass test scenarios/ --parallel -w 4   # Parallel execution
uv run compass test qa.yaml -m gpt-5 -m claude-5 # Multi-model leaderboard
uv run compass grade ./traces -s qa.yaml     # Offline grading, no agent re-run
uv run compass analyze results/              # Analyze results
uv run compass compare a.json b.json         # Paired comparison (flips + CI)
uv run compass insights results.json        # LLM conclusions, each checked against the data
uv run compass site build results.json -o site/  # Publish a static site
uv run compass site compare a.json b.json -o site/  # Publish a paired comparison
uv run compass site serve results.json       # Live view, recomputed per request
uv run compass trace results/case.json       # View transcript
uv run compass import session.jsonl          # Import trace (pi/codex/OTLP/Claude/OpenAI/ATIF)
uv run compass docs [topic]                  # Read Compass's own docs
uv run compass list                          # List registered graders/adapters
```

## Architecture

### Four-Layer Design

```
1. Interface Layer      → CLI (compass.cli.main), Python SDK
2. Test Orchestration   → Scenario Engine, Trial Manager, Parallel Executor
3. Core Engine          → Grader System (3-tier), Transcript Collector, Report Gen
4. Agent Adapter Layer  → image, coding, environment, claude_code, pi, codex adapters (registry-based;
                          adapters/cli_agent.py is the shared base of the last three, and
                          adapters/llm.py holds LLMToolCallMixin; neither is a registered adapter)

Cutting across 3 and 4: compass/llm/ — pricing, token-usage extraction and the
structured-completion client. Control plane, and deliberately not under
adapters/: a trace importer or an LLM judge needs all three, and used to have to
import the whole adapter registry (and dodge a cycle) to get them.
```

### Core Concepts

| Concept | Description |
|---------|-------------|
| **Task** | Single test case with input, expectations, and graders |
| **Trial** | One execution attempt of a Task (supports multiple trials) |
| **Grader** | Scorer with three types: Code (deterministic), Model (LLM), Human |
| **Transcript** | Execution trace - HOW the agent worked (tool calls, timing, cost) |
| **Outcome** | Final result - WHAT the agent produced (image, data, blocked status) |
| **GraderScope** | Data requirement: `OUTCOME` / `TRANSCRIPT` / `BOTH` |

### Transcript/Outcome Separation (Key Design)

Graders declare their data needs via `GraderScope`:
- **OUTCOME graders** - Evaluate final output only (exact_match, json_schema)
- **TRANSCRIPT graders** - Evaluate execution process only (cost_budget, tool_usage)
- **BOTH graders** - Need both for efficiency analysis (efficiency, external_checker)

This enables independent evaluation of "did it work?" vs "did it work efficiently?"

### Three-Tier Grader System

The tier says *how* a grader decides; it says nothing about where the code
lives. Examples below are the framework's own — see the positioning section for
why `semantic_match`, `vlm_judge` and `safety_check` are Model graders that live
under `domains/image/` rather than under `model/`.

```
Code Graders (deterministic)  → json_schema, style_convention, tool_usage
Model Graders (LLM-based)     → rubric, trajectory_judge, groundedness
Human Graders                 → human_review (expert annotation)
```

### Code Organization

```
src/compass/
├── cli/
│   ├── app.py            # the Click group + shared console
│   ├── main.py           # entry point: imports commands/, which registers them
│   └── commands/         # one module per command (run, grade, compare, site, …)
├── core/
│   ├── runner.py         # Compass class - main test runner
│   ├── scenario.py       # Scenario/TestCase/GraderConfig models
│   ├── trial.py          # TrialManager, multi-attempt testing
│   ├── transcript.py     # TranscriptRecorder, Outcome, Transcript
│   └── result.py         # EvalResult, TestStatus
├── graders/
│   ├── base.py           # Grader, GraderScope, GradeContext, GradeResult
│   ├── registry.py       # @register_grader; loads domains/ on a lookup miss
│   ├── content.py        # extract_content — pull the graded subject out of a context
│   ├── code/common/      # the framework's Code graders (process & reliability)
│   ├── model/            # the framework's Model graders (rubric, trajectory, groundedness)
│   ├── human/            # human_review, pairwise_comparison
│   └── domains/          # domain CORRECTNESS, lazily loaded: coding / data / image
├── adapters/             # Agent adapters (image, coding, environment, claude_code, pi, codex)
├── integrations/         # Trace importers (pi, codex, otlp, claude_agent, openai_agents, atif)
│                         #   _common.py holds what all six ask of untyped JSON
├── sandbox/              # Execution sandbox — used by the coding/environment
│                         #   adapters and the test_runner/integration_test graders
└── report/               # console, html, site, analyzer, compare, insights
```

## Key Patterns

### Creating Custom Graders

```python
from compass.graders import CodeGrader, GradeContext, GradeResult, GraderScope, register_grader

@register_grader("my_grader")
class MyGrader(CodeGrader):
    grader_scope = GraderScope.OUTCOME  # Declare data requirement

    async def grade(self, context: GradeContext) -> GradeResult:
        image = context.image  # Access via context properties
        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=True,
            score=1.0,
            details={},
        )
```

### Scenario YAML Structure

```yaml
name: "Test Scenario"
agent:
  adapter: image
  endpoint: "http://localhost:8000"

cases:
  - id: "test_case"
    input:
      prompt: "Generate an image"
    expect: pass  # or fail for negative tests
    graders:
      - name: semantic_match
        config:
          threshold: 0.25
    aggregation:
      method: weighted_sum
      pass_threshold: 0.7
```

### Metrics

- `pass@k` = P(at least 1 success in k attempts) - for exploration
- `pass^k` = P(all k attempts successful) - for reliability

## Testing Notes

- Tests use `pytest-asyncio` with `asyncio_mode = "auto"`
- Test files in `tests/` directory mirror the source structure
- Run specific grader tests: `pytest tests/test_*_graders.py`

## document
 - 每次增加新功能特性，请更新到文档和 interview.md 文件中。
 - 文档结构：README.md 只保留骨架（定位/概念/架构/CLI/快速开始/导航）；细节按主题放在 docs/ 专题文档——cheatsheet.md（单页速查，agent 入口）、core-design.md（Transcript/Outcome、ToolCall 协议）、graders.md（内置与自定义评分器）、scenario-config.md（YAML 与指标）、analysis.md（analyze/compare/报告）、integrations.md（轨迹导入与 Adapter）、skills.md（Agent Skill 评测）、roadmap.md（路线图归档）。新特性写进对应专题文档，README 只在导航表/特性要点里加一句。
 - 这 8 份专题文档同时是 `compass docs <topic>` 的内容源，且在 pyproject 的 `force-include` 里逐个打进 wheel。**新增专题文档要三处同步**：`src/compass/docs_index.py` 的 TOPICS、pyproject 的 force-include、README 导航表——漏掉任一处，装出来的包就读不到它。
 - 改评分器数量时记得同步：README 导航表两处 + 安装校验那行、docs/graders.md 的分组表格、docs/cheatsheet.md。数量以 `list_graders()` 为准（当前 43 = 框架 20 + 领域 23）；`compass.graders.domains.domain_of(name)` 给出某个 grader 属于哪一半。改完可以跑这个核对，不要靠人数：
   ```bash
   uv run python -c "from compass.graders import list_graders; from compass.graders.domains import domain_of; import collections; print(collections.Counter(domain_of(g) for g in list_graders()))"
   ```
 - docs/ 下的 *.html、*_files/、interview.md、todos.md、idea.md 等是本地参考资料（gitignore），专题 *.md 文档是版本库的一部分。
