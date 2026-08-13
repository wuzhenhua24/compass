# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Compass is a **substrate for Agent evaluation** — not a "evals everything out of the box" framework. It provides the reusable spine (a standard Transcript/Outcome model + ToolCall protocol, trace ingestion, domain-agnostic *process* graders, and reliability metrics); domain *correctness* graders and datasets are user code that plugs in (like `tests/` using pytest). Design inspired by [Anthropic: Demystifying Evals for AI Agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents).

Positioning discipline (keep this in mind when extending): the core stays small. Process/reliability checks (TRANSCRIPT scope: cost, latency, loops, tool usage, dangerous-op execution) are reusable and belong in the framework; correctness checks (OUTCOME scope: is the answer/image/code right) are domain-specific and belong in user graders. Domain-specific fields (e.g. `expected_doc`, `key_facts`) go in grader config or a user harness, NOT the core Scenario/GradeContext models — resist adding them to core. See README "定位：是什么 / 不是什么".

## Common Commands

Everything runs through `uv run` — Compass is not installed on PATH in a source
checkout, `uv sync` only puts it in `.venv`.

```bash
# Install dependencies
uv sync

# Run all tests (2044 as of now; keep them green)
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
so it would reformat 113 of 151 files (~8.8k lines) and bury real changes. Match
the surrounding style by hand instead. Adopting the formatter is a deliberate,
separate commit if it ever happens.

**The mypy ratchet.** `uv run mypy src/compass` passes, but that is a floor, not
a clean bill of health: 31 modules are quarantined by `ignore_errors` in
`pyproject.toml` and still carry ~110 findings. The other 53 modules are gated —
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
uv run compass site build results.json -o site/  # Publish a static site
uv run compass site compare a.json b.json -o site/  # Publish a paired comparison
uv run compass site serve results.json       # Live view, recomputed per request
uv run compass trace results/case.json       # View transcript
uv run compass import session.jsonl          # Import trace (pi/codex/OTLP/Claude/OpenAI)
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
                          adapters/llm.py provides LLM mixins/pricing; neither is a registered adapter)
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
- **OUTCOME graders** - Evaluate final output only (semantic_match, aesthetic_score)
- **TRANSCRIPT graders** - Evaluate execution process only (cost_budget, tool_usage)
- **BOTH graders** - Need both for efficiency analysis

This enables independent evaluation of "did it work?" vs "did it work efficiently?"

### Three-Tier Grader System

```
Code Graders (deterministic)  → image_assertions, json_schema, style_convention
Model Graders (LLM-based)     → semantic_match, vlm_judge, rubric, safety_check
Human Graders                 → human_review (expert annotation)
```

### Code Organization

```
src/compass/
├── cli/main.py           # CLI entry point
├── core/
│   ├── runner.py         # Compass class - main test runner
│   ├── scenario.py       # Scenario/TestCase/GraderConfig models
│   ├── trial.py          # TrialManager, multi-attempt testing
│   ├── transcript.py     # TranscriptRecorder, Outcome, Transcript
│   └── result.py         # EvalResult, TestStatus
├── graders/
│   ├── base.py           # Grader, GraderScope, GradeContext, GradeResult
│   ├── registry.py       # @register_grader decorator
│   ├── code/             # Deterministic graders by domain
│   │   ├── common/       # style_convention, json_schema, structure_check
│   │   ├── coding/       # functional, quality, diff, security graders
│   │   ├── data/         # SQL, data correctness, query quality
│   │   └── image/        # image_assertions, technical_quality
│   └── model/            # LLM-based graders (semantic, vlm, rubric, safety)
├── adapters/             # Agent adapters (image, coding, environment, claude_code, pi, codex)
└── report/               # Console and HTML reporting, analyzer
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
 - 改评分器数量时记得同步：README 导航表两处、docs/graders.md 的"全部 N 个"与其表格、docs/cheatsheet.md。数量以 `list_graders()` 为准（当前 42）。
 - docs/ 下的 *.html、*_files/、interview.md、todos.md、idea.md 等是本地参考资料（gitignore），专题 *.md 文档是版本库的一部分。
