# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Compass is a universal Agent QA framework for testing and evaluating AI Agents. Design inspired by [Anthropic: Demystifying Evals for AI Agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents).

## Common Commands

```bash
# Install dependencies
uv sync

使用uv运行测试
# Run all tests
uv run pytest tests/

# Run a single test file
uv run pytest tests/test_scenario.py

# Run a specific test
uv run pytest tests/test_scenario.py::test_function_name -v

# Lint
ruff check .

# Format
ruff format .

# Type check
mypy src/compass

# CLI usage
compass test <scenario.yaml>              # Run tests
compass test scenarios/ --parallel -w 4   # Parallel execution
compass analyze results/                  # Analyze results
compass trace results/case.json           # View transcript
```

## Architecture

### Four-Layer Design

```
1. Interface Layer      → CLI (compass.cli.main), Python SDK
2. Test Orchestration   → Scenario Engine, Trial Manager, Parallel Executor
3. Core Engine          → Grader System (3-tier), Transcript Collector, Report Gen
4. Agent Adapter Layer  → image, llm, coding adapters (registry-based)
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
├── adapters/             # Agent adapters (image, llm, coding)
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
 - 每次增加新功能特性，请更新到 README.md 和 interview.md 文件中。
