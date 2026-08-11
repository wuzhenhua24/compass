# 核心设计：Transcript / Outcome 分离与 ToolCall 协议

> Compass 专题文档 · 返回 [README](../README.md)


## Transcript / Outcome 分离设计

> 设计灵感来源：[Anthropic: Demystifying Evals for AI Agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)

这是 Compass 最核心的架构设计之一。Transcript（执行轨迹）和 Outcome（最终结果）是两个**独立的数据维度**，评分器可以选择性地、独立地访问其中任意一个或两者。

### 为什么要分离？

Agent 的执行包含两个本质不同的方面：

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Agent 执行的两个维度                              │
├──────────────────────────────┬──────────────────────────────────────┤
│       Transcript（过程）       │          Outcome（结果）              │
│       HOW the agent worked   │      WHAT the agent produced        │
├──────────────────────────────┼──────────────────────────────────────┤
│ • 调用了哪些工具              │ • 最终生成的图像                      │
│ • 每步的执行耗时              │ • 输出的质量和属性                    │
│ • 推理过程和决策链            │ • 是否被安全过滤器拦截                │
│ • 消耗的 token / 成本        │ • 输出的元数据                       │
│ • 错误和重试行为              │ • 最终环境状态                       │
├──────────────────────────────┼──────────────────────────────────────┤
│ 回答：Agent 做事效率如何？     │ 回答：Agent 做出了什么？              │
│       是否遵循了约束？         │       结果是否符合预期？              │
└──────────────────────────────┴──────────────────────────────────────┘
```

分离的关键价值在于：

- **单独评估结果**（Outcome）：不关心 Agent 用了什么路径，只关心最终输出是否正确。同一张高质量图像，不管是通过 LoRA 还是 ControlNet 生成的，都应该拿到高分。
- **单独评估过程**（Transcript）：即使最终结果正确，也可以评估 Agent 是否高效工作——有没有浪费 token？有没有反复重试？有没有超出成本预算？
- **联合评估**（Both）：综合考量过程与结果的关系——投入的工作量是否与产出质量成正比？

### GraderScope：评分器声明数据需求

每个评分器通过 `GraderScope` 声明自己需要访问哪些数据：

```python
from enum import Enum

class GraderScope(str, Enum):
    OUTCOME    = "outcome"      # 只需要最终结果
    TRANSCRIPT = "transcript"   # 只需要执行轨迹
    BOTH       = "both"         # 两者都需要
```

框架会在调用评分器前自动验证所需数据是否可用，如果缺失会直接返回错误而非静默失败。

### GradeContext：统一上下文，独立访问

`GradeContext` 是传递给所有评分器的核心数据对象，它将 Transcript 和 Outcome 作为**一等公民**：

```python
@dataclass
class GradeContext:
    # === 输入 ===
    prompt: str = ""
    negative_prompt: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    # === Transcript: 执行轨迹 ===
    transcript: Transcript | None = None

    # === Outcome: 最终结果 ===
    outcome: Outcome | None = None

    # === 参考数据 ===
    reference_image: Image.Image | None = None
```

评分器通过便捷属性独立访问所需数据：

```python
# Outcome 评分器访问最终结果
image = context.image               # → outcome.image
is_blocked = context.is_blocked     # → outcome.blocked

# Transcript 评分器访问执行轨迹
tools = context.tool_calls          # → transcript.tool_calls
steps = context.reasoning_steps     # → transcript.reasoning_steps
duration = context.total_duration_ms  # → transcript.total_duration_ms
```

### 三种作用域的实际应用

**Outcome 评分器** — 评估最终产出物：

```python
@register_grader("semantic_match")
class SemanticMatchGrader(ModelGrader):
    grader_scope = GraderScope.OUTCOME  # 只看结果

    async def grade(self, context: GradeContext) -> GradeResult:
        image = context.image           # ← 访问 Outcome
        # 用 CLIP 计算图文相似度，不关心生成过程
        similarity = await self._clip_score(image, context.prompt)
        ...
```

**Transcript 评分器** — 评估执行过程：

```python
@register_grader("cost_budget")
class CostBudgetGrader(CodeGrader):
    grader_scope = GraderScope.TRANSCRIPT  # 只看过程

    async def grade(self, context: GradeContext) -> GradeResult:
        tool_calls = context.tool_calls  # ← 访问 Transcript
        # 统计工具调用成本，不关心最终图像
        total_cost = sum(self._tool_cost(tc) for tc in tool_calls)
        ...
```

**联合评分器** — 综合评估效率：

```python
@register_grader("efficiency")
class EfficiencyGrader(CodeGrader):
    grader_scope = GraderScope.BOTH  # 同时看过程和结果

    async def grade(self, context: GradeContext) -> GradeResult:
        tool_count = len(context.tool_calls)  # ← Transcript
        has_output = context.image is not None  # ← Outcome
        # 比较投入（过程复杂度）与产出（是否有结果）
        ...
```

### Outcome vs Path 原则

> "Grade outcomes rather than paths" — 评估结果，而非路径

分离设计自然支持这一原则。Outcome 评分器天然遵循此原则——它们**无法**访问执行路径，只能看到最终结果：

```yaml
# ❌ 在 Outcome 评分器中检查执行路径（做不到，也不应该做）
graders:
  - type: model
    criteria:
      - "Agent 是否先加载了 LoRA"
      - "Agent 是否使用了 ControlNet"

# ✅ Outcome 评分器检查最终结果
graders:
  - type: model
    criteria:
      - "最终图像是否包含一只猫"
      - "图像风格是否符合赛博朋克美学"
      - "整体构图是否平衡"
```

而 Transcript 评分器则用于那些**确实需要关注过程**的场景（成本控制、延迟要求、合规审计），这些场景不适合用 Outcome 评分器来处理。

### Code-First 短路模式

Compass 支持 **Code-First 短路模式**，优化评估成本和速度：

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Code-First 短路执行流程                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Phase 1: Code Graders (快速、低成本、确定性)                        │
│   ┌─────────────────────────────────────────────────────────┐       │
│   │  safety_check → json_schema → style_convention → ...    │       │
│   └─────────────────────────────────────────────────────────┘       │
│                              ↓                                      │
│                      短路条件检查                                    │
│                     /                  \                            │
│              触发短路                不触发                          │
│                   ↓                      ↓                          │
│            立即返回失败              Phase 2: Model Graders          │
│           (跳过 Model,               ┌────────────────────┐         │
│            节省成本)                  │ semantic_match     │         │
│                                      │ rubric             │         │
│                                      │ vlm_judge          │         │
│                                      └────────────────────┘         │
│                                              ↓                      │
│                                         聚合结果                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**短路模式配置：**

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| `disabled` | 不短路，全部运行（默认） | 需要完整评估结果 |
| `code_fail` | 任何 Code Grader 失败时短路 | 激进：快速失败，节省成本 |
| `code_pass` | 所有 Code Grader 通过时短路 | 规则确认即可，失败时需模型判断 |
| `required_fail` | required 的 Code Grader 失败时短路 | 推荐：平衡成本和完整性 |

**配置示例：**

```yaml
aggregation:
  method: weighted_sum
  pass_threshold: 0.7
  required_graders: [safety_check, json_schema]
  short_circuit: required_fail  # 仅当 required Code Grader 失败时跳过 Model Graders
```

**收益对比：**

| 场景 | 默认模式 | Code-First 短路 |
|------|---------|----------------|
| Code 失败 + Model 未运行 | 全运行，浪费 Model API 成本 | 短路，节省成本 |
| 快速反馈 | 等全部完成 | Code 失败即返回 |
| 完整诊断 | 有全部结果 | 短路时 Model 结果为 skipped |

## Transcript 和 Outcome 数据结构

```yaml
transcript:
  task_id: "cat_portrait_001"
  trial_id: "trial_3"

  # === Transcript 部分：执行轨迹 ===
  tool_calls:
    - tool: "load_model"
      args: { model: "sdxl_base" }
      duration_ms: 1200
      error: null
    - tool: "generate"
      args: { prompt: "...", steps: 30 }
      duration_ms: 8500
      result: { tokens_used: 1200 }

  reasoning_steps:
    - "Analyzing prompt for style keywords"
    - "Selected photorealistic pipeline"

  total_duration_ms: 12500

  environment:
    model_version: "sdxl-1.0"
    gpu_memory_used: "8.2GB"
    random_seed: 42

  # === Outcome 部分：最终结果 ===
  outcome:
    image: <PIL.Image>              # 生成的图像对象
    blocked: false                  # 是否被安全过滤器拦截
    blocked_reason: null            # 拦截原因
    output_data:
      image_path: "/outputs/result_001.png"
      image_hash: "sha256:abc123..."
      resolution: [1024, 1024]
      generation_params: { ... }
```

**关键区别**：Transcript 记录完整的执行过程（工具调用链、时间线、推理步骤），Outcome 只记录最终状态（生成物、是否被拦截）。评分器根据自身 Scope 选择性访问这两部分数据。

### JSONL 事件流导出

> 设计灵感来源：[OpenAI Codex CLI](https://developers.openai.com/blog/eval-skills) 的 `codex exec --json` 输出格式

Transcript 支持导出为 JSONL（JSON Lines）格式的事件流，便于用脚本进行确定性检查和分析。每行一个 JSON 事件，按时间顺序排列。

**CLI 使用：**

```bash
# 运行测试并保存 JSONL 轨迹
compass test scenario.yaml --trace-dir ./traces --trace-format jsonl

# 生成的文件：traces/<case_id>.jsonl
```

**Python SDK 使用：**

```python
from compass.core.transcript import Transcript

# 导出为 JSONL 字符串
jsonl_content = transcript.to_jsonl()

# 保存到文件
transcript.save_jsonl("trace.jsonl")

# 从 JSONL 加载
loaded = Transcript.load_jsonl("trace.jsonl")
```

**事件类型：**

| 事件类型 | 说明 | 关键字段 |
|----------|------|----------|
| `transcript.started` | 执行开始 | `task_id`, `trial_id`, `input` |
| `tool_call.started` | 工具调用开始 | `call_id`, `tool_name`, `input` |
| `tool_call.completed` | 工具调用完成 | `call_id`, `status`, `duration_ms`, `cost`, `tokens` |
| `reasoning.step` | 推理步骤 | `index`, `step` |
| `outcome.set` | 结果设置 | `blocked`, `has_image`, `artifacts` |
| `transcript.completed` | 执行完成 | `final_passed`, `total_cost_usd`, `tool_call_count` |

**JSONL 输出示例：**

```jsonl
{"type":"transcript.started","ts":1706234567.0,"task_id":"cat_on_sofa","trial_id":"trial_001","input":{"prompt":"生成一只猫"}}
{"type":"tool_call.started","ts":1706234567.1,"call_id":"a1b2c3","tool_name":"comfyui.generate","tool_type":"image","input":{"prompt":"..."}}
{"type":"tool_call.completed","ts":1706234572.1,"call_id":"a1b2c3","status":"ok","duration_ms":5000,"cost":{"total_usd":0.02},"tokens":{"input_tokens":100}}
{"type":"outcome.set","ts":1706234572.2,"blocked":false,"has_image":true,"image_path":"/outputs/result.png"}
{"type":"transcript.completed","ts":1706234572.3,"final_passed":true,"tool_call_count":1,"total_cost_usd":0.02,"total_tokens":150}
```

**用 jq 进行快速分析：**

```bash
# 统计工具调用次数
cat trace.jsonl | jq -s '[.[] | select(.type == "tool_call.completed")] | length'

# 计算总成本
cat trace.jsonl | jq -s '[.[] | select(.type == "tool_call.completed") | .cost.total_usd // 0] | add'

# 找出所有失败的调用
cat trace.jsonl | jq 'select(.type == "tool_call.completed" and .status == "error")'

# 检查是否执行了特定命令
cat trace.jsonl | jq 'select(.type == "tool_call.started" and (.input.command? // "" | contains("npm install")))'

# 检测可能的无限循环（同一工具调用超过 10 次）
cat trace.jsonl | jq -s '
  [.[] | select(.type == "tool_call.completed") | .tool_name]
  | group_by(.)
  | map({tool: .[0], count: length})
  | .[] | select(.count > 10)'
```

**用 Python 编写确定性检查：**

```python
import json
from pathlib import Path

def load_events(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]

def check_ran_npm_install(events: list[dict]) -> bool:
    """检查是否执行了 npm install"""
    return any(
        e["type"] == "tool_call.started"
        and "npm install" in e.get("input", {}).get("command", "")
        for e in events
    )

def check_no_errors(events: list[dict]) -> bool:
    """检查没有错误"""
    return not any(
        e["type"] == "tool_call.completed" and e["status"] == "error"
        for e in events
    )

def check_cost_budget(events: list[dict], max_usd: float) -> bool:
    """检查成本预算"""
    total = sum(
        e.get("cost", {}).get("total_usd", 0)
        for e in events
        if e["type"] == "tool_call.completed"
    )
    return total <= max_usd
```

JSONL 格式与现有的 JSON 格式互补：JSON 用于完整持久化和程序间交换，JSONL 用于事件流分析和脚本检查。

### 效率评估与循环检测

> 设计目标：检测 Agent 是否高效工作，避免无限循环和浪费行为

Compass 提供专门的效率评估 Graders，检测 Agent 执行中的低效模式：

**LoopDetectionGrader (`loop_detection`)**

检测四种低效模式：

| 检测类型 | 说明 | 示例 |
|----------|------|------|
| 单工具重复 | 同一工具调用次数过多 | `search` 被调用 15 次 |
| 精确调用重复 | 相同工具+相同输入重复执行（无进展） | `read_file("a.py")` 连续调用 5 次 |
| 序列模式循环 | 重复的工具调用序列 | `A→B→C→A→B→C→A→B→C` |
| 输出重复 | 产生相同输出（浪费工作） | 多次查询返回相同结果 |

**配置示例：**

```yaml
graders:
  - name: loop_detection
    config:
      max_single_tool_calls: 10    # 单工具最大调用次数
      max_exact_repetitions: 3     # 相同调用最大重复次数
      max_sequence_repeats: 3      # 序列模式最大重复次数
      min_pattern_length: 2        # 最小检测模式长度
      max_pattern_length: 5        # 最大检测模式长度
      check_output_repetition: true
      ignore_tools: ["log", "print"]  # 忽略的工具
```

**检测结果示例：**

```json
{
  "passed": false,
  "score": 0.65,
  "details": {
    "total_tool_calls": 25,
    "issues_found": 2,
    "issues": [
      {
        "type": "sequence_pattern_loop",
        "description": "Repeating sequence pattern detected",
        "details": [{"pattern": "fetch→parse→save", "repeats": 5}],
        "severity": "critical"
      },
      {
        "type": "exact_call_repetition",
        "description": "Identical call repeated without progress",
        "details": {"read_file:{\"path\": \"config.json\"}": 4},
        "severity": "high"
      }
    ]
  }
}
```

**其他效率 Graders：**

| Grader | 用途 | 关键配置 |
|--------|------|----------|
| `tool_usage` | 工具调用合规性、重试检测 | `max_tool_calls`, `max_retries` |
| `cost_budget` | 成本/Token 预算 | `max_cost_usd`, `max_tokens` |
| `latency_budget` | 延迟预算 | `max_total_ms`, `max_per_tool_ms` |
| `efficiency` | 综合效率评估（工具数 vs 产出） | `expected_tool_count` |

### 模型价格配置（可覆盖）

`LLMToolCallMixin` 会用内置价格表（`compass.adapters.MODEL_PRICING`，按 1M token 计价，含 input/output/cached-input）自动估算每次调用的成本。价格经常变动，因此**内置表只是默认值，用户可以覆盖**，优先级为 `外部文件 / 运行时注册 > 内置默认`。

**方式一：外部价格文件（适合 CLI / 团队共享）**

设置环境变量 `COMPASS_PRICING_FILE` 指向一个 JSON 或 YAML 文件，或放在默认位置 `~/.compass/pricing.{json,yaml,yml}`（`import compass` 时自动加载）：

```json
{
  "gpt-5":     { "input": 2.0, "output": 8.0, "cached": 0.2 },
  "my-model":  { "input": 1.5, "output": 6.0 }
}
```

```bash
export COMPASS_PRICING_FILE=/path/to/pricing.json
```

`cached` 可选（缓存读取价，通常约为 input 的 0.1×）。文件是「模型 → 价格」映射，同名覆盖内置默认、新名新增。

**方式二：SDK 运行时注册（适合程序化调用 / 测试）**

```python
from compass.adapters import register_pricing, load_pricing_file, reset_pricing

register_pricing("gpt-5", {"input": 2.0, "output": 8.0, "cached": 0.2})
register_pricing("my-model", (1.5, 6.0))          # (input, output[, cached]) 元组
load_pricing_file("pricing.yaml")                  # 批量从文件合并
reset_pricing()                                    # 恢复内置默认（丢弃所有覆盖）
```

> 模型名会被规范化（小写、`_`→`-`），带日期后缀的 ID 走**最长前缀匹配**（如 `claude-opus-4-8-20260101` → `claude-opus-4-8`）。未收录且未配置的模型，`calculate_cost` 返回 `None`（成本视为未知）。

### 风格约定校验

> 设计目标：检测 Agent 输出是否遵循预定义的格式、模板和命名约定

Compass 提供 `StyleConventionGrader` 评分器，从多个维度校验输出风格：

**StyleConventionGrader (`style_convention`)**

| 检查类型 | 说明 | 示例 |
|----------|------|------|
| 必需段落 | 输出必须包含指定的标题/段落 | `## Summary`, `## Recommendations` |
| 段落顺序 | 段落必须按指定顺序出现 | Summary → Details → Conclusion |
| 必需短语 | 输出必须包含特定短语 | `"Based on the analysis"` |
| 禁止短语 | 输出不能包含特定短语 | `"I don't know"`, `"I'm not sure"` |
| 正则模式 | 必须/不能匹配特定正则表达式 | `ERR-\d{5}` 错误码格式 |
| 长度约束 | 字符/单词/行数限制 | 50-500 词 |
| Markdown 格式 | 代码块、标题深度、列表 | 必须包含代码块 |
| 命名规范 | 代码中的函数/类/变量命名 | 函数用 snake_case |

**配置示例：**

```yaml
graders:
  - name: style_convention
    config:
      source: output_data               # 内容来源
      required_sections:                # 必需段落
        - "Summary"
        - "Details"
        - "Recommendations"
      section_order: true               # 段落必须按顺序
      required_phrases:                 # 必需短语
        - "Based on the analysis"
      forbidden_phrases:                # 禁止短语
        - "I don't know"
        - "I'm not sure"
      case_sensitive: false             # 短语匹配不区分大小写
      required_patterns:                # 必需正则模式
        - "ERR-\\d{5}"
      forbidden_patterns:               # 禁止正则模式
        - "\\b\\d{3}-\\d{2}-\\d{4}\\b"  # SSN 格式
      min_words: 50                     # 最少词数
      max_words: 500                    # 最多词数
      markdown_checks:
        require_code_blocks: true       # 必须包含代码块
        max_heading_level: 3            # 最多 3 级标题
        require_lists: true             # 必须包含列表
      naming_conventions:               # 代码命名规范
        functions: snake_case
        classes: PascalCase
```

**检测结果示例：**

```json
{
  "passed": false,
  "score": 0.75,
  "details": {
    "total_checks": 4,
    "checks_passed": ["required_phrases", "length", "markdown"],
    "violation_count": 2,
    "violations": [
      {
        "type": "missing_section",
        "message": "Missing required sections: ['Recommendations']",
        "severity": "high"
      },
      {
        "type": "forbidden_phrase",
        "message": "Contains forbidden phrases: [\"I'm not sure\"]",
        "severity": "high"
      }
    ],
    "content_stats": {
      "length": 1250,
      "words": 187,
      "lines": 23
    }
  }
}
```

**内容来源配置：**

`source` 参数指定从哪里获取待检查的内容：

| source 值 | 说明 |
|-----------|------|
| `output_data` (默认) | 从 outcome.output_data 获取 |
| `text` / `text_artifact` | 从 TextArtifact.content 获取 |
| `code` / `code_artifact` | 从 CodeArtifact 文件内容获取 |
| `metadata.<key>` | 从 outcome.metadata 指定键获取 |
| `output_data.<key>` | 从 output_data 中指定键获取 |

### 结构化输出校验与部分合规评分

> 设计目标：校验 Agent 输出的结构化数据（JSON/SQL/YAML），支持部分合规评分而非二元判断

**JsonSchemaGrader (`json_schema`)**

支持三种评分模式：

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `strict` | 任何校验错误 = score 0（默认） | 严格合规要求 |
| `partial` | 按字段合规比例评分 | 允许部分正确 |
| `weighted` | 按字段权重评分 | 重要字段优先 |

**配置示例（严格模式）：**

```yaml
graders:
  - name: json_schema
    config:
      schema:
        type: object
        properties:
          name: { type: string }
          age: { type: integer, minimum: 0 }
        required: [name, age]
      scoring_mode: strict  # 任何错误 = 0 分
      strict: true          # 禁止额外字段
```

**配置示例（部分合规评分）：**

```yaml
graders:
  - name: json_schema
    config:
      schema:
        type: object
        properties:
          name: { type: string }
          email: { type: string, format: email }
          age: { type: integer }
          bio: { type: string }
        required: [name, email]
      scoring_mode: partial     # 按字段合规率评分
      pass_threshold: 0.7       # 70% 合规即通过
      field_weights:            # 自定义字段权重
        name: 2.0               # name 更重要
        email: 1.5              # email 重要
      required_weight: 2.0      # 必需字段权重乘数
      count_nested: true        # 计算嵌套字段
```

**检测结果示例：**

```json
{
  "passed": true,
  "score": 0.83,
  "details": {
    "scoring_mode": "partial",
    "pass_threshold": 0.7,
    "field_compliance": {
      "total": 4,
      "passed": 3,
      "failed": 1,
      "score_breakdown": {
        "total_weight": 6.5,
        "earned_weight": 5.5
      }
    },
    "error_summary": {
      "missing": [],
      "type_error": [{"path": "age", "message": "'not a number' is not of type 'integer'"}],
      "constraint": [],
      "format": []
    },
    "field_results": [
      {"path": "name", "required": true, "passed": true, "weight": 4.0},
      {"path": "email", "required": true, "passed": true, "weight": 3.0},
      {"path": "age", "required": false, "passed": false, "weight": 1.0},
      {"path": "bio", "required": false, "passed": true, "weight": 1.0}
    ]
  },
  "reasoning": "Partial compliance: 3/4 fields valid (score: 0.83, threshold: 0.7)"
}
```

**错误分类：**

| 错误类型 | 说明 | 示例 |
|----------|------|------|
| `missing` | 缺少必需字段 | `'email' is a required property` |
| `type_error` | 类型不匹配 | `'abc' is not of type 'integer'` |
| `constraint` | 约束违规 | `-5 is less than the minimum of 0` |
| `format` | 格式错误 | `'invalid' is not a 'email'` |
| `additional` | 额外字段（strict 模式） | `Additional properties not allowed` |

**其他结构化校验 Graders：**

| Grader | 说明 | 关键配置 |
|--------|------|----------|
| `sql_syntax` | SQL 语法校验 + 安全检查 | `allowed_statements`, `forbidden_keywords` |
| `structure_check` | 通用格式校验（JSON/YAML/XML/TOML） | `format`, `required_keys` |

### Rubric 评审与结构化输出

**RubricGrader (`rubric`)** 使用 LLM 进行多维度 Rubric 评审，通过结构化输出确保评估结果一致可解析。

**设计动机：**
- 传统 LLM 评审输出自由文本，需要脆弱的后处理解析
- 结构化输出（OpenAI JSON Schema / Claude Tool Use）强制保证一致格式
- 多维度评分支持细粒度诊断和权重自定义

**评分模式对比：**

| 维度 | 自由文本输出 | 结构化输出 |
|------|-------------|-----------|
| 输出一致性 | 依赖 prompt 工程 | Schema 强制保证 |
| 解析可靠性 | 正则/启发式解析 | 直接 JSON 反序列化 |
| 多维度评分 | 手动拆分 | 原生支持 criteria_scores |
| 校准难度 | 高（格式不一致） | 低（结构统一） |

**配置示例：**

```yaml
graders:
  - name: rubric
    config:
      criteria:
        - name: accuracy
          description: "事实准确性：回答是否正确无误"
          weight: 2.0
          score_guidance:
            "0": "完全错误"
            "0.5": "部分正确"
            "1": "完全正确"
        - name: completeness
          description: "完整性：是否覆盖所有要点"
          weight: 1.5
        - name: clarity
          description: "清晰度：表达是否清晰易懂"
          weight: 1.0
      pass_threshold: 0.7
      model: gpt-4o
      provider: openai  # 或 anthropic
      source: output_data  # 或 text_artifact, metadata.<key>
```

**结果示例：**

```json
{
  "passed": true,
  "score": 0.85,
  "reasoning": "Overall good quality with minor completeness issues",
  "details": {
    "criteria_results": [
      {"name": "accuracy", "score": 0.9, "weight": 2.0, "weighted_score": 1.8, "reasoning": "Factually correct"},
      {"name": "completeness", "score": 0.7, "weight": 1.5, "weighted_score": 1.05, "reasoning": "Missing one point"},
      {"name": "clarity", "score": 0.9, "weight": 1.0, "weighted_score": 0.9, "reasoning": "Well structured"}
    ],
    "confidence": 0.95,
    "suggestions": ["Add coverage of edge cases"]
  }
}
```

**Provider 支持：**

| Provider | 结构化输出方式 | 模型 |
|----------|---------------|------|
| OpenAI | `response_format: json_schema` | gpt-4o, gpt-4o-mini |
| Anthropic | Tool Use (强制调用) | claude-sonnet-4-20250514 |

## 协议演进史

以下按版本记录 ToolCall 协议的演进（当前版本 2.0）。

**多 Agent / 多轮上下文：一等字段**（ToolCall Protocol v1.2）

`agent_name` / `turn_index` 已从 `metadata` 提升为 `ToolCall` 的一等字段——因为「哪个 Agent、第几轮发起的调用」是多 Agent 场景下的核心分析维度，不该埋在自由扩展字段里。三个集成都会填充它们（OpenAI/OTLP 走父链回溯，pi 按 assistant 消息计轮次）。向后兼容：`ToolCall.from_dict` 在顶层缺失时会回退读取旧的 `metadata` 位置，老的 transcript 仍能正确加载。

**State Delta：环境状态变更是一等评估面**（ToolCall Protocol v1.3）

Agent 靠**改变环境**完成任务，最终输出和状态变更可能不一致——日历 agent 说"约好了"，但 state delta 里是错误时区或重复邀请；coding agent 交了能过测试的 patch，但顺手删了不相关的文件。"Hidden Technical Debt" 一文的判断是：*"如果你的 eval 不捕获 state delta，它就不足以评估有状态的任务。"* 因此协议 1.3 给 `ToolCall` 增加了 `state_delta: list[StateChange]` 槽位：

```python
from compass.core import StateChange

transcript.add_tool_call(
    tool_name="sandbox.bash",
    input={"command": "rm /etc/nginx/nginx.conf"},
    state_delta=[
        StateChange(kind="file", op="delete", target="/etc/nginx/nginx.conf",
                    before="sha256:ab12...", after=None),
    ],
)
```

`StateChange` 是领域无关的：`kind`（file/db/env/git/http/browser/memory/process，开放词汇）、`op`（create/update/delete/move/execute）、`target`（路径/表/键/ref）、`before`/`after`（哈希或指针）、`metadata`（自由扩展）。配套的 `state_delta` 过程守卫 grader：

```yaml
graders:
  - name: state_delta
    config:
      readonly: true                    # 只读 agent：任何状态变更即违规
  # 或细粒度规则：
  - name: state_delta
    config:
      forbid:
        - { op: delete, target: "/etc/*" }   # kind/op 精确匹配，target 是 glob
      require:
        - { kind: db, op: update }           # 任务预期的变更必须被记录到
      max_changes: 10
```

**边界纪律（控制面/数据面）**：Compass 定义槽位、词汇和守卫——**捕获** delta（快照、diff、覆盖文件系统）是数据面的活，由 adapter / harness / importer 填充。所以空 `state_delta` 意味着"没记录"而非"没变更"；`require` 规则因此兼作捕获检查——预期的变更没被记录也会失败。违规会带上肇事调用的 `call_id`，这正是 outcome 级检查给不了的**失败归因**（哪一步搞坏的）。向后兼容：1.3 之前的 transcript 加载后 `state_delta` 为空列表。

**第一个内置捕获：Claude 轨迹的文件编辑**。上面这条纪律有个副作用——槽位定义好了却没人填，`state_delta` grader 在 Claude 轨迹上就是**空过**（`readonly: true` 永远通过）。`compass.integrations.claude_agent` 现在填它：成功的 `Write`/`Edit`/`MultiEdit`/`NotebookEdit` → `StateChange(kind="file", …)`，路径从工具入参里直接读，是**精确值不是推断**。三条边界写在 [docs/integrations.md](integrations.md)：只记成功的调用、target 相对 session cwd（否则 worktree 的随机路径没法写 glob）、**Bash 造成的变更不记**（可靠解析 shell 不是这层该假装能做的事，错的 delta 比缺的 delta 更糟）。

这让"改了不该改的东西"变成确定性检查。典型的抓获场景：agent 改不动实现，转头把测试改成通过——`integration_test` 一片绿，`state_delta` 的 `forbid: [{kind: file, target: "tests/*"}]` 把它抓出来。**过程评估在这里不是锦上添花，是唯一发现得了的路径。**

**审计溯源：run_id / config_hash / grader_version**（ToolCall Protocol v1.4）

《Hidden Technical Debt》给出的**最小 trace 记录**要求包含 run id、prompt/config hash、verifier 版本——*"少于这些，就难以 replay、比较、审计。"* 否则就会掉进它描述的 cargo cult 评估：仪表盘的数字变好了，但没人能回答"这是同一份配置吗？判分器换过没有？"——数字上涨可能只是因为**测量本身变了**。协议 1.4 把这三样落成一等字段，runner 自动盖章，用户零配置：

```jsonc
// trace 文件（compass test --trace-dir 产出）
{
  "task_id": "case_1",
  "run_id": "9f3c2a1b04de",     // 本次 run 的唯一标识（与 checkpoint 共用同一 id）
  "config_hash": "5b1e8c...",   // scenario 指纹（agent 配置 + grader 配置 + 判分参数）
  "grading": {
    "results": [
      { "name": "exit_code_check", "score": 1.0, "grader_version": "1.0", ... }
    ]
  }
}
```

三个字段各答一个审计问题：

| 字段 | 位置 | 回答的问题 |
|---|---|---|
| `run_id` | `Transcript` / `EvalResult` | 这条 trace 是**哪次运行**产出的？（同 run 的所有 case/trial 共享一个 id，跨文件可关联） |
| `config_hash` | `Transcript` / `EvalResult` | 两次运行**可比吗**？hash 不同 = 配置变了，分数差异不能归因于 agent。复用 checkpoint 的 `scenario_fingerprint`——同一个 hash 同时守护 resume 和审计 |
| `grader_version` | 每条评分结果 | 分数差异是 **agent 变了还是判分器变了**？自定义 grader 改判分逻辑（阈值、judge prompt、rubric）时 bump `version` 类属性，历史分数即刻标记为不可直接比较 |

```python
@register_grader("my_grader")
class MyGrader(CodeGrader):
    version = "2.0"  # 判分逻辑变更时 bump：2.0 的分数不能和 1.x 直接对比

    async def grade(self, context: GradeContext) -> GradeResult:
        ...
```

**边界纪律**：这里记录的是**评测控制面自己的**溯源（哪次 run、哪份配置、哪个版本的判分器）——Compass 能自证的部分自动盖章；被测 agent 侧的 model/harness 版本属于数据面事实，槽位在 `Environment.model_version` / `adapter_version`，由数据面填充（pi / Claude 两个 importer 已填 `model_version`，内置 adapter 尚未填——这是已知缺口，已列入后续规划）。向后兼容：1.4 之前的 transcript 加载后两字段为空字符串（"未记录"），导入的外部 trace 同理；JSONL 事件流只在字段非空时写入。

**trace 写侧瘦身**（ToolCall Protocol v2.0）

`ToolCall.to_dict()` 不再输出 legacy 别名键 `tool` / `args` / `result`——它们完整复制 `tool_name` / `input` / `output`，工具输出大时 trace 文件体积近乎翻倍。这是纯写侧变更，读侧兼容不变：`from_dict()` 仍接受老键（老 trace 文件照常加载），Python 层的 `tc.tool` / `tc.args` / `tc.result` 属性别名照常可用。若有外部脚本直接读 trace JSON 里的这三个键，请改读 `tool_name` / `input` / `output`。
