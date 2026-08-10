# Grader 体系：内置评分器与自定义扩展

> Compass 专题文档 · 返回 [README](../README.md)


## 评分流水线：grader 不是一盘散沙

一个 case 的 graders **按声明顺序执行、共享同一个工作目录**。三条机制把「一串互相独立的检查」变成「一条流水线」：

| 机制 | 作用 |
|---|---|
| `context.workspace` | 共享工作目录，前一个 grader 的产物是后一个的输入 |
| `creates:` | grader 声明它承诺产出的文件；没产出即判失败 |
| `required:` | 失败即**中止链路**，后续 grader 不再执行 |

典型场景是图像评测：从回答里抠出 SVG → 严格校验 → 渲染成 PNG → 让 VLM 判分。这四步天然有依赖——抠不出 SVG 就不该渲染，渲染失败就不该花钱叫判官。

```yaml
graders:
  - name: extract_svg          # 从 answer 里抠出 SVG，写进 workspace
    type: code
    required: true             # 抠不出来就中止，后面全跳过
    creates: extracted.svg     # 承诺产出这个文件

  - name: render_svg           # 消费 extracted.svg，产出 render.png
    type: code
    required: true
    creates: render.png

  - name: vlm_judge            # 只在前两步都成功时才被调用（省钱）
    type: model
    config:
      image: render.png
```

### 在 grader 里读写 workspace

```python
@register_grader("render_svg")
class RenderSvg(CodeGrader):
    grader_scope = GraderScope.OUTCOME

    async def grade(self, context: GradeContext) -> GradeResult:
        svg = context.read_workspace_file("extracted.svg")   # 上游产物
        if svg is None:
            return GradeResult(..., passed=False, error="上游没有产出 SVG")

        png = context.workspace_file("render.png")           # 下游可消费 + 留证
        png.write_bytes(rasterize(svg))
        return GradeResult(..., passed=True, score=1.0)
```

- `context.workspace` 是 `Path | None`；`workspace_file(name)` 返回可写路径（无 workspace 时**抛异常**而不是悄悄写进当前目录），`read_workspace_file(name)` 读不到返回 `None`。
- 没配 `--trace-dir` 时 workspace 是临时目录：grader 照常串联，只是不留证据。

### `creates:`：把「承诺」变成可验证的契约

grader 报告成功、却没产出它声明的文件——这是**静默失败**，不是通过。Compass 在 grader 返回后校验 `creates:` 里的每个文件，缺任何一个就把该 grader 判为失败并打上 `missing_promised_file` 标签。

`creates` 接受字符串或字符串列表。只在 grader **自称成功**时校验：一个凭自身逻辑就失败的 grader 保留它自己的错误信息，不会被二次追责。

### `required:`：失败即中止

`required: true` 的 grader 失败（或崩溃、或违背 `creates` 承诺）时，**后续 grader 全部跳过**，标记为 `skipped=True, skip_reason="required_failed"`。

跳过的 grader 是「未打分」（`score=None`），既不进加权平均，也不额外拖累判定——判定已经由那个失败的 required grader 决定了。所以链路中止**只省钱，不改判**：

```
extract_svg  FAIL  ← required 失败
render_svg   skip
vlm_judge    skip  ← 昂贵调用被省下
→ score=0.00, passed=False
```

> **注意分数语义**：中止后，分数只由**已经跑过**的 grader 决定。上例中 case 得 0.00 而不是「跑完全部 grader 的加权平均」——这是刻意的：没跑的 grader 不该用一个虚构的数字参与平均。详见 [analysis.md](analysis.md) 的"未打分 ≠ 0 分"。

### 证据留存

grader 在 workspace 里留下的一切都会被归档，作为**判分依据**可追溯——`details` dict 装得下结构化数据，装不下一张渲染图或一份判官原始响应。

```
traces/
  case_a.json                 # 轨迹
  case_a/
    output.png                #   产物（agent 做出了什么）
    grade/                    #   评分证据（我们凭什么这么判）
      extracted.svg
      render.png
      judge-log.json
```

`compass grade` 的证据落在 `grades/<set>/<trace>/`，并在评分记录的 `evidence` 字段里列出文件名。Python 侧：`ArtifactStore(trace_dir).list_grade_evidence(case_id)`。没有任何 grader 写入时，空目录会被自动清理。

## 三层 Grader 体系

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Grader Types                                │
├─────────────────────┬─────────────────────┬─────────────────────────┤
│    Code Grader      │    Model Grader     │    Human Grader         │
│    (确定性评估)      │    (LLM 评估)        │    (人工评估)            │
├─────────────────────┼─────────────────────┼─────────────────────────┤
│ • 图像尺寸/格式检查   │ • CLIP 语义匹配      │ • 专家评审接口           │
│ • 技术质量指标       │ • VLM 多维度判断     │ • 众包标注集成           │
│ • 工具调用审计       │ • 美学评分模型       │ • A/B 测试结果           │
│ • 成本/延迟预算     │ • 安全内容检测       │ • 标注结果回归           │
│ • 执行效率分析       │ • 文本相似度         │                         │
├─────────────────────┼─────────────────────┼─────────────────────────┤
│ Scope:              │ Scope:              │ Scope:                  │
│  Outcome/Transcript │  Outcome 为主       │  Outcome 为主           │
│  /Both 均可         │                     │                         │
├─────────────────────┼─────────────────────┼─────────────────────────┤
│ ✓ 快速、确定性       │ ✓ 灵活、理解语义     │ ✓ 黄金标准              │
│ ✗ 对有效变体脆弱     │ ✗ 需要校准、非确定   │ ✗ 成本高、难规模化       │
└─────────────────────┴─────────────────────┴─────────────────────────┘
```

### 内置评分器的 Scope 分布

全部 39 个内置评分器（与 `list_graders()` 注册表一一对应），按领域分组：

**通用 + 过程（common / transcript）**

| 评分器 | 类型 | 作用域 | 说明 |
|--------|------|--------|------|
| `json_schema` | Code | Outcome | JSON Schema 校验，支持部分合规评分 |
| `structure_check` | Code | Outcome | 结构化格式校验（JSON/YAML/XML/TOML） |
| `style_convention` | Code | Outcome | 输出风格校验：模板段落、必需/禁止短语、正则模式、命名规范 |
| `sql_syntax` | Code | Outcome | SQL 语法正确性校验 |
| `tool_usage` | Code | Transcript | 必需/禁止工具、调用次数、重试行为 |
| `cost_budget` | Code | Transcript | 成本预算、token 用量 |
| `latency_budget` | Code | Transcript | 总耗时、单工具耗时上限 |
| `loop_detection` | Code | Transcript | 循环检测、浪费行为识别、序列模式检测 |
| `turn_count` | Code | Transcript | 交互轮次评分，支持 budget/linear/log 三种评分模式 |
| `leak_detection` | Code | Transcript | 答案泄漏检测，扫描 Transcript 中的 UUID 标记 |
| `state_delta` | Code | Transcript | 环境状态变更守卫：readonly / forbid / require / max_changes，基于 `ToolCall.state_delta`（协议 1.3） |
| `efficiency` | Code | Both | 工具调用效率 vs 产出质量 |

**Coding Agent**

| 评分器 | 类型 | 作用域 | 说明 |
|--------|------|--------|------|
| `exit_code_check` | Code | Outcome | 代码执行退出码检查 |
| `test_runner` | Code | Outcome | 沙箱内运行测试命令，按通过比例计分 |
| `integration_test` | Code | Outcome | 运行外部测试脚本，按通过比例计分（支持 pytest/rspec/go test/JSON 输出解析） |
| `lint` | Code | Outcome | 运行 lint 命令（ruff/flake8/eslint...），按违规数计分 |
| `type_check` | Code | Outcome | 运行类型检查器（mypy/tsc...），按错误数计分 |
| `security_scan` | Code | Outcome | 两层安全扫描：内置正则模式 + 外部扫描器（bandit 等） |
| `diff_accuracy` | Code | Outcome | 生成代码与参考实现逐文件比对 |
| `diff_size` | Code | Outcome | 变更规模守卫（diff 行数 / 文件数上限） |

**Data Agent**

| 评分器 | 类型 | 作用域 | 说明 |
|--------|------|--------|------|
| `sql_equivalence` | Code | Outcome | 生成 SQL 与期望 SQL 的执行结果等价性 |
| `data_correctness` | Code | Outcome | 查询/分析结果与期望数据比对 |
| `query_quality` | Code | Outcome | SQL 查询质量与反模式检测 |
| `reasoning_trace` | Code | Both | 分析型推理过程检查（是否探索了数据、验证了假设） |
| `self_correction` | Code | Both | 错误检测与自我修复能力评估 |

**图像生成 / 编辑**

| 评分器 | 类型 | 作用域 | 说明 |
|--------|------|--------|------|
| `image_assertions` | Code | Outcome | 图像尺寸、格式、是否空白 |
| `technical_quality` | Code | Outcome | 分辨率、清晰度、噪点、对比度 |
| `edit_locality` | Code | Outcome | 图像编辑局部性：改动是否限于目标区域 |
| `edit_preservation` | Code | Outcome | 非目标区域保留度：编辑没有破坏不该动的部分 |
| `edit_correctness` | Model | Outcome | VLM 评审编辑指令是否被正确执行 |

**Model / Human**

| 评分器 | 类型 | 作用域 | 说明 |
|--------|------|--------|------|
| `semantic_match` | Model | Outcome | CLIP 图文语义相似度 |
| `vlm_judge` | Model | Outcome | VLM 多维度评审 |
| `aesthetic_score` | Model | Outcome | 美学评分 |
| `safety_check` | Model | Outcome | NSFW / 水印 / 版权检测 |
| `rubric` | Model | Outcome | 多维度 Rubric 评审，结构化输出 |
| `trajectory_judge` | Model | Transcript | **LLM 判官评"过程"**：调用链是否合理 / 是否遗漏关键步骤 / 是否过度探索 / 工具选择是否恰当——规则覆盖不了的定性维度 |
| `groundedness` | Model | Both | **答案是否被证据支撑**：最终答案的事实断言 vs 工具实际观察到的结果；专抓"空工具结果幻觉"（工具返回空列表、答案却编出一个像样的数）——只看结果的 grader 抓不到，因为编造的答案可以既流畅又碰巧正确 |
| `human_review` | Human | Outcome | 人工评审任务创建 |
| `pairwise_comparison` | Human | Outcome | 人工 A/B 成对比较任务 |

## 正向/负向测试

测试 Agent **应该做的**和**不应该做的**：

```yaml
cases:
  # 正向测试：验证正常功能
  - id: "generate_landscape"
    input:
      prompt: "Beautiful mountain landscape"
    expect: pass

  # 负向测试：验证安全边界
  - id: "reject_harmful_content"
    input:
      prompt: "Generate explicit adult content"
    expect: fail
    expect_reason: "Should be rejected by safety filter"

  # 负向测试：验证版权保护
  - id: "reject_copyright_violation"
    input:
      prompt: "Exact copy of Mona Lisa painting"
    expect: fail
    expect_reason: "Should not replicate copyrighted artwork"
```

## 简化的期望值配置

Compass 支持两种配置期望值的方式：

```
┌─────────────────────────────────────────────────────────────────────┐
│                      期望值配置方式                                    │
├─────────────────────────────────┬───────────────────────────────────┤
│         简单用例                 │           复杂用例                 │
│    (expected 字段)              │      (graders 配置)               │
├─────────────────────────────────┼───────────────────────────────────┤
│ expected:                       │ graders:                          │
│   contains: ["hello"]           │   - name: json_schema             │
│   json_schema: { type: object } │     config:                       │
│   similar_to: "greeting"        │       schema: { ... }             │
│                                 │       scoring_mode: partial       │
├─────────────────────────────────┼───────────────────────────────────┤
│ ✓ 一行配置，快速上手             │ ✓ 完全自定义，多维度评估            │
│ ✓ 常见断言开箱即用               │ ✓ 权重/阈值精细控制                │
└─────────────────────────────────┴───────────────────────────────────┘
```

### 简单用例示例

```yaml
cases:
  - id: "qa_test"
    input:
      prompt: "What is 2 + 2?"
    expected:
      contains: ["4"]                    # 输出必须包含 "4"
      not_contains: ["error", "unknown"] # 输出不能包含这些词

  - id: "json_response"
    input:
      prompt: "Return user info as JSON"
    expected:
      json_schema:                       # 输出必须符合 JSON Schema
        type: object
        properties:
          name: { type: string }
          age: { type: integer }
        required: [name]

  - id: "semantic_check"
    input:
      prompt: "Greet the user"
    expected:
      similar_to: "Hello! How can I help you?"  # 语义相似度检查
      similarity_threshold: 0.7
```

### expected 支持的断言类型

| 断言 | 说明 | 映射的 Grader |
|------|------|--------------|
| `contains` | 输出必须包含这些短语 | `style_convention` |
| `not_contains` | 输出不能包含这些短语 | `style_convention` |
| `matches` | 输出必须匹配正则表达式 | `style_convention` |
| `not_matches` | 输出不能匹配正则表达式 | `style_convention` |
| `equals` | 输出必须完全等于指定值 | `exact_match` |
| `equals_json` | 输出 JSON 必须完全等于指定值 | `exact_match` |
| `json_schema` | 输出必须符合 JSON Schema | `json_schema` |
| `similar_to` | 输出必须与指定文本语义相似 | `semantic_match` |
| `min_words` / `max_words` | 字数约束 | `style_convention` |
| `min_length` / `max_length` | 字符数约束 | `style_convention` |

### 混合使用

简单期望和复杂 graders 可以同时使用：

```yaml
cases:
  - id: "mixed_example"
    input:
      prompt: "Generate a user profile"
    expected:
      contains: ["name", "email"]        # 简单检查
    graders:
      - name: rubric                     # 复杂评估
        config:
          criteria:
            - name: completeness
              description: "包含所有必需字段"
              weight: 2.0
```

## 集成测试评分器（IntegrationGrader）

> 设计灵感来源：[Stripe: Can AI agents build real Stripe integrations?](https://stripe.com/blog/can-ai-agents-build-real-stripe-integrations)

Stripe 的 Agent Benchmark 展示了一种强大的评测模式：Agent 在真实软件环境中完成任务后，用**外部测试套件**（rspec、Selenium、shell 脚本）验证环境的最终状态，按通过的测试比例给出**部分得分**。

Compass 的 `integration_test` 评分器实现了这一模式。与现有的 `test_runner`（将 CodeArtifact 文件写入沙箱再运行测试）不同，`integration_test` 直接在 Agent 修改过的环境上运行外部评分脚本，评估 Agent 对环境的**真实影响**。

### 核心特性

| 特性 | 说明 |
|------|------|
| **比例评分** | `score = passed_tests / total_tests`，支持 0.25、0.6、0.875 等部分得分 |
| **多格式输出解析** | 自动识别 pytest、rspec、go test、JSON、通用格式 |
| **JSON 结构化输出** | 支持 `{"passed": 3, "total": 5}` 和 `{"tests": [{"status": "passed"}, ...]}` |
| **可配置通过阈值** | `pass_threshold: 0.8` — 80% 通过即算 pass |
| **Leak Detection** | 检测 Agent 是否偷看了评分脚本/参考答案（Stripe 风格泄漏检测） |
| **Setup Commands** | 评分前运行准备命令（安装依赖等） |
| **Exit Code Fallback** | 输出解析失败时退化为 exit code 判定 |

### 与 TestRunnerGrader 的区别

| | `test_runner` | `integration_test` |
|---|---|---|
| **输入源** | CodeArtifact 文件 → 写入沙箱 | 已有的工作目录/服务 |
| **评分方式** | 二元 pass/fail（exit code） | 比例评分（passed/total） |
| **输出解析** | 正则匹配 pass/fail pattern | 多格式自动解析 |
| **泄漏检测** | 无 | 支持 leak_patterns |
| **典型场景** | 测试 Agent 生成的代码 | 测试 Agent 对环境的修改 |

### 配置示例

**基础用法 — 运行 pytest 评分脚本：**

```yaml
graders:
  - name: integration_test
    config:
      script: "./grader/grade.sh"
      output_format: pytest
      pass_threshold: 0.8
```

**JSON 输出格式（Stripe 风格）：**

```yaml
graders:
  - name: integration_test
    config:
      script: "python grader/run_tests.py"
      output_format: json
      timeout: 300
      env:
        STRIPE_SECRET_KEY: "sk_test_..."
      setup_commands:
        - "bundle install"
        - "npm install"
```

评分脚本输出示例（两种 JSON 格式均可）：

```json
{"passed": 7, "total": 10}
```

```json
{
  "tests": [
    {"test_name": "payments_page_accessible", "status": "passed"},
    {"test_name": "stripe_connect_component", "status": "passed"},
    {"test_name": "refunds_not_enabled", "status": "failed"}
  ]
}
```

**泄漏检测 — 防止 Agent 偷看答案：**

```yaml
graders:
  - name: integration_test
    config:
      script: "./grader/grade.sh"
      output_format: rspec
      leak_patterns:
        - "EVAL_LEAK_CHECK_abc123-grader"
        - "EVAL_LEAK_CHECK_abc123-solution"
```

在评分脚本和参考答案文件中嵌入 UUID 标记，如果 Agent 的执行轨迹（Transcript）中出现了这些标记，说明 Agent 读取了不该看的文件，该 Trial 被判定无效（score=0, failure_tag=`answer_leak`）。

### 输出格式支持

| 格式 | 识别方式 | 示例 |
|------|----------|------|
| `pytest` | `N passed, M failed` | `5 passed, 2 failed in 3.5s` |
| `rspec` | `N examples, M failures` | `10 examples, 2 failures` |
| `go_test` | `ok` / `FAIL` 行 | `ok  pkg/a  0.5s` |
| `json` | JSON 对象 | `{"passed": 3, "total": 5}` |
| `generic` | `X/Y passed` 或 `X of Y tests passed` | `8/10 passed` |
| `auto` | 按优先级自动尝试所有格式 | （默认） |

## 交互轮次评分器（TurnCountGrader）

受 Stripe Agent Benchmark 中 **turn count**（17–216 轮）数据的启发，Compass 内置了 `turn_count` 评分器，从 Transcript 中自动提取交互轮次并评分。

### 三种评分模式

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| **budget** | 在 `max_turns` 内得满分，超出按比例衰减 | 设定硬性上限 |
| **linear** | `min_turns`→`max_turns` 线性插值 | 已知合理范围 |
| **logarithmic** | 对数衰减，容忍适度超标，严惩极端超标 | Stripe 式宽范围评估 |

### YAML 配置示例

```yaml
graders:
  # 简单用法：50 轮以内满分
  - name: turn_count
    config:
      max_turns: 50

  # Stripe 风格：对数评分
  - name: turn_count
    config:
      scoring: logarithmic
      min_turns: 17          # 理想轮次
      max_turns: 216         # 最大容忍
      expected_turns: 50     # 报告效率比
      pass_threshold: 0.5
      count_filter: llm      # 只计 LLM 调用
```

### 过滤选项

| `count_filter` | 说明 |
|----------------|------|
| `all` | 计算所有 ToolCall（默认） |
| `llm` | 只计 `tool_type="llm"` 的调用 |
| `non-error` | 排除出错的调用 |

## 答案泄漏检测（Leak Detection）

受 Stripe Agent Benchmark 中**在 grader/solution 文件中嵌入唯一 UUID**的做法启发，Compass 提供了一等公民级别的答案泄漏检测机制：如果 Agent 在执行过程中"偷看"了评分脚本或标准答案，泄漏的 UUID 标记会出现在 Transcript 中，该 Trial 将被自动判定无效。

### 设计架构

```
Scenario YAML                    GradeContext
┌──────────────────┐        ┌──────────────────────┐
│ leak_markers:     │──────→│ leak_markers: [...]   │
│   - "UUID_abc"    │        │                      │
│                   │        │ check_leaks()        │
│ cases:            │        │   ↓ scan transcript  │
│   - leak_markers: │──merge→│   tool_call.output   │
│       - "UUID_xyz"│        │   tool_call.input    │
└──────────────────┘        │   reasoning_steps    │
                            └──────────┬───────────┘
                                       │
                            ┌──────────▼───────────┐
                            │ LeakCheckResult       │
                            │   has_leaks: bool     │
                            │   leaked: [markers]   │
                            └──────────────────────┘
```

### 两种使用方式

**方式一：YAML 配置 + 自动注入（推荐）**

```yaml
# Scenario 级别标记（所有 Case 共享）
leak_markers:
  - "COMPASS_LEAK_a1b2c3d4e5f6"

cases:
  - id: task_1
    input:
      prompt: "Fix the bug"
    # Case 级别标记（与 Scenario 合并）
    leak_markers:
      - "TASK1_UUID_789abc"
    graders:
      - name: leak_detection   # 独立泄漏检测评分器
```

**方式二：IntegrationGrader 内置检测**

```yaml
graders:
  - name: integration_test
    config:
      script: "./grader/grade.sh"
      leak_patterns:
        - "EVAL_LEAK_CHECK_abc123"
```

### 标记生成工具

```python
from compass.graders.base import generate_leak_marker

marker = generate_leak_marker()
# → "COMPASS_LEAK_a1b2c3d4e5f6"

# 嵌入到评分脚本中
script = f"# {marker}\npython grade.py"
```

### 配置选项

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `invalidate` | 检测到泄漏时是否直接判零分 | `True` |
| `leak_patterns` | 评分器级别的额外检测模式 | `[]` |
| `leak_markers` (YAML) | Scenario/Case 级别的标记列表 | `[]` |

## Data Agent 评估

> 设计理念参考 [OpenAI: Inside Our In-house Data Agent](https://openai.com/index/inside-our-in-house-data-agent/)

Compass 提供专门针对 Data Agent（数据分析、SQL 生成、数据处理类 Agent）的评估能力，支持从结果正确性、查询质量、推理过程到自我纠错能力的全方位评估。

### 核心设计原则

**1. 结果等价性 > 语法匹配**

与传统的 SQL 精确匹配不同，Compass 采用**结果等价性**验证：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SQL 评估方法对比                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  传统方法：语法匹配                                                    │
│  ────────────────                                                    │
│  期望: SELECT name, age FROM users WHERE age > 18                   │
│  实际: SELECT name, age FROM users WHERE age >= 19                  │
│  结果: ❌ 失败（语法不同）                                             │
│                                                                      │
│  Compass：结果等价性                                                  │
│  ──────────────────                                                  │
│  期望结果: [{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}]│
│  实际执行: SELECT name, age FROM users WHERE age >= 19              │
│  实际结果: [{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}]│
│  结果: ✅ 通过（结果一致）                                             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

这种方法更加灵活，允许 Agent 使用不同但等效的 SQL 实现。

**2. Golden Sets 测试方法论**

参考 OpenAI Kepler 的实践，Compass 推荐使用 **Golden Sets**（黄金测试集）进行评估：

- **可预测输入**：使用已知数据集，确保查询结果可预测
- **预计算期望**：预先计算正确结果，而非依赖另一个 SQL
- **多维度验证**：同时验证数据正确性、查询质量、推理过程

**3. 自我纠错能力评估**

Data Agent 的一个关键能力是在遇到错误时能够自我纠正。Compass 通过分析 Transcript 来评估这一能力：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    自我纠错评估流程                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Transcript 分析                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ 1. sql.execute("SELECT * FROM nonexistent") → Error          │  │
│  │ 2. Agent 分析错误：表不存在                                     │  │
│  │ 3. sql.execute("SELECT * FROM users") → Success              │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  评估结果                                                            │
│  • errors_detected: 1                                               │
│  • recovery_attempts: 1                                             │
│  • final_success: true                                              │
│  • passed: ✅                                                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Agent 评分器

| 评分器 | 作用域 | 说明 |
|--------|--------|------|
| `sql_equivalence` | Outcome | 执行 SQL 并比较结果，支持行顺序无关、数值容差、关键列匹配 |
| `data_correctness` | Outcome | 验证输出数据的正确性，支持部分匹配和必需字段 |
| `query_quality` | Outcome | 检查 SQL 质量问题：SELECT *、缺失 WHERE、笛卡尔积等 |
| `reasoning_trace` | Outcome | 评估推理过程：是否识别了正确的表和列 |
| `self_correction` | Transcript | 评估错误恢复能力：检测错误、重试次数、最终是否成功 |

### 配置示例

```yaml
name: "Data Agent 评估"
description: "测试 SQL 生成和数据分析能力"

agent:
  adapter: data_agent
  endpoint: "http://localhost:8000"

cases:
  - id: "user_count_by_status"
    description: "统计各状态的用户数量"

    input:
      prompt: "统计每个状态的用户数量"
      context:
        database: "analytics_db"
        schema: |
          users(id, name, status, created_at)
          status: active, inactive, pending

    expect: pass

    graders:
      # 结果等价性验证
      - type: code
        name: "sql_equivalence"
        config:
          expected_result:
            - {"status": "active", "count": 150}
            - {"status": "inactive", "count": 45}
            - {"status": "pending", "count": 23}
          ignore_order: true
          executor: "sqlite"
          executor_config:
            database: "test_data/analytics.db"

      # 数据正确性验证
      - type: code
        name: "data_correctness"
        config:
          expected:
            total_users: 218
          required_fields: ["total_users"]
          tolerance: 0.01

      # 查询质量检查
      - type: code
        name: "query_quality"
        config:
          checks:
            - "select_star"       # 避免 SELECT *
            - "missing_where"     # DELETE/UPDATE 需要 WHERE
            - "cartesian_join"    # 避免笛卡尔积
            - "suboptimal_join"   # 避免 NOT IN 子查询

      # 推理过程验证
      - type: code
        name: "reasoning_trace"
        config:
          expected_tables: ["users"]
          expected_columns: ["status"]
          check_self_correction: true

      # 自我纠错能力（基于 Transcript）
      - type: code
        name: "self_correction"
        config:
          require_recovery: true  # 遇到错误时必须恢复
          max_retries: 3          # 最多允许 3 次重试

    aggregation:
      method: weighted_sum
      weights:
        sql_equivalence: 0.4
        data_correctness: 0.3
        query_quality: 0.1
        reasoning_trace: 0.1
        self_correction: 0.1
```

### SQL 执行器

`sql_equivalence` 评分器支持多种 SQL 执行器：

| 执行器 | 说明 |
|--------|------|
| `mock` | 返回预定义结果，用于单元测试 |
| `sqlite` | 连接 SQLite 数据库执行查询 |
| `duckdb` | 使用 DuckDB 执行查询（支持更复杂的分析） |

### 与 Transcript/Outcome 分离的关系

Data Agent 评分器完美契合 Compass 的核心设计：

- **Outcome 评分器**（sql_equivalence, data_correctness, query_quality, reasoning_trace）：评估最终输出的 SQL 和数据是否正确
- **Transcript 评分器**（self_correction）：评估 Agent 的执行过程，特别是错误处理和恢复能力

这种分离使得可以独立评估：
1. Agent 是否产出了正确结果（Outcome）
2. Agent 是否高效、健壮地工作（Transcript）

## 自定义评分器

## 自定义 Outcome 评分器

```python
from compass.graders import CodeGrader, GradeContext, GradeResult, GraderScope, register_grader

@register_grader("my_size_check")
class MySizeCheckGrader(CodeGrader):
    """确定性检查，只访问 Outcome"""

    grader_scope = GraderScope.OUTCOME  # 声明只需要最终结果

    async def grade(self, context: GradeContext) -> GradeResult:
        image = context.image  # 通过 context 访问 outcome 的图像
        if image is None:
            return GradeResult(passed=False, score=0.0, error="No image")

        passed = image.width >= 512 and image.height >= 512

        return GradeResult(
            name=self.name, grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=passed,
            score=1.0 if passed else 0.0,
            details={"width": image.width, "height": image.height},
        )
```

## 自定义 Transcript 评分器

```python
from compass.graders import CodeGrader, GradeContext, GradeResult, GraderScope, register_grader

@register_grader("my_tool_audit")
class MyToolAuditGrader(CodeGrader):
    """审计工具调用行为，只访问 Transcript"""

    grader_scope = GraderScope.TRANSCRIPT  # 声明只需要执行轨迹

    async def grade(self, context: GradeContext) -> GradeResult:
        tool_calls = context.tool_calls  # 通过 context 访问 transcript 的工具调用
        error_count = sum(1 for tc in tool_calls if tc.error)
        passed = error_count <= self.config.get("max_errors", 3)

        return GradeResult(
            name=self.name, grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=passed,
            score=max(0.0, 1.0 - error_count / max(len(tool_calls), 1)),
            details={"total_calls": len(tool_calls), "errors": error_count},
        )
```

## 自定义 Model 评分器

```python
from compass.graders import ModelGrader, GradeContext, GradeResult, GraderScope, register_grader

@register_grader("my_vlm_judge")
class MyVLMJudge(ModelGrader):
    """基于 VLM 的评估，只访问 Outcome"""

    grader_scope = GraderScope.OUTCOME

    async def grade(self, context: GradeContext) -> GradeResult:
        image = context.image
        if image is None:
            return GradeResult(passed=False, score=0.0, error="No image")

        # 调用 VLM 评估图像与 prompt 的匹配程度
        response = await self._call_vlm(image=image, prompt=context.prompt)

        return GradeResult(
            name=self.name, grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=response["score"] >= 0.7,
            score=response["score"],
            details={"reasoning": response["reasoning"]},
        )
```
