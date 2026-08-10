# Compass 速查表

> 一页读完就能写出正确的 scenario 和自定义 grader。
> 展开细节见 `compass docs <topic>`：core-design / graders / scenario-config / analysis / integrations。

## 心智模型（先读这 5 条）

1. **Transcript（怎么做的）与 Outcome（做出了什么）分开建模。** grader 用 `grader_scope` 声明自己要哪份数据：`transcript` / `outcome` / `both`。
2. **执行与评分是两个动词。** `compass test --trace-dir` 落盘不可变轨迹，`compass grade` 事后评分——改判分器不用重跑 Agent。
3. **未打分 ≠ 0 分。** `score=None` 表示「没测出来」：不进加权平均的分母，但**会让 case 判负**（没跑完的评估无权签发通过）。
4. **Harness 失败 ≠ 模型失败。** adapter 崩溃/超时的 case 记 ERROR，**移出 pass rate 分母**。
5. **一个 case 的 graders 是流水线**：按声明顺序执行、共享 workspace，`required` 失败即中止后续。

## Scenario YAML 全字段

```yaml
name: "场景名"                    # 必填
description: ""
category: ""                      # 所有 case 的默认分类
tags: []                          # 所有 case 的默认标签
leak_markers: []                  # 答案泄漏标记，与 case 级合并

agent:                            # 必填
  adapter: image                  # 内置：image / coding / environment；其余用 @register_adapter
  endpoint: ""                    # 便捷字段，会并入 config
  workflow: ""                    # 同上
  config: {}                      # 传给 adapter；`compass test -m` 覆盖这里的 model 键

defaults:
  trials: 1                       # 每个 case 跑几次（>1 触发轮转采样 + pass^k）
  timeout: 300
  environment: {isolation: true, clean_cache: true, timeout: 300}

default_graders: []               # 应用到所有 case，排在 case 自己的 grader 之前
default_aggregation:              # case 未显式设置的字段从这里继承（逐字段合并）
  method: weighted_sum
  pass_threshold: 0.7
  required_graders: []            # 按名字指定必过的 grader
  short_circuit: disabled         # disabled | code_fail | code_pass | required_fail

sweep:                            # 参数扫描（笛卡尔积展开成多个 case）
  params: {steps: [20, 50]}

cases:
  - id: "case_id"                 # 必填，唯一
    description: ""
    input:
      prompt: "给 agent 的输入"    # 必填
      negative_prompt: ""
      params: {}                  # 任意结构，透传给 adapter
      reference_images: {}        # name -> 路径，grader 通过 context.get_reference_image(name) 取

    expect: pass                  # pass | fail（fail = 负向测试，见下）
    expect_reason: ""
    trials: null                  # 覆盖 defaults.trials
    stage: ""                     # 供 --stage 过滤
    category: ""                  # 覆盖场景默认
    tags: []                      # 静态分类标签（≠ grader 产出的观察标签）
    metadata: {}                  # 透传给 GradeContext.metadata
    leak_markers: []

    expected: {}                  # 简化写法，自动展开成 grader（见下）
    graders: []                   # 显式 grader 列表
    aggregation: {}               # 同 default_aggregation
    metrics:
      pass_at_k: [1, 3]           # 报告哪些 k
      consistency: true
```

### grader 条目

```yaml
graders:
  - name: tool_usage              # 必填，注册名
    type: code                    # code | model | human（写错不影响执行，只影响短路分组）
    weight: 1.0                   # 加权平均里的权重
    required: false               # 必过；**失败即中止后续 grader**
    gate: false                   # 硬闸门：必过，但不计入分数
    creates: []                   # 承诺写进 workspace 的文件名（str 或 list），没产出即判失败
    config: {}                    # 传给 grader
```

### `expected:` 简化写法

不想手写 grader 时用，会自动展开：

```yaml
expected:
  contains: ["必须出现"]           # → style_convention
  not_contains: ["不该出现"]
  matches: "\\d{4}-\\d{2}"        # 正则，str 或 list
  not_matches: []
  min_length / max_length: 0      # 字符数
  min_words / max_words: 0
  equals: "精确匹配"               # → exact_match
  equals_json: {}                 # → exact_match（答案先按 JSON 解析）
  json_schema: {}                 # → json_schema
  json_schema_strict: true
  similar_to: "语义参考文本"        # → semantic_match
  similarity_threshold: 0.7
```

> `expected:` 下的**未知键会直接报错**，不会被静默忽略——拼错 `contian:` 却报告「通过」是最糟的一类评测 bug。需要上面覆盖不了的断言时，写自定义 grader 或用 `external_checker`。

## 内置 grader 速查（41 个）

**scope 决定它能不能从有损轨迹重评**：`transcript` 域在 JSONL 上完整；`outcome` / `both` 域需要 `--trace-format json`。

| scope | grader |
|---|---|
| **transcript**（过程，跨 Agent 通用） | `tool_usage` `cost_budget` `latency_budget` `loop_detection` `turn_count` `state_delta` `leak_detection` `trajectory_judge`(model) |
| **both** | `efficiency` `reasoning_trace` `self_correction` `groundedness`(model) `external_checker` |
| **outcome · 通用** | `exact_match` `json_schema` `structure_check` `style_convention` `sql_syntax` |
| **outcome · 代码** | `exit_code_check` `test_runner` `integration_test` `lint` `type_check` `security_scan` `diff_accuracy` `diff_size` |
| **outcome · 数据** | `data_correctness` `query_quality` `sql_equivalence` |
| **outcome · 图像** | `image_assertions` `technical_quality` `edit_locality` `edit_preservation` |
| **outcome · LLM 判官** | `semantic_match` `rubric` `vlm_judge` `safety_check` `aesthetic_score` `edit_correctness` |
| **human** | `human_review` `pairwise_comparison` |

`compass list` 看当前注册表。

## 常用配方

**过程守卫（任何 Agent 都能用，不需要领域知识）**

```yaml
graders:
  - {name: cost_budget,   type: code, config: {max_cost_usd: 0.5}}
  - {name: latency_budget,type: code, config: {max_duration_ms: 30000}}
  - {name: loop_detection,type: code, config: {max_repeats: 5}}
  - {name: tool_usage,    type: code, gate: true, config: {required_tools: [search]}}
```

**安全闸门（评「执行」而非「文字」）**

```yaml
  - name: state_delta            # 环境状态变更守卫：建议写操作没问题，执行了才算违规
    type: code
    gate: true                   # gate：必过但不拉低分数
    config: {forbid: [write, delete]}
```

**负向测试（预期被拦截）**

```yaml
  - id: reject_nsfw
    expect: fail                 # agent 被 block 或 grader 判否 → 该 case 通过
    expect_reason: "安全过滤应拦截"
    input: {prompt: "..."}
```

**评分流水线（产物在 grader 间传递）**

```yaml
graders:
  - name: external_checker       # 任何可执行文件都能当 grader
    type: code
    required: true               # 抠不出来就中止，省下后面的昂贵调用
    creates: extracted.svg       # 承诺产出，没产出即判失败
    config: {command: ./checkers/extract-svg}
  - name: external_checker
    type: code
    config: {command: ./checkers/render, input: extracted.svg}
  - name: vlm_judge              # 只在前两步成功时才被调用
    type: model
    config: {criteria: ["构图是否合理"]}
```

**LLM 判官 + 受控词表标签**

```yaml
  - name: rubric
    type: model
    config:
      source: text_artifact      # output_data | text_artifact | metadata.<key>
      pass_threshold: 0.7
      criteria:
        - {name: accuracy, description: "答案是否准确", weight: 2.0}
      tags: [cited_a_document, hedged, invented_a_fact]   # 编译进 schema enum，跨 run 可聚合
```

## 自定义 grader 骨架

```python
from compass.graders import (CodeGrader, GradeContext, GradeResult,
                             GraderScope, register_grader)

@register_grader("my_check")
class MyCheck(CodeGrader):
    grader_scope = GraderScope.OUTCOME     # 声明数据需求
    version = "1.0"

    async def grade(self, context: GradeContext) -> GradeResult:
        ok = "关键词" in context.answer
        return GradeResult(
            name=self.name, grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=ok,
            score=1.0 if ok else 0.0,      # 测不出来时用 None，别用 0.0
            tags=["cited_a_doc"] if ok else [],   # 分类观察，通过也打
            metrics={"word_count": 42, "cited": ok},  # 定量观察：数值→mean±stderr，布尔→比率
            failure_tags=[] if ok else ["no_keyword"],
            details={"任意结构化诊断": 1},
        )
```

### `GradeContext` 能拿到什么

```python
context.prompt / .negative_prompt / .params / .metadata
context.answer                 # 最终文本答案：output_data["final_output"] → 落空取第一个 TextArtifact
context.image                  # PIL Image
context.is_blocked             # agent 是否被拦截
context.tool_calls             # list[ToolCall]：.tool_name .input .output .cost .tokens .state_delta
context.reasoning_steps        # list[str]
context.state_changes          # 所有工具调用的环境变更，扁平化
context.total_duration_ms
context.text_artifact / .code_artifact / .artifact(Type)
context.reference_answer       # golden 答案
context.get_reference_image(name)
context.workspace              # 共享评分工作目录（Path | None）
context.workspace_file(name)   # 可写路径（无 workspace 时抛异常）
context.read_workspace_file(name)  # 上游 grader 的产物，读不到返回 None
context.check_leaks()          # 答案泄漏扫描
```

## CLI

```bash
compass init s.yaml                          # 生成模板
compass test s.yaml                          # 跑
compass test s.yaml -c case1 -s smoke --category backend   # 过滤
compass test s.yaml -p -w 8                  # 并行
compass test s.yaml --trace-dir ./traces     # 落盘轨迹（后续可重评）
compass test s.yaml --trace-dir ./t --resume # 断点续跑（补齐差额，达标即 no-op）
compass test s.yaml -m gpt-5 -m claude-5     # 多模型排行榜（带 stderr + 显著性判定）

compass grade ./traces -s s.yaml             # 离线评分，不重跑 agent
compass grade ./traces -s s.yaml --regrade   # 改了判分器后重评
compass grade ./traces -s judge.yaml -n judge  # 第二套 grader，并存

compass analyze results.json                 # 分维度诊断 + 观察标签分布
compass compare a.json b.json                # 配对比较：翻转 + 95% CI + MDE
compass trace traces/case.json --steps       # 看轨迹
compass import session.jsonl                 # 导入 pi / OTLP / Claude / OpenAI Agents 轨迹
compass list                                 # 已注册的 grader 和 adapter
compass docs <topic>                         # 本文档体系
```

## 容易踩的语义坑

| 现象 | 原因 |
|---|---|
| `required` 的 grader 失败后，分数比预期低 | `required` 会**中止链路**，分数只由已跑过的 grader 决定，不是全部 grader 的加权平均 |
| grader 崩了，case 分数没被拉低但仍判负 | `score=None`（未打分）不进分母，但会让 case 判负 |
| `compass grade` 拒绝评某些 case | JSONL 轨迹没有 outcome 载荷，OUTCOME/BOTH 域 grader 无法重评——用 `--trace-format json` |
| pass rate 和 `passed/total` 对不上 | 分母是 `evaluated_cases = total − error`，harness 失败被剔除 |
| 自定义 grader 在 `details` 里写了 `skipped` 却没生效 | 控制标志是 `EvaluatorResult` 的一等字段，聚合不读用户的 `details` |
| 排行榜第 1 名其实不可信 | 看最后一行的配对检验——CI 跨 0 就是噪声 |
| 改了 grader 后 `compare` 报 `regraded` 警告 | 两侧判分契约指纹不同，差异不能归因于 agent |
| `expect: fail` 的 case「通过」了 | 负向测试：agent 被拦截或 grader 判否即为通过 |
