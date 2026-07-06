# 结果分析、对比与报告

> Compass 专题文档 · 返回 [README](../README.md)


## 评估结果分析与可视化

Compass 提供内置的结果分析引擎，延续 Transcript / Outcome 分离思想，从**过程**和**结果**两个维度深度剖析评估数据。

### 分析能力总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                     EvalResultAnalyzer                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐    │
│  │   Summary    │    │  Scope 分维度分析  │    │ Scope 对比诊断  │    │
│  │  汇总统计    │    │ Transcript Graders│    │ Transcript vs   │    │
│  │  通过率/均分  │    │  Outcome Graders  │    │    Outcome      │    │
│  └──────────────┘    └──────────────────┘    └──────────────────┘    │
│                                                                      │
│  ┌──────────────────┐    ┌────────────────────────────────────────┐  │
│  │  失败模式识别    │    │           改进建议生成                  │  │
│  │  T: / O: / B:   │    │  瓶颈检测 · 波动预警 · Scope 差距诊断  │  │
│  │  组合聚合与排名  │    │  优先级排序 · 可操作建议                │  │
│  └──────────────────┘    └────────────────────────────────────────┘  │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│  输出 → ConsoleReporter (Rich 终端)  │  JSON 文件  │  HTML 报告     │
└─────────────────────────────────────────────────────────────────────┘
```

### Scope 分维度分析

分析器根据评分器的 `GraderScope`，自动将结果拆分为 Transcript 维度和 Outcome 维度，分别统计：

- **逐评分器统计**：通过率、平均分、标准差、通过/失败数
- **瓶颈标记**：通过率低于 70% 的评分器自动标记为 `BOTTLENECK`
- **Transcript Graders** 表：展示 tool_usage、cost_budget、latency_budget 等过程评分器的表现
- **Outcome Graders** 表：展示 semantic_match、aesthetic_score、safety_check 等结果评分器的表现

### Transcript vs Outcome 对比诊断

分析器计算两个维度的平均分差距，自动给出诊断结论：

| 差距 | 诊断 | 含义 |
|------|------|------|
| \|gap\| < 0.1 | `balanced` | 过程和结果均衡，Agent 表现稳定 |
| T > O + 0.2 | `process_good_result_bad` | 过程正确但结果不好，可能最后一步执行有问题 |
| O > T + 0.2 | `result_good_process_bad` | 结果好但过程不规范，需优化效率或合规性 |

这种对比是 Transcript / Outcome 分离设计带来的独特分析能力——只有将两个维度独立评分，才能发现"过程与结果不匹配"的深层问题。

### 失败模式识别

自动聚合所有失败用例中的评分器失败组合，按频率排名：

```
失败模式示例：
#1  O:semantic_match, O:aesthetic_score    — 33.3%  (语义和美学同时不合格)
#2  T:cost_budget, T:latency_budget       — 20.0%  (成本和延迟同时超标)
#3  B:efficiency, O:image_assertions      — 13.3%  (效率低且图像基础检查不通过)
```

前缀 `T:` / `O:` / `B:` 标识失败发生在 Transcript、Outcome 还是 Both 维度，帮助快速定位问题根源。

### 改进建议

基于分析数据自动生成可操作建议：

- **瓶颈评分器**：指出 Transcript 或 Outcome 中通过率最低的评分器
- **Scope 差距**：当 Transcript 和 Outcome 分数存在显著差距时给出解读
- **分数波动**：当标准差过大时提示 Agent 行为不稳定
- **优先级区分**：Outcome 类问题标注为"核心功能问题，需优先解决"

### CLI 使用

```bash
# 分析评估结果（终端可视化输出）
compass analyze results/eval_results.json

# 分析结果目录下的所有 JSON 文件
compass analyze results/

# 同时导出分析报告为 JSON
compass analyze results/ --output analysis_report.json
```

### Python SDK 使用

```python
from compass.report import EvalResultAnalyzer, TaskEvalResult, ConsoleReporter
from compass.graders.base import GradeResult, GraderScope, GraderType

# 构造评估结果
results = [
    TaskEvalResult(
        task_id="cat_on_sofa",
        passed=True,
        grade_results=[
            GradeResult(
                name="semantic_match",
                grader_type=GraderType.MODEL,
                grader_scope=GraderScope.OUTCOME,
                passed=True, score=0.85,
            ),
            GradeResult(
                name="tool_usage",
                grader_type=GraderType.CODE,
                grader_scope=GraderScope.TRANSCRIPT,
                passed=True, score=0.9,
            ),
        ],
        duration_ms=9500,
    ),
    # ... 更多任务结果
]

# 运行分析
analyzer = EvalResultAnalyzer(results)
report = analyzer.analyze()

# 终端可视化输出
ConsoleReporter().render(report)

# 获取结构化数据
print(report.summary)                # 汇总统计
print(report.transcript_analysis)    # Transcript 维度分析
print(report.outcome_analysis)       # Outcome 维度分析
print(report.scope_comparison)       # Scope 对比诊断
print(report.failure_patterns)       # 失败模式
print(report.recommendations)        # 改进建议

# 序列化为 JSON
import json
json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
```

### 结果 JSON 格式

`compass analyze` 命令接受以下 JSON 格式的评估结果文件：

```json
[
  {
    "task_id": "cat_on_sofa",
    "passed": true,
    "duration_ms": 9500,
    "grade_results": [
      {
        "name": "semantic_match",
        "grader_type": "model",
        "grader_scope": "outcome",
        "passed": true,
        "score": 0.85
      },
      {
        "name": "tool_usage",
        "grader_type": "code",
        "grader_scope": "transcript",
        "passed": true,
        "score": 0.9
      }
    ]
  }
]
```

每条 `grade_results` 中的 `grader_scope` 字段（`outcome` / `transcript` / `both`）决定该评分结果归入哪个分析维度。

### 配对比较（`compass compare`）：把对比当测量，不当读数

`analyze` 看一次运行，`compare` 回答控制面最常见的问题：**改了一个变量（换模型/改 prompt/加工具）之后，B 比 A 真的好了吗？** 平均分涨 2.4 个点可能是真提升、也可能纯是噪声——均值本身分不出来。

```bash
compass compare results_a/ results_b.json        # 输入格式与 analyze 相同（文件或目录）
compass compare a.json b.json --json             # 机器可读输出
compass compare a.json b.json -o cmp.json        # 保存报告
```

按 case id 配对后输出三层证据：

1. **配对差值 + 95% 置信区间**：pass rate 与平均分的 B−A 差值都带 CI；CI 跨 0 就是"在噪声带内"，同时报告当前样本量下的**最小可检测效应（MDE）**——把"没有显著差异"和"样本太少测不出来"区分开
2. **Case 翻转清单**：`pass → fail`（回归）与 `fail → pass`（改进）逐个列出——平均分不动不代表没有翻转，一个回归 + 一个改进在均值上完全抵消
3. **Verdict**：显著提升 / 显著回归 / 噪声带内（pass rate 为主轴，分数 CI 补充"没翻转但分数系统性变好"的情形）

```
│ Pass rate  │ 83.3% │ 87.5% │  +4.2% │ [-10.2%, +18.5%] │   ← 看似涨了，CI 跨 0
│ Mean score │ 0.614 │ 0.634 │ +0.021 │ [+0.002, +0.039] │   ← 小但显著
...
Verdict: pass rate within noise band, but significant score improvement
```

细节：只在一侧出现的 case 会**明确列出并排除出统计**（覆盖范围变了要可见，不静默丢弃）；多 trial 的 case 用 `passed_trials/total_trials` 作为该 case 的通过分数，比布尔更细。Python 侧 `from compass.report import compare_paths, compare_results, paired_stats` 可编程使用。

## Best-of-k 评分展示

受 Stripe Agent Benchmark **每个任务跑 3 次，取最高分**的做法启发，Compass 在多次 Trial 场景下同时展示 **average score** 和 **best-of-k score**，帮助区分"模型能力上限"与"稳定输出水平"。

### 核心概念

| 指标 | 计算方式 | 含义 |
|------|----------|------|
| **Average Score** | `mean(scores)` | 模型稳定输出水平 |
| **Best-of-k Score** | `max(scores[:k])` | 模型能力上限（取前 k 次最高分） |
| **Best vs Avg Gap** | `best - avg` | 差距越大说明输出不稳定，有优化空间 |

### 数据流

```
TrialMetrics.best_of_k(k)          # 单 Case 的 best-of-k
  → CaseResult.best_score           # 优先取 best_of_k，回退到 score_max
    → EvalResult.best_of_k_score    # 所有 Case 的 best_score 平均值
      → Analyzer summary            # avg_best_score / best_vs_avg_gap
```

### 报告展示

- **CLI Summary**：多 Trial 时自动增加 Best-of-k 列
- **Console Reporter**：Summary 面板显示 Avg Score (mean) / Avg Score (best-of-k) / Best vs Avg Gap
- **HTML Reporter**：每个 Case 卡片显示 Avg / Best / Trials 信息

### 使用示例

```bash
# 每个任务跑 3 次，报告自动展示 best-of-k
compass test scenarios/my_test.yaml --trials 3

# 分析结果中包含 best-of-k 数据
compass analyze results/ --output json
```

输出示例：
```
┌─────────────────────────────┐
│        Summary              │
├─────────────────────────────┤
│ Avg Score (mean)      0.450 │
│ Avg Score (best-of-k) 0.700 │
│ Best vs Avg Gap      +0.250 │
└─────────────────────────────┘
```
