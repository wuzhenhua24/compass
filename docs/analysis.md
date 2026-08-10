# 结果分析、对比与报告

> Compass 专题文档 · 返回 [README](../README.md)


## 离线评分（`compass grade`）：跑一次，评多次

**执行和评分是两个动词。** `compass test --trace-dir` 负责跑 agent 并把轨迹落盘，`compass grade` 负责给已落盘的轨迹打分：

```bash
compass test scenarios/qa.yaml --trace-dir ./traces    # 数据面：执行 + 记录
compass grade ./traces -s scenarios/qa.yaml            # 控制面：评分 + 重评
```

轨迹是**不可变的证据**，评分只往 `traces/grades/<name>/` 里**新增**文件，从不修改轨迹本身。这解开了三个原本被"跑评一体"锁死的场景：

| 场景 | 没有 `grade` 时 | 有了 `grade` 后 |
|---|---|---|
| 改一句 rubric / 挪一个阈值 | 必须重跑 agent，而 agent 非确定性让新旧分数**根本不可比**——`compass compare` 想测量的东西正好被这一步毁掉 | 同一批轨迹重评，分数差异**只可能**来自判分器 |
| 判官校准（LLM judge vs human） | 做不到：两套 grader 拿不到同一批样本 | 两个 grade set 评同一批轨迹，直接算一致性 |
| 评别人的 agent | `compass import` 只能打印摘要 | 导入轨迹 → 直接用 Compass 的 grader 评分，闭环 |

### 落盘布局

```
traces/
  cat_on_sofa.json              # 轨迹，永不修改
  cat_on_sofa/output.png        # 产物二进制（ArtifactStore 写入）
  grades/
    default/                    # 一个 grade set
      _spec.json                #   判分器规格快照（逐 case 指纹）
      cat_on_sofa.json          #   评分记录
    judge/                      # 另一个 grade set，与上面并存
      ...
```

每条评分记录同时带**两份溯源**：证据侧的 `run_id` / `config_hash`（从轨迹继承，回答"我评的是哪次运行"）和判定侧的 `spec_fingerprint`（回答"我用的是哪版判分器"）。

### 判分器指纹与 staleness

`spec_fingerprint` 是**逐 case** 计算的内容哈希——覆盖该 case 解析后的 grader 列表、aggregation 规则、leak markers 和 `expect`。它是自动的，不依赖手工维护 `Grader.version`：改了阈值就变，忘记 bump 版本号也拦不住它。

于是重复执行 `compass grade` 是幂等的，并且会主动报告过期：

```bash
compass grade ./traces -s qa.yaml     # 第二次：Graded 0, reused 2
# 编辑 qa.yaml 里某个 case 的阈值后再跑：
compass grade ./traces -s qa.yaml
# ⚠ 1 existing grade(s) came from an older version of this grader spec — use --regrade
compass grade ./traces -s qa.yaml --regrade   # 丢弃重评，不留陈旧记录
```

改动只影响**被改的那个 case**：同一次编辑里没动过的 case 依然是 up-to-date，不会被无谓地重评。

### 多套 grade set 并存

`-n/--name` 指定 grade set 名字（默认取 scenario 文件名）。廉价的确定性 grader 和昂贵的 LLM 判官可以并排评同一批轨迹：

```bash
compass grade ./traces -s cheap.yaml -n default    # 秒级、零成本
compass grade ./traces -s judge.yaml -n judge      # 慢、花钱，但只跑一次
```

### 常用选项

```bash
compass grade ./traces -s qa.yaml -c case_a -c case_b   # 只评指定 case
compass grade ./traces -s qa.yaml -v                    # 展开逐 grader 明细
compass grade ./traces -s qa.yaml -o regraded.json      # 输出喂给 analyze / compare
compass grade ./traces/one_case.json -s qa.yaml         # 也接受单个轨迹文件
```

`-o` 输出的就是标准 `EvalResult` 结构，因此**重评结果与实跑结果在下游完全同权**：

```bash
compass grade ./traces -s strict.yaml -n strict -o a.json
compass grade ./traces -s relaxed.yaml -n relaxed -o b.json
compass compare a.json b.json      # 同一批轨迹、两套判分器的配对比较
compass analyze b.json
```

### 不评什么，会明说

沉默等于"全都评了"，所以凡是没评的都会列出来：

- **轨迹的 `task_id` 在 scenario 里没有对应 case** → 列出 task_id
- **scenario 里的 case 没有任何轨迹** → 列出 case id
- **JSONL 轨迹 + OUTCOME/BOTH 域 grader** → **拒绝评分**并提示改用 `--trace-format json`

最后一条值得单独说：JSONL 是事件流，`outcome.set` 事件里没有 `output_data`、也没有 artifact 载荷，所以 OUTCOME 域的 grader 从 JSONL 重评必然看到空产物。Compass 不会因此给出一个 0 分——**"没测出来"和"确实是 0 分"是两回事**，这类 case 报为 ERROR 而不是 FAIL。纯 TRANSCRIPT 域的 grader（工具调用、成本、延迟、绕圈）在 JSONL 上是完整的，照常评。

### Python SDK

```python
from compass.core.regrade import grade_traces
from compass.core.scenario import Scenario

report = await grade_traces(
    Scenario.from_yaml("qa.yaml"), "./traces", grade_set="default"
)
print(report.graded, report.skipped, report.stale)
print(report.result.pass_rate)          # 标准 EvalResult
print(report.unmatched, report.missing_traces, report.unreadable)
```

> **实现纪律**：`compass test` 与 `compass grade` 共用同一条评分路径（`Compass.grade_transcript`）——重评一条轨迹得到的分数，就是实跑时面对同样证据会给出的分数，两者不会漂移。

## 评测卫生：什么算进数字，什么不算

评测框架最容易犯的错，是把**不知道**渲染成一个看起来很合理的数字。Compass 在聚合层守住四条区分。

### 1. Harness 失败 ≠ 模型失败

网络抖动、API 限流、adapter 崩溃说明不了模型任何事。这类 case 记为 `ERROR`，并**移出分母**：

```
pass_rate = passed_cases / evaluated_cases        # evaluated = total − error
```

否则一次限流就能把 pass rate 压下去，还和"模型确实做错了"混成同一个数——发布门禁会因为机房抖动而拦下一个好版本。`average_score` / `best_of_k_score` 同样排除 ERROR case。被排除的数量始终显式上报（`compass test` 的 Error 列 + 一行说明），沉默地剔除数据比不剔除更糟。

同一条规则在 trial 级生效：死于 harness 的试验是**缺失的样本**而非失败的尝试，不进 pass@k / pass^k 的分母（见 [scenario-config.md](scenario-config.md) 的"采样纪律"）。

### 2. 未打分 ≠ 0 分

`GradeResult.score = None` 表示**没测出来**（判官超时、依赖缺失），与 `score = 0.0`（测了，很差）是两件事：

- 未打分的 grader **不进加权平均的分母**——一次 LLM judge 超时不会把 0.9 拉成 0.45
- 但它**会让这个 case 判负**：一次没跑完的评估无权签发"通过"

```python
# 两个 grader：一个给 1.0，一个崩了
case.overall_score == 1.0      # 分数诚实：只报测到的
case.passed is False           # 判定诚实：没测全就不能算过
```

短路跳过（`skipped=True`）是另一回事：那是聚合层**主动决定**不跑，所以既不计分也不拖累判定。

内置 model grader 在**找不到输入**时同样记为未打分——"outcome 里没有图"不是一次美学测量，"判官超时"也不是一次质量测量。这类早退（无图 / 无 criteria / 无内容 / LLM 调用失败）全部 `score=None`，而不是伪装成 0.0 拉低平均分。

| | 计入分数 | 影响判定 |
|---|---|---|
| 正常打分 | ✅ | ✅ |
| 未打分（崩溃/超时） | ❌ | ❌ 判负 |
| 短路跳过 | ❌ | ✅ 放行 |

### 3. 核心字段不可被用户 grader 覆写

控制标志（`skipped` / `skip_reason`）是 `EvaluatorResult` 上的**一等字段**，聚合只读它们。用户 grader 返回的 `details` 落在 `metadata` 里，永远不参与控制流——一个自定义 grader 在 `details` 里写 `{"skipped": True}` 只是普通数据，不会把自己悄悄摘出计分。

### 4. 分数只在同一份判分契约下可比

每个 CaseResult 带 `grader_fingerprint`：该 case 的判分契约（解析后的 grader 列表 + aggregation + leak_markers + expect）的内容哈希。它是自动的，改了阈值就变，不依赖手工 bump `Grader.version`；实跑（`compass test`）与重评（`compass grade`）用的是同一个函数，所以两边的结果直接可比。

`compass compare` 因此能给出归因警告：

```
⚠ 3 case(s) were graded by a different grader spec in B than in A —
  their diff is not attributable to the agent
```

没有这条，"B 比 A 涨了 5 个点"里混着判分器改动和 agent 改动，而你分不出来。旧结果文件没有指纹时不会误报——**缺失不等于不匹配**。

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

### 观察标签分布（行为画像）

`analyze` 除了失败聚类，还会输出 **Observed Tags** 表——grader 在评分时打的中性观察标签，**通过的样本也计入**：

```
│ short_answer     │    4 │   67% │       25% │  ← 67% 的回答很短，而它们只有 25% 通过
│ cited_a_document │    3 │   50% │      100% │
│ hedged           │    2 │   33% │        0% │
```

`Pass Rate` 是可行动的那一列：**「打了这个标签的样本更容易挂」**通常就是结论本身，而失败计数给不出它。语义是 presence-only（未出现 = 未观察到，不等于"否"）。

产出方式与 LLM 判官的受控词表见 [graders.md](graders.md) 的"观察标签"。结构化数据在 `AnalysisReport.observed_tag_analysis`（`count` / `share` / `pass_rate` / `graders` / `task_ids`）。

### Grader 指标聚合

grader 通过 `GradeResult.metrics` 产出的定量观察会按类型分别聚合——**数值 mean ± stderr，布尔比率**：

```
┃ Metric       ┃        Value ┃       Range ┃ n ┃
│ cited_source │          75% │    3/4 true │ 4 │
│ precision    │ 0.150 ±0.041 │ 0.05 … 0.25 │ 4 │
```

`n` 是答案的一部分：只有真正上报该 metric 的 case 进入聚合，40 个里 3 个的均值不该被读成 40 个的均值。结构化数据在 `AnalysisReport.metrics_analysis`。

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

`analyze` 与 `compare` 共用同一个读取器（`compass.report.analyzer.iter_case_dicts`），因此 Compass 各命令写出的**任何一种**结果形状都能被两者消费：

| 形状 | 来源 |
|---|---|
| `{"results": [EvalResult, ...], "summary": {...}}` | `compass test --report json` |
| `{"case_results": [...]}`（EvalResult） | `compass grade -o` / `EvalResult.to_dict()` |
| `[case, ...]` 或单个 case dict | 手写 / 自定义 harness |

最朴素的那种（case 列表）长这样：

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

### 报告数据层（`compass.report.site`）：一次运行 → 一份文档

报告的**取数**和**渲染**是分开的。`collect_run()` 把一次运行的 `EvalResult` 变成一份纯 JSON 文档，HTML 报告只是这份文档的一个渲染器——没有任何 viewer 直接去碰结果对象。

```python
from compass.report import collect_run
from compass.report.html import HTMLReporter

doc = collect_run(results, name="nightly")   # 纯 dict，json.dumps 即可落盘
html = HTMLReporter().render_document(doc)   # 渲染器只读文档
```

文档形状：

```jsonc
{
  "schema": "compass.run/1",     // viewer 先读它，旧页面可以拒绝新文档而不是渲染错
  "generated": "2026-08-10T08:00:00+00:00",
  "run":       { "name": …, "total_cases": …, "evaluated_cases": …,
                 "pass_rate": …, "average_score": …, "best_of_k_score": …,
                 "contract": "…" },   // 这次运行的判分契约 id，跨运行可比性的依据
  "scenarios": [ { "name": …, "run_id": …, "config_hash": …, "cases": [ … ] } ],
  "categories":[ { "category": …, "passed": …, "failed": …, "errors": … } ],
  "scopes":    { "outcome": true, "transcript": true }
}
```

这份文档承诺三件事，任何消费方都可以依赖：

- **分数和比率一律是 0..1 的分数**，百分号是渲染层的事。
- **harness 错误不进任何分母**，与 `EvalResult.pass_rate` 一致（见上文"评测卫生"）。报告里的 Pass Rate 卡片在有 error 时会写明分母：`Pass Rate (2 evaluated)`。
- **跳过 / 未打分的 grader 不进平均**——"没测"不是"测了 0 分"。

Case 行是 `EvalResult.to_dict()` 的 case 记录的近亲，因此 `iter_case_dicts()` 及其下游（`analyze` / `compare`）能直接读。两点刻意的差异：去掉了与 `evaluator_results` 完全重复的 `grade_results` 别名（发布出去的文档不该为同一份数据付两次字节），并为每行补上所属 scenario 与两条 scope 轴（`outcome_score` / `transcript_score`）。

轴的**存在性**和轴的**数值**是分开记录的（`has_outcome` / `has_transcript`）：一次所有 outcome grader 都打 0 分的运行，仍然是有 outcome 轴的——而且恰恰是最值得画散点图的那种。

### 可分享的静态站（`compass site build`）

把上面那份文档发布成一个静态站——一条链接代替一堆附件，而且**可累积**。

```bash
compass site build results.json -o site/                       # slug 默认取文件名
compass site build results.json -o site/ --slug image-evals \
    --name "Image Evals" --trace-dir traces/
python -m http.server -d site/                                 # 浏览器不会从 file:// 取数据
```

```
site/
  index.html              viewer（单文件、零依赖、无需构建）
  index.json              清单：一次运行一条 + 趋势快照
  runs/<slug>/run.json    该次运行的文档
  runs/<slug>/traces/…    显式发布的轨迹与产物
```

**合并语义是全部的关键**：一次 build 只写自己那个 slug，`index.json` 里其它条目原样保留。所以多个仓库的 CI 可以 build 进**同一个目录**（一个共享的 gh-pages 分支、一个对象存储前缀），清单自己累积起来——没有服务、没有数据库，合并点就是一个可幂等重写的文件。

站点里能做的事，都是把 Compass 已有的数据变得可点开：总览页按 pass rate 排、带趋势线 → 某次运行（双轴散点、分类表、Pass Rate 分母写明）→ 某个 case（每个 grader 的 scope/分数/观察标签/metrics、k 次 trial 的分布、判分契约指纹）→ 那次 case 的轨迹文件。observed_tags、failure_tags、category 都是点击筛选，不是重跑。

**发布 = 公开，所以默认是收着的：**

- **grader 的 `metadata` 默认不发布**。它是 case 行里唯一的自由字段，装的是 grader 自己的 `details`——经常就是模型原文、prompt、评审理由。要发布得显式 `--include-details`，页面上也会写明这次 build 有没有带上。
- **轨迹靠 `--trace-dir` 显式点名才发布**，因为一条 transcript 带着完整的输入输出。命令行会明说发布了多少个文件、多大。
- **slug 会被规范化成单个路径段**，`../` 之类折成 `-`，不可能写到站点目录外面。

**趋势线只留摘要**：同一个 slug 重复 build 时，上一次的汇总数字（pass rate / 平均分 / 用例数 / 判分契约 id）进入 `history`，默认留 20 次（`--history`）。留的是画一条趋势线所需的数，不是归档——case 行和产物只有最新一次在站上。运行页据此画趋势图，并在判分契约变化处断开（见下文"CI 配方"）。

**跨 slug 不排名**。A 项目和 B 项目的 case 不同，把它们的分数放进一张榜是误导，所以总览页只并列展示。要比大小，用 `compass compare` 对同一批 case 做配对检验。

### 实时查看（`compass site serve`）

不 build、不落盘，每个请求都从磁盘现算——**跑到一半的运行也能看**。这也是"为什么不能直接双击 index.html"：浏览器不会从 `file://` 取 JSON。

```bash
compass site serve results.json                    # 一个结果文件
compass site serve results/ -p 8000                # 目录下每个 *.json 一个 run（不递归）
compass site serve results.json --trace-dir traces/
compass site serve site/                           # 已 build 好的站，静态伺服
```

页面看到 manifest 里的 `live: true` 就每 3 秒轮询一次；数字变了才重渲染，**并且保留你正在用的筛选条件和已展开的行**——否则一边看一边被刷掉就没法用。build 出来的站是快照，`live: false`，页面不会轮询。

服务端只在结果文件的 mtime 变了才重新解析（轮询要足够便宜），但**轨迹列表每次都重扫**：一条 trace 落盘时结果文件不一定跟着改，"要等别的东西变了才点得开"比多扫一次目录糟糕得多。结果文件正被改写到一半会返回 503 而不是断连——轮询期间撞上这个是常态。

**默认值随绑定地址走**：绑 localhost 是本地调试，grader 细节照常给；绑到别的地址就是发布，默认脱敏并提示"这台机器能被谁访问"，要带细节得显式 `--include-details`。反过来 `--redact` 也能在本地强制脱敏。

### CI 配方：让同一个 eval 集攒出趋势

最常见的用法不是跨仓库，是**同一个仓库、同一个 eval 集、看它随时间怎么变**。每天跑一次、build 到同一个 slug，趋势自己就攒出来了。

配方只有三步，顺序不能变：

```bash
set -e                                   # 拉不回来就不要 build（见下）

# 1. 先把已有的站拉回来 —— 整个目录，不是只拉 index.json
aws s3 sync s3://evals/site ./site

# 2. build 到固定的 slug
compass test qa.yaml --report json -o results.json --trace-dir ./traces
compass site build results.json -o site/ --slug agent-qa --history 90 --trace-dir ./traces

# 3. 写回去
aws s3 sync ./site s3://evals/site
```

换成 git 分支同理（`git clone --depth 1 --branch site-data` → build → commit → push），换成内网 docroot 更简单——直接 build 进挂载路径，第 1、3 步都不用。

**四个坑，每个都真的会踩：**

**① slug 必须固定。** 它是趋势线的身份。写成 `--slug agent-qa-$(date +%F)` 或带上 commit sha，结果是每天新建一条 run、每条都没有历史——总览页会变成一堆一次性条目。要区分环境或分支，把它做成 slug 的**固定前缀**（`agent-qa-main` / `agent-qa-staging`），不要放变量。

**② 拉不回来就不能 build。** 历史快照是 build 时从"上一条 entry"里取的。如果第 1 步静默失败（网络问题、凭证过期、bucket 名打错），第 2 步会当成首次 build，**整条趋势归零且不报错**。所以 `set -e`，或者显式判断 `site/index.json` 存在再继续。这是这套流程唯一会静默丢数据的地方。

**③ 只有最新一次的 case 行和轨迹在站上。** `--history` 留的是摘要（pass rate / 平均分 / 用例数 / 判分契约），不是归档；每次 build 会清空并重写 `runs/<slug>/`。要留完整历史，那是 artifact 仓库的事，不是站点的事。日跑 + `--history 90` ≈ 一个季度的趋势线，默认 20 约三周。

**④ 别让 PR 跑污染趋势。** 如果主干夜跑和 PR 验证都 build，用不同的 slug（`agent-qa` vs `agent-qa-pr`），否则一次 PR 的临时结果会挤掉一天的真实数据点。

### 趋势线只在同一份判分契约内连起来

运行页的趋势图会把**判分契约变了的那些点断开、置灰，并明说断在哪**：

```
3 earlier build(s) were graded under a different contract, so the line breaks there.
```

这是把"分数只在同一份判分契约下可比"（见上文"评测卫生"）这条规则延伸到时间轴上——趋势本来就是拉长了的对比。契约 id（`run.contract`）由各 case 的 `grader_fingerprint` 派生，没有指纹时回退到 scenario 的 `config_hash`；两者都没有就记为空，**报告成"未知"而不是"没变"**——后者会画出一条穿过断点的直线，比不画更糟。

所以改了 grader 阈值之后趋势线断开是**正确行为**，不是 bug。要回答"改判分器之后到底好了还是差了"，用 `compass grade --regrade` 把同一批轨迹按新契约重评，再 `compass compare`。

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

## 多模型排行榜（`compass test -m`）

一个 scenario 跑多个模型并排名：

```bash
compass test qa.yaml -m gpt-5 -m claude-sonnet-5 -m tiny-model
compass test qa.yaml -m a -m b --model-key endpoint     # 覆盖 agent.config 的别的键
compass test qa.yaml -m a -m b --trace-dir ./traces     # 轨迹落在 traces/<model>/
```

```
                         Leaderboard
┏━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━┓
┃ # ┃ Model      ┃ Score (mean ± stderr) ┃ Pass Rate ┃ Cases ┃
┡━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━┩
│ 1 │ big-model  │          0.925 ±0.032 │    100.0% │     4 │
│ 2 │ mid-model  │          0.875 ±0.032 │    100.0% │     4 │
│ 3 │ tiny-model │          0.175 ±0.085 │      0.0% │     4 │
└───┴────────────┴───────────────────────┴───────────┴───────┘
big-model beats mid-model by +0.050 (95% CI [+0.050, +0.050])
```

### 排行榜是读数，不是测量

排名表天生诱导人从噪声里读出一个赢家——20 个 case、两个模型差 2 个点，通常根本分不出来。所以每一行都带**标准误**，并且**最后一行明说 top 2 的差距是否经得起配对检验**：

```
big-model leads mid-model by +0.020, but the 95% CI [-0.041, +0.081] spans 0
— the ranking is within noise (detectable at this n: ~0.089)
```

这也是把它建在 `compass compare` **之上**而不是旁边的原因：**表格给出顺序，配对统计告诉你这个顺序有没有意义。**

### 其它行为

- **模型是 `agent.config` 上的一个轴**：每个变体是 scenario 的深拷贝，只改 `agent.config[--model-key]`（默认 `model`），case / grader / 阈值完全相同，所以各次运行可比。
- **每个模型独立的 trace 子目录** `traces/<model>/`：否则模型之间会互相覆盖轨迹，而且 agent config 进了 scenario 指纹，后一个模型会让前一个的 checkpoint 失效。
- **并列同名次**：分数显示相同的行共享名次（standard competition ranking）。
- **多试验时多一列 `pass^k`**——仅当所有 case 报告了同一个 k，否则省略（把 pass^3 和 pass^5 平均是个没有意义的数）。
- **harness 错误照常移出分母**（见上文"评测卫生"）。
- **覆盖差异会列出**：只有 top 2 之一跑过的 case 不参与配对检验，并显式报告。

### 与 compare 串起来

`-o` 除了写合并报告（含 `leaderboard` 字段），还会为**每个模型**单写一份结果文件——因为 `compass compare` 按 case_id 配对，需要把各次运行分开：

```bash
compass test qa.yaml -m a -m b -m c --report json -o out.json
# → out.json  out.a.json  out.b.json  out.c.json
compass compare out.b.json out.a.json      # 任意两个做严格配对检验
```

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
