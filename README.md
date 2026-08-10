# Compass - Agent QA Framework

Compass 是 **Agent 评测的基座（substrate）**：提供一套标准的执行轨迹模型、多来源轨迹接入、可复用的过程评分器与可靠性指标——领域相关的"答案对不对"由你用几十行自定义 grader 补齐。它之于 Agent 评测，就像 pytest 之于测试、OpenTelemetry 之于可观测：**框架给骨架和标准，业务判定你来写**。

> 设计理念参考 [Anthropic: Demystifying Evals for AI Agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) 与 [Hidden Technical Debt of AI Systems: Agent Evaluation Infrastructure](https://leehanchung.github.io/blogs/2026/06/13/hidden-technical-debt-agent-evaluation-infra/)

## 定位：是什么 / 不是什么

面对市面上千差万别的 Agent 场景，Compass **不追求"开箱即评一切"**——那不现实，定制化必然存在。它的目标是把**所有 Agent 都需要的那层基座做厚**，让**每个 Agent 特有的定制层尽量薄**。定制不是缺陷，是产品留给你的插槽；框架的活是让它变小。

**✅ 是什么**

- **一套标准**：Transcript（怎么做的）/ Outcome（做出了什么）+ ToolCall 协议，让评分器面向统一数据结构，跨 Agent 复用
- **轨迹接入**：把 OpenAI Agents SDK / pi / OTLP·OpenInference / Claude Agent SDK 的原生轨迹归一成 Transcript（见 [docs/integrations.md](docs/integrations.md)）
- **可复用的过程评分器**：规则式的 `cost_budget` / `latency_budget` / `loop_detection` / `tool_usage` / `state_delta`（环境状态变更守卫），以及两个 LLM 判官——`trajectory_judge`（过程侧：调用链是否合理/遗漏关键步骤/过度探索）和 `groundedness`（答案 vs 证据：最终答案是否被工具观察支撑，专抓"空工具结果幻觉"）；机器通用、criteria 由你配
- **执行与评分解耦**：轨迹是不可变证据，`compass grade` 给已落盘的轨迹打分——改判分器不用重跑 Agent，多套 grader 可并排评同一批样本；判分器指纹自动标记过期评分（见 [docs/analysis.md](docs/analysis.md)）
- **评测卫生**：harness 失败不算模型失败（移出 pass rate 分母）、未打分 ≠ 0 分（判官超时不会伪装成低分，但也不许签发"通过"）、多试验轮转采样 + 补齐语义（中断后样本均衡，pass^k 不被偏样本污染）
- **观察标签与指标**：grader 产出中性标签与定量指标（通过的样本也打）——标签聚合成占比 + 每标签通过率，指标按类型聚合（数值 mean±stderr、布尔比率）——「短回答占 67%、其中只有 25% 通过」这类行为画像，失败统计给不出；LLM 判官支持受控词表（编译进 schema enum，跨 run 可聚合）
- **子进程 Checker**：`external_checker` 让**任何可执行文件**成为 grader（shell / Go 二进制 / `npm test`），契约是进程边界（env 进、stdout JSON 出、exit code 判定），checker 无需 import Compass——扩展面从「会写 Python」扩到「会写脚本」
- **评分流水线**：grader 按声明顺序共享 workspace，产物可在 grader 间传递（抠 SVG → 渲染 → VLM 判分）；`creates:` 让"承诺产出"可验证，`required:` 失败即中止链路省下昂贵调用；中间产物作为判分证据落盘（见 [docs/graders.md](docs/graders.md)）
- **多模型排行榜**：`compass test -m a -m b` 一次跑多个模型并排名——但排名是**读数不是测量**，每行带标准误，并明说 top 2 的差距是否经得起配对检验（建在 `compass compare` 之上）
- **可靠性指标与工程底座**：pass@k / pass^k（无偏估计）、聚合、报告、checkpoint 续跑、并行执行、`compass compare` 配对比较（case 翻转 + 置信区间，涨分是真提升还是噪声）、审计溯源（trace 自带 run_id / config_hash / grader_version，两次运行可比性可验证）
- **领域 recipe（可选）**：如 [`examples/ops_qa/`](examples/ops_qa/)（文档问答 bot 评测），是"如何自己写定制层"的模板

**🚫 不是什么**

- 不是"开箱评测任意 Agent"的银弹——**领域正确性判定必然要你写**（这是设计，不是缺陷）
- 不内置每个领域的正确性 grader（图像美学、代码功能、答案事实……天然定制）
- 不强求"驱动任意 Agent 跑"——自跑的 Agent 走**导入轨迹**更合适（见下）

### 控制面 / 数据面：这条边界的专业名字

[Hidden Technical Debt of AI Systems: Agent Evaluation Infrastructure](https://leehanchung.github.io/blogs/2026/06/13/hidden-technical-debt-agent-evaluation-infra/) 把 agent 评估基础设施拆成两层，恰好精确描述了 Compass 的边界：

- **控制面（control plane）**：决定"跑什么、结果是否该改变发布决策"——任务集与验证器选择、评分聚合、回归追踪、报告与发布门禁
- **数据面（data plane）**：真正跑 agent 并记录发生了什么——模型、harness、运行时、工具、记忆、环境状态、trace

**Compass = 控制面 + 连接两层的 trace schema；数据面归你。** 对应到模块：Scenario / Grader / 聚合 / pass^k / 报告 / checkpoint 续跑是控制面；Transcript + ToolCall 协议是那份 trace schema（文章公式里的 τ）；而 agent 怎么跑、沙箱、世界状态属于数据面——Compass 只通过 Adapter 驱动或 Import 消费轨迹与之接触，不试图拥有它（文章说 "agents need worlds"，但造世界是运行时的活，不是评测框架的）。这也解释了上面的"不是什么"：**数据面千差万别、不可复用，控制面与 trace 标准才是可复用资产。**

### 核心 vs 领域：一条干净的分界线

哪些评分通用、哪些必然定制，由 **GraderScope**（Transcript/Outcome 分离）直接预测：

| | 域 | 例子 | 谁写 |
|---|---|------|------|
| **过程** | TRANSCRIPT | 成本 / 延迟 / 绕圈 / 工具使用 / 是否执行危险操作 | ✅ 框架内置，跨 Agent 复用 |
| **正确性** | OUTCOME | 答案对不对 / 图美不美 / 代码能不能跑 | ✍️ 你写领域 grader（通常几十行） |

> **纪律**：核心 schema 保持小。领域字段（如 ops_qa 里的 `expected_doc` / `key_facts`）放进 grader config 或用户 harness，**别塞进核心模型**——这是防止"定制爆炸"淹没框架的关键。（反例参照物：`reference_answer` 之所以能进 `GradeContext`，是因为它像 `reference_image` 一样是跨领域的"参考数据"通用概念，而非某个领域的字段。）

### 两种接入方式（按 Agent 能否被外部驱动来选）

- **驱动（Adapter）**：Compass 亲自跑 Agent——适合图像生成端点、代码沙箱这类可被外部调用的场景
- **消费轨迹（Import）**：Agent 自己跑，Compass 消费它产出的轨迹——适合多数现代 LLM Agent（见 [docs/integrations.md](docs/integrations.md)）

### 何时用 Compass / 何时直接 DIY

Compass 在这些场景最省事；否则一个几十行的 pytest 可能就够了——框架不假装自己永远划算：

| 用 Compass 划算 | DIY 就好 |
|-----------------|----------|
| 关心**过程**（成本/工具/安全/检索/绕圈），不只最终答案 | 只需"最终字符串 vs golden"比对 |
| 有**多个** Agent / 框架，想要一套评测词汇 | 一次性的单个 Agent |
| 要**可靠性指标**（pass^k）与趋势 | 一次性 check 就行 |
| 评一个**你不驱动、只拿到轨迹**的 Agent | 你完全掌控 Agent 的进出 |

## 核心概念

基于业界最佳实践，Compass 定义了以下核心概念：

| 概念 | 说明 |
|------|------|
| **Task** | 单个测试用例，包含输入、预期和评估标准 |
| **Trial** | 同一 Task 的单次执行尝试（支持多次试验以应对模型不确定性） |
| **Grader** | 评分器，分为 Code/Model/Human 三类，每类声明自己的数据作用域（Scope） |
| **Transcript** | 执行轨迹——Agent **怎么做的**：工具调用、推理步骤、时间线、成本 |
| **Outcome** | 最终结果——Agent **做出了什么**：生成的图像、输出数据、是否被拦截 |
| **GraderScope** | 评分器的数据需求声明：`OUTCOME`（只看结果）/ `TRANSCRIPT`（只看过程）/ `BOTH`（两者都看） |

## 架构概览

```
┌─────────────────────────────────────────────────────────────────────┐
│                        1. Interface Layer                           │
│   compass CLI    │    Python SDK    │    Web Dashboard (后期)       │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────────────┐
│                    2. Test Orchestration Layer                      │
│  ┌────────────────┐  ┌─────────────────┐  ┌──────────────────────┐  │
│  │ Scenario Engine│  │ Trial Manager   │  │ Parallel Executor    │  │
│  │ (YAML + DSL)   │  │ (Multi-attempt) │  │ (Asyncio + Worker)   │  │
│  └────────────────┘  └─────────────────┘  └──────────────────────┘  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────────────┐
│                       3. Core Engine Layer                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Grader System (三层评估)                    │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │   │
│  │  │ Code Grader │  │Model Grader │  │ Human Grader        │   │   │
│  │  │ (确定性检查) │  │(LLM-as-Judge)│  │ (人工标注/复核)      │   │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────┐  │
│  │ Transcript   │  │ Metrics      │  │ Asset Manager│  │ Report  │  │
│  │ Collector    │  │ (pass@k/^k)  │  │ (Image Store)│  │ Gen     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────┘  │
│  ┌──────────────┐  ┌──────────────┐                                 │
│  │ Sandbox Mgr  │  │ Environment  │                                 │
│  │ (隔离执行)    │  │ (状态管理)    │                                 │
│  └──────────────┘  └──────────────┘                                 │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────────────┐
│                      4. Agent Adapter Layer                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ ┌────────┐  │
│  │ ComfyUI  │  │ SD WebUI │  │ Midjourney│  │ DALL-E  │ │ Custom │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘ └────────┘  │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐                       │
│  │ Coding   │  │  LLM     │  │ Environment  │                       │
│  └──────────┘  └──────────┘  └──────────────┘                       │
└─────────────────────────────────────────────────────────────────────┘
```

> 第 4 层有**两种接入**：上图的 **Adapter**（Compass 亲自驱动 Agent，适合图像/代码沙箱），
> 以及 **Integrations**（消费 Agent 自产的轨迹，适合自跑的 LLM Agent——见 [docs/integrations.md](docs/integrations.md)）。
> 二者产出的都是同一个 Transcript，下游评分/指标完全共用。
>
> 图中 ComfyUI / SD WebUI / Midjourney / DALL-E 为可对接的目标示意；当前内置注册的 adapter 是
> `image` / `coding` / `environment`，其余通过 `@register_adapter` 自定义接入。

### CLI 命令总览

| 命令 | 用途 |
|------|------|
| `compass test <scenario.yaml \| 目录>` | 运行测试场景（`--parallel` / `--trace-dir` / `--resume` / `--stage` / `--category`）；`-m` 可一次跑多个模型并出排行榜 |
| `compass grade <traces> -s <scenario.yaml>` | **离线评分**：给已落盘的轨迹打分，不重跑 Agent（`-n` grade set / `--regrade`） |
| `compass analyze <results>` | 分析评估结果：Scope 分维度、失败模式、改进建议 |
| `compass compare <a.json> <b.json>` | 配对比较两次运行：case 翻转 + 置信区间 + MDE |
| `compass site build <results>` | 把结果发布成可分享的静态站；多个仓库可 build 进同一个目录，索引自动累积 |
| `compass site serve <results \| site>` | 本地实时查看：每个请求从磁盘现算，跑到一半的运行也能看 |
| `compass trace <trace 文件>` | 查看执行轨迹（JSON/JSONL，`--steps` 展开工具调用） |
| `compass import <trace 文件>` | 导入外部轨迹（pi / OTLP·OpenInference / Claude stream-json，自动识别） |
| `compass eval <image>` | 单张图像快速评估（不写 scenario） |
| `compass init [output.yaml]` | 生成场景模板 |
| `compass docs [topic]` | 在终端里打印 Compass 自身文档（raw markdown，可管道）；不带参数列出主题 |
| `compass list` | 列出已注册的 grader 和 adapter |
| `compass baseline set/list` | 回归基线管理：把某 case 的产物存为基线 |
| `compass checkpoint list` | 列出 trace 目录下的运行断点（配合 `compass test --resume`） |

## 安装

```bash
# 使用 uv 安装
uv pip install compass-qa

# 或从源码安装
git clone https://github.com/your-org/compass.git
cd compass
uv sync
```

## 快速开始

### 1. 初始化场景模板

```bash
compass init my_scenario.yaml
```

### 2. 运行测试

```bash
# 运行单个场景
compass test scenarios/my_test.yaml

# 并行运行
compass test scenarios/ --parallel --workers 4

# 生成报告
compass test scenarios/ --report html --output report.html

# 保存执行轨迹（JSONL 格式，推荐用于脚本分析）
compass test scenarios/my_test.yaml --trace-dir ./traces --trace-format jsonl

# 保存执行轨迹（JSON 格式，用于程序间交换）
compass test scenarios/my_test.yaml --trace-dir ./traces --trace-format json

# 完整示例：并行运行 + 生成报告 + 保存 JSONL 轨迹
compass test scenarios/ -p -w 8 --report html -o report.html --trace-dir ./traces --trace-format jsonl
```

**Trace 选项说明：**

| 选项 | 说明 |
|------|------|
| `--trace-dir PATH` | 保存执行轨迹的目录，每个 case 生成一个文件 |
| `--trace-format [json\|jsonl]` | 轨迹文件格式，默认 `json` |

- **JSONL 格式**：每行一个事件，便于用 `jq` 进行快速分析（见 [docs/core-design.md](docs/core-design.md) 的"JSONL 事件流导出"）
- **JSON 格式**：完整的 Transcript 结构，便于程序间交换和持久化

> 多次试验数在 YAML 里配置（`defaults.trials` 或 case 级 `trials:`），不是 CLI 选项。

### 3. 离线评分：跑一次，评多次

执行和评分是两个动词——轨迹是不可变证据，评分只往 `traces/grades/<name>/` 里新增：

```bash
# 改了 rubric / 挪了阈值后重评，不用重跑 Agent
compass grade ./traces -s scenarios/my_test.yaml --regrade

# 廉价 grader 与 LLM 判官并存，评同一批轨迹
compass grade ./traces -s cheap.yaml -n default
compass grade ./traces -s judge.yaml -n judge -o judge.json
```

重复执行是幂等的；判分器配置一改，对应 case 的评分自动被标记为过期（内容指纹，不靠手工 bump 版本号）。这也是让 `compass compare` 有意义的前提：**同一批轨迹重评，分数差异只可能来自判分器，而不是 Agent 的随机性。** 详见 [docs/analysis.md](docs/analysis.md)。

### 4. 分析评估结果

```bash
# 分析单个结果文件（终端可视化输出）
compass analyze results/eval_results.json

# 分析整个目录
compass analyze results/

# 导出分析报告为 JSON
compass analyze results/ --output analysis_report.json
```

输出包含：汇总统计、Transcript/Outcome 分维度分析、Scope 对比诊断、失败模式排名和改进建议。

```bash
# 配对比较两次运行（A = 基线，B = 候选）：case 翻转 + 置信区间
compass compare results_a/ results_b/
```

```bash
# 发布成可分享的静态站（同一目录可被多个仓库反复 build，索引累积）
compass site build results.json -o site/ --slug agent-qa

# 本地实时查看：不 build，每个请求从磁盘现算，跑到一半的运行也能看
compass site serve results.json
```

### 5. 查看执行轨迹

```bash
# 查看某个 case 的 transcript（工具调用、耗时、成本、评分）
compass trace traces/cat_on_sofa.json
compass trace traces/cat_on_sofa.jsonl --steps
```

### 6. Python SDK

```python
import asyncio

from compass import Compass, Scenario

async def main():
    scenario = Scenario.from_yaml("scenarios/my_test.yaml")

    runner = Compass()
    result = await runner.run(scenario, trace_dir="./traces")

    # 场景级指标 + 审计溯源
    print(f"Pass Rate: {result.pass_rate:.1%}")
    print(f"run_id={result.run_id}  config_hash={result.config_hash}")

    # 逐 case 查看（多次试验时 trial_metrics 含 pass@k / pass^k）
    for case in result.case_results:
        if not case.passed:
            print(f"Failed: {case.case_id}  score={case.overall_score:.2f}")
            if case.trial_metrics:
                print(f"  metrics: {case.trial_metrics}")

asyncio.run(main())
```

## 核心特性与文档导航

Compass 的能力全貌按主题拆分为专题文档，README 只保留骨架。这些文档也随包分发——`compass docs <topic>` 可直接在终端读，无需回到仓库：

| 主题 | 内容 | 文档 |
|------|------|------|
| **速查表** | 一页读完就能写出正确的 scenario 和 grader：YAML 全字段、41 个内置评分器、常用配方、易踩的语义坑 | [docs/cheatsheet.md](docs/cheatsheet.md) |
| **核心设计** | Transcript/Outcome 分离、GraderScope、ToolCall 协议（当前 2.0：多 Agent 字段、state_delta、run_id/config_hash 审计溯源）、JSONL 事件流、成本/token 聚合 | [docs/core-design.md](docs/core-design.md) |
| **Grader 体系** | 三层体系（Code/Model/Human）、全部 39 个内置评分器、expected 简化配置、正负向测试、泄漏检测、Data Agent 评分器、自定义 grader | [docs/graders.md](docs/graders.md) |
| **场景配置与指标** | 场景 YAML 完整参考、多次试验、pass@k / pass^k、分类聚合（category/tags） | [docs/scenario-config.md](docs/scenario-config.md) |
| **分析与报告** | `compass analyze` 分维度诊断、`compass compare` 配对比较（case 翻转 + 置信区间 + MDE）、best-of-k、HTML 报告 | [docs/analysis.md](docs/analysis.md) |
| **接入外部 Agent** | OpenAI Agents SDK / pi / OTLP·OpenInference / Claude Agent SDK 轨迹导入、Environment Adapter、自定义 Adapter、黑盒 Agent 的 ToolCall 获取 | [docs/integrations.md](docs/integrations.md) |

几个贯穿全部文档的设计要点：

- **Transcript / Outcome 分离**：执行过程（怎么做的）与最终产物（做出了什么）独立建模，grader 通过 `GraderScope` 声明数据需求——"做对了吗"和"做得高效/安全吗"可以独立回答。
- **三层 Grader**：Code（确定性，先跑、可短路）→ Model（LLM 判官，判非判不可的）→ Human（金标准）。领域正确性 grader 是你的代码，几十行插进来。
- **可靠性指标**：pass@k（探索）/ pass^k（可靠性）无偏估计；`compass compare` 把"涨没涨"变成带置信区间的测量。
- **审计溯源**：每条 trace 自带 run_id / config_hash / grader_version——两次运行是否可比、分数变化归因于 agent 还是判分器，可验证。

## 端到端示例：文档问答 Agent 评估（`examples/ops_qa/`）

一个把 Compass 各能力串起来的**可跑模板**——评估基于文档的 agentic-RAG 运维问答 bot（`ops-qa-bot` 形态：`Read`/`Grep` 检索 `docs/`、只读 `Bash`/SSH 诊断、写操作只提议）。核心是：**doc-grounded 问答不能只判"语义对不对"**，模板把它拆成四类样本，每类配一组合适的 grader：

| 样本类型 | 评什么 | grader |
|---|---|---|
| `answerable` | 语义正确 + 关键事实 + 检索命中 | `key_facts`(gate) · `retrieval_hit` · `rubric`* |
| `unanswerable` | 正确弃答、不编造 | `abstention`(gate) |
| `live` | 跑了只读诊断（无固定 golden 答案）| `tool_usage`(gate) |
| `forbidden_write` | 只提议不执行（**P0 安全**）| `no_write_ops`(gate) |

关键洞察：bot 的检索是 `Read`/`Grep` 工具调用、诊断是 `Bash` 工具调用，所以"读没读对文档""有没有越权写"都能从 `transcript.tool_calls` **确定性**评（agentic-RAG 相比黑盒 RAG 的评估红利）；安全评的是"**执行**"而非"文字"——答案里**建议** `CONFIG SET` 没问题、**执行**才算违规。

```bash
uv run python examples/ops_qa/eval.py    # 离线跑，无需真 bot / API key
```

配套：`GradeContext` 新增一等字段 `reference_answer`（golden 答案，对称于 `reference_image`）和 `.answer` 便捷属性（取 `outcome.output_data["final_output"]`）。详见 [`examples/ops_qa/README.md`](examples/ops_qa/README.md)。

## 自定义扩展（骨架）

领域正确性判定是你的代码——像写 pytest 用例一样写 grader，注册后在 YAML 里引用：

```python
from compass.graders import CodeGrader, GradeContext, GradeResult, GraderScope, register_grader

@register_grader("my_domain_check")
class MyDomainCheck(CodeGrader):
    grader_scope = GraderScope.OUTCOME   # 或 TRANSCRIPT / BOTH

    async def grade(self, context: GradeContext) -> GradeResult:
        ok = "expected keyword" in context.answer
        return GradeResult(
            name=self.name, grader_type=self.grader_type,
            grader_scope=self.grader_scope, passed=ok, score=1.0 if ok else 0.0,
        )
```

Transcript / Model grader、自定义 Adapter、黑盒 Agent 的 ToolCall 获取等完整指南：[docs/graders.md](docs/graders.md) · [docs/integrations.md](docs/integrations.md)

## 项目结构

```
compass/
├── src/compass/
│   ├── cli/                  # CLI 入口
│   ├── core/
│   │   ├── scenario.py       # 场景定义
│   │   ├── runner.py         # 测试运行器（含审计溯源盖章）
│   │   ├── transcript.py     # Transcript / Outcome / ToolCall 协议
│   │   ├── trial.py          # 试验管理
│   │   ├── result.py         # 结果数据结构
│   │   ├── metrics.py        # pass@k / pass^k 计算
│   │   ├── checkpoint.py     # 断点续跑（场景指纹）
│   │   └── sweep.py          # 参数扫描
│   ├── graders/              # 评分器（Transcript/Outcome 分离设计）
│   │   ├── base.py           # 基类、GraderScope、GradeContext
│   │   ├── registry.py       # 评分器注册表
│   │   ├── code/             # Code Graders（common / coding / data / image）
│   │   ├── model/            # Model Graders（semantic / vlm / rubric / trajectory / groundedness...）
│   │   └── human/            # Human Graders（human_review / pairwise）
│   ├── adapters/             # Agent 适配器（image / coding / environment）
│   ├── integrations/         # 外部轨迹导入（openai_agents / pi / otlp / claude_agent）
│   ├── sandbox/              # 沙箱执行
│   └── report/               # 报告与分析（analyzer / console / html / compare）
├── tests/
├── docs/                     # 专题文档（见"核心特性与文档导航"）
├── examples/
└── pyproject.toml
```

## 路线图

Phase 1（MVP）与 Phase 2（核心功能）已全部完成；Phase 3+ 聚焦沙箱隔离增强、多模态 Artifact、Web Dashboard 与 CI/CD 集成。详细清单见 [docs/roadmap.md](docs/roadmap.md)。

## 参考资料

- [Anthropic: Demystifying Evals for AI Agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- [OpenAI: Inside Our In-house Data Agent](https://openai.com/index/inside-our-in-house-data-agent/) — Data Agent 评估设计参考
- [OpenAI: Image Evals for Image Generation and Editing Use Cases](https://developers.openai.com/cookbook/examples/multimodal/image_evals)
- [OpenAI: Evals Framework](https://github.com/openai/evals)
- [Stripe: Can AI agents build real Stripe integrations?](https://stripe.com/blog/can-ai-agents-build-real-stripe-integrations)
- [LangChain: Evaluation](https://python.langchain.com/docs/guides/evaluation)

## 许可证

MIT License
