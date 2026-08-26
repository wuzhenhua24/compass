# Compass — Agent Evaluation Substrate

Compass 是 **Agent 评测的基座（substrate）**：提供一套标准的执行轨迹模型、多来源轨迹接入、可复用的过程评分器与可靠性指标——领域相关的"答案对不对"由你用几十行自定义 grader 补齐。它之于 Agent 评测，就像 pytest 之于测试、OpenTelemetry 之于可观测：**框架给骨架和标准，业务判定你来写**。

> 设计理念参考 [Anthropic: Demystifying Evals for AI Agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) 与 [Hidden Technical Debt of AI Systems: Agent Evaluation Infrastructure](https://leehanchung.github.io/blogs/2026/06/13/hidden-technical-debt-agent-evaluation-infra/)

## 定位：是什么 / 不是什么

面对市面上千差万别的 Agent 场景，Compass **不追求"开箱即评一切"**——那不现实，定制化必然存在。它的目标是把**所有 Agent 都需要的那层基座做厚**，让**每个 Agent 特有的定制层尽量薄**。定制不是缺陷，是产品留给你的插槽；框架的活是让它变小。

**✅ 是什么**

- **一套标准**：Transcript（怎么做的）/ Outcome（做出了什么）+ ToolCall 协议，让评分器面向统一数据结构、跨 Agent 复用（[core-design](docs/core-design.md)）
- **轨迹接入**：OpenAI Agents SDK / pi / Codex / OTLP·OpenInference / Claude Agent SDK / ATIF·Harbor 的原生轨迹归一成 Transcript（[integrations](docs/integrations.md)）
- **可复用的过程评分器**：`cost_budget` / `latency_budget` / `loop_detection` / `tool_usage` / `state_delta`，外加两个 LLM 判官——`trajectory_judge`（调用链是否合理）与 `groundedness`（答案是否被工具观察支撑，专抓"空工具结果幻觉"）（[graders](docs/graders.md)）
- **执行与评分解耦**：轨迹是不可变证据，`compass grade` 给已落盘的轨迹打分——改判分器不用重跑 Agent，判分器指纹自动标记过期评分（[analysis](docs/analysis.md)）
- **评测卫生**：harness 失败不算模型失败（移出 pass rate 分母）、未打分 ≠ 0 分（判官超时不伪装成低分，也不许签发"通过"）、多试验轮转采样让 pass^k 不被偏样本污染
- **观察标签与指标**：grader 对通过的样本也打标签和定量指标，聚合成「短回答占 67%、其中只有 25% 通过」这类行为画像——失败统计给不出
- **有据可依的结论**：`compass insights` 让模型下结论，但**逐条拿数据核**，核不过的整条丢弃并报出理由
- **可靠性与工程底座**：pass@k / pass^k 无偏估计、`compass compare` 配对比较（翻转 + 置信区间）、多模型排行榜、checkpoint 续跑、并行执行、审计溯源（run_id / config_hash / grader_version）
- **扩展面**：`external_checker` 让**任何可执行文件**成为 grader（契约是进程边界，checker 不必 import Compass）；评分流水线共享 workspace，`creates:` 让产出可验证、`required:` 失败即中止省下昂贵调用
- **Skill 评测**：把"装哪个版本"变成可扫的轴，量触发率与误触发，查 bundle 脚本是否真被用上（[skills](docs/skills.md)）
- **发布即脱敏**：`site build` 与非 loopback 的 `site serve` 写出前扫一遍凭据，命中数报出来而不是悄悄删
- **领域 recipe（可选）**：[`examples/`](examples/) 四个模板是"如何自己写定制层"的样板，都能离线跑

**🚫 不是什么**

- 不是"开箱评测任意 Agent"的银弹——**领域正确性判定必然要你写**（这是设计，不是缺陷）
- 不内置**每个**领域的正确性 grader（图像美学、代码功能、答案事实……天然定制）。随包带的 `coding` / `data` / `image` 三个域是**用户代码的样板**，不是承诺——你的领域不在这三个里是常态，那正是你写 grader 的地方
- 不强求"驱动任意 Agent 跑"——自跑的 Agent 走**导入轨迹**更合适

### 控制面 / 数据面：这条边界的专业名字

[Hidden Technical Debt of AI Systems](https://leehanchung.github.io/blogs/2026/06/13/hidden-technical-debt-agent-evaluation-infra/) 把 agent 评估基础设施拆成控制面（跑什么、结果是否该改变发布决策）与数据面（真正跑 agent 并记录发生了什么），恰好精确描述了 Compass 的边界：

**Compass = 控制面 + 连接两层的 trace schema；数据面归你。** Scenario / Grader / 聚合 / pass^k / 报告 / 续跑是控制面，Transcript + ToolCall 协议是那份 trace schema，而 agent 怎么跑、沙箱、世界状态属于数据面——Compass 只通过 Adapter 驱动或 Import 消费轨迹与之接触，不试图拥有它。这也解释了上面的"不是什么"：**数据面千差万别、不可复用，控制面与 trace 标准才是可复用资产。**

### 核心 vs 领域：一条干净的分界线

哪些评分通用、哪些必然定制，由 **GraderScope**（Transcript/Outcome 分离）直接预测：

| | 域 | 例子 | 谁写 |
|---|---|------|------|
| **过程** | TRANSCRIPT | 成本 / 延迟 / 绕圈 / 工具使用 / 是否执行危险操作 | ✅ 框架内置，跨 Agent 复用 |
| **正确性** | OUTCOME | 答案对不对 / 图美不美 / 代码能不能跑 | ✍️ 你写领域 grader（通常几十行） |

这条线在目录里是看得见的：`compass/graders/`（框架 20 个）对 `compass/graders/domains/`（领域 23 个），`compass list` 也分两栏打印。领域包**查不到时才加载**——`import compass` 不会为你不评的领域付出代价。

> **纪律**：核心 schema 保持小。领域字段（如 ops_qa 里的 `expected_doc` / `key_facts`）放进 grader config 或用户 harness，**别塞进核心模型**——这是防止"定制爆炸"淹没框架的关键。（反例参照物：`reference_answer` 之所以能进 `GradeContext`，是因为它像 `reference_image` 一样是跨领域的"参考数据"通用概念。）

### 何时用 Compass / 何时直接 DIY

Compass 在这些场景最省事；否则一个几十行的 pytest 可能就够了——框架不假装自己永远划算：

| 用 Compass 划算 | DIY 就好 |
|-----------------|----------|
| 关心**过程**（成本/工具/安全/检索/绕圈），不只最终答案 | 只需"最终字符串 vs golden"比对 |
| 有**多个** Agent / 框架，想要一套评测词汇 | 一次性的单个 Agent |
| 要**可靠性指标**（pass^k）与趋势 | 一次性 check 就行 |
| 评一个**你不驱动、只拿到轨迹**的 Agent | 你完全掌控 Agent 的进出 |

## 核心概念

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
1. 接口层      compass CLI · Python SDK · compass site（静态站）
2. 编排层      Scenario Engine（YAML）· Trial Manager（多试验）· Parallel Executor
3. 核心引擎    Grader System（Code / Model / Human 三层）
               Transcript + ToolCall 协议 · ArtifactStore（Image/Code/Text，插件式）
               Metrics（pass@k / pass^k）· Report（console / html / site / compare / insights）
4. 接入层      Adapter      Compass 亲自驱动：image · coding · environment ·
                            claude_code · pi · codex（后三者共用 cli_agent.py）
               Integrations 消费自产轨迹：openai_agents · pi · codex_exec ·
                            otlp · claude_agent · atif

横跨 3 / 4     compass/llm/      定价 · token 用量提取 · 结构化输出（控制面）
               compass/sandbox/  隔离 workdir + 执行 + 收集（adapter 与 coding 域 grader 共用）
```

> **两种接入，同一个 Transcript。** Adapter 是 Compass 亲自驱动 Agent（适合图像端点、代码沙箱
> 这类可被外部调用的场景）；Integrations 是消费 Agent 自产的轨迹（适合自跑的 LLM Agent）。二者
> 产出的都是同一个 Transcript，下游评分与指标完全共用。ComfyUI / SD WebUI / Midjourney 这类目标
> 用 `@register_adapter` 自己接——见 [docs/integrations.md](docs/integrations.md)。

### CLI 命令总览

| 命令 | 用途 |
|------|------|
| `compass test <scenario.yaml \| 目录>` | 运行测试场景；`-m` 可一次跑多个模型并出排行榜 |
| `compass grade <traces> -s <scenario.yaml>` | **离线评分**：给已落盘的轨迹打分，不重跑 Agent |
| `compass analyze <results>` | 分析评估结果：Scope 分维度、失败模式、改进建议 |
| `compass compare <a.json> <b.json>` | 配对比较两次运行：case 翻转 + 置信区间 + MDE |
| `compass insights <results> [<b.json>]` | 让模型下结论，但每条断言都拿数据核过，过不了的丢弃 |
| `compass site build <results>` | 把结果发布成可分享的静态站（多仓库可 build 进同一目录） |
| `compass site compare <a> <b>` | 把两次运行的**配对比较**发布进同一个站点 |
| `compass site serve <results \| site>` | 本地实时查看：每个请求现算，跑到一半的运行也能看 |
| `compass trace <trace 文件>` | 查看执行轨迹（JSON/JSONL，`--steps` 展开工具调用） |
| `compass import <trace 文件>` | 导入外部轨迹（pi / Codex / OTLP / Claude / ATIF·Harbor，自动识别） |
| `compass init [output.yaml]` | 生成场景模板 |
| `compass docs [topic]` | 在终端里打印 Compass 自身文档（raw markdown，可管道） |
| `compass list` | 列出已注册的 grader 和 adapter |
| `compass checkpoint list/clean` | 管理运行断点（配合 `compass test --resume`） |

各命令的完整选项见 `compass <命令> --help`。

## 安装

> Compass **尚未发布到 PyPI**（当前处于内部试用阶段），只能从源码装。不要 `pip install compass-qa`——
> 那个名字不在我们手上，装到的不会是这个项目。

**A. 试用 / 开发 Compass 本身**——在 checkout 里直接跑：

```bash
git clone https://github.com/wuzhenhua24/compass.git
cd compass
uv sync
uv run compass --help
```

`uv sync` 建好 `.venv` 并把 `compass` 命令装进去。本文档后续示例都写成裸 `compass ...`；
在 checkout 里请前缀 `uv run`（或 `source .venv/bin/activate` 后直接用）。

**B. 在你自己的项目里用**——推荐这种，自定义 grader 和你的业务代码同处一个环境：

```bash
uv pip install -e /path/to/compass     # 可编辑安装，Compass 更新后无需重装
compass list                           # 验证：应列出 43 个 grader（框架 20 + 领域 23）和 6 个 adapter
```

**领域后端按需装（extras）**——框架本体和 43 个 grader 都会装上，两个域的**重后端**是可选的，缺了的 grader 在结果里说明情况而不是 import 报错：

```bash
uv pip install -e '/path/to/compass[data]'    # sqlparse + duckdb → sql_syntax / sql_equivalence
uv pip install -e '/path/to/compass[image]'   # torch + open-clip → semantic_match / aesthetic_score
uv pip install -e '/path/to/compass[all]'     # 两个都要
```

`coding` 域调用你项目自己的工具链（ruff / mypy / pytest），没有对应的 extra。在 checkout 里开发时 `uv sync` 已包含 `data` 后端与测试/lint/类型工具，`image` 后端用 `uv sync --extra image`。

## 快速开始

```bash
compass init my_scenario.yaml                            # 1. 生成场景模板

compass test scenarios/my_test.yaml                      # 2. 运行
compass test scenarios/ -p -w 8 --report html -o r.html \
    --trace-dir ./traces --trace-format jsonl            #    并行 + 报告 + 落盘轨迹
```

**留住轨迹**（`--trace-dir`）是后面每一步的前提：离线评分、配对比较、发布站点都读它。多次试验数在 YAML 里配（`defaults.trials` 或 case 级 `trials:`），不是 CLI 选项。

```bash
# 3. 离线评分：跑一次，评多次——轨迹是不可变证据，评分只往 traces/grades/<name>/ 里新增
compass grade ./traces -s scenarios/my_test.yaml --regrade   # 改了 rubric / 阈值后重评
compass grade ./traces -s judge.yaml -n judge -o judge.json  # 另一套 grader 评同一批轨迹

# 4. 分析
compass analyze results/                       # 分维度诊断：Scope 对比、失败模式、改进建议
compass compare results_a/ results_b/          # 配对比较：case 翻转 + 置信区间
compass site build results.json -o site/       # 发布静态站
compass trace traces/cat_on_sofa.json          # 看单个 case 的轨迹（--steps 展开工具调用）
```

判分器配置一改，对应 case 的评分自动标记为过期（内容指纹，不靠手工 bump 版本号）。这也是让 `compass compare` 有意义的前提：**同一批轨迹重评，分数差异只可能来自判分器，而不是 Agent 的随机性**（[docs/analysis.md](docs/analysis.md)）。

### Python SDK

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
| **速查表** | 一页读完就能写出正确的 scenario 和 grader：YAML 全字段、43 个内置评分器（框架 20 / 领域 23）、常用配方、易踩的语义坑 | [docs/cheatsheet.md](docs/cheatsheet.md) |
| **核心设计** | Transcript/Outcome 分离、GraderScope、ToolCall 协议（当前 2.0：多 Agent 字段、state_delta、审计溯源）、JSONL 事件流、成本/token 聚合 | [docs/core-design.md](docs/core-design.md) |
| **Grader 体系** | 三层体系（Code/Model/Human）、框架 20 个 + 领域 23 个内置评分器、expected 简化配置、正负向测试、泄漏检测、自定义 grader | [docs/graders.md](docs/graders.md) |
| **场景配置与指标** | 场景 YAML 完整参考、多次试验、pass@k / pass^k、分类聚合（category/tags） | [docs/scenario-config.md](docs/scenario-config.md) |
| **分析与报告** | `compass analyze` 分维度诊断、`compass compare` 配对比较（翻转 + 置信区间 + MDE）、`compass insights` 逐条核过的 LLM 结论、best-of-k、HTML 报告 | [docs/analysis.md](docs/analysis.md) |
| **接入外部 Agent** | 六种轨迹导入、Claude Code · pi · Codex 三个 CLI Adapter（真实仓库上跨栈对比编程 Agent）、Environment Adapter、自定义 Adapter、黑盒 Agent 的 ToolCall 获取 | [docs/integrations.md](docs/integrations.md) |
| **Skill 评测** | 把"装哪个版本"变成可扫的轴、`skill_trigger` 量触发率、负向控制抓误触发、`required_resources` 查 bundle 脚本、v1→v2 的配对判断 | [docs/skills.md](docs/skills.md) |

## 端到端示例（四个模板，都能离线跑，不需要 API key、不花钱）

每个都演示一类评测里最容易做错的事。离线不是靠绕过流水线——跑的是真的 adapter、真的 git
worktree、真的 grader，唯一的替身是 agent CLI 本身。完整讲解在各自的 README：

| 模板 | 它演示的那件事 | 跑 |
|---|---|---|
| **文档问答**<br>[`examples/ops_qa/`](examples/ops_qa/README.md) | doc-grounded 问答不能只判「语义对不对」：检索命中与越权写从 `transcript.tool_calls` 确定性地评，安全评的是**执行**而非文字 | `python examples/ops_qa/eval.py` |
| **编程 Agent**<br>[`examples/coding_agent/`](examples/coding_agent/README.md) | 编程 agent 搞砸评测有三种方式（老实做 / 改测试作弊 / 结果对但绕远路），单一 pass/fail 会把它们糊成一团 | `python examples/coding_agent/eval.py` |
| **Skill 版本对比**<br>[`examples/skill_eval/`](examples/skill_eval/README.md) | 证明 v2 是优化而不是改坏：触发率显著提升但总体判定仍在噪声带里，而负向用例暴露出 v2 开始抢别人的活 | `python examples/skill_eval/eval.py` |
| **公开数据集**<br>[`examples/swebench/`](examples/swebench/README.md) | 官方判定协议的三个坑：测试对 agent 全程不可见、先重置测试文件再打 patch、没跑起来的测试算失败 | `python examples/swebench/demo.py` |

> 在 checkout 里请前缀 `uv run`。

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
│   ├── cli/            # CLI：app.py 是 Click group，commands/ 一个命令一个模块
│   ├── core/           # Scenario / Runner / Transcript · ToolCall 协议 / Artifact /
│   │                   #   Trial / Metrics / 离线评分与判分器指纹 / checkpoint / sweep
│   ├── graders/        # 框架的 Code · Model · Human 评分器（过程与可靠性）
│   │   └── domains/    #   领域正确性判定，查不到时才加载：coding / data / image
│   ├── adapters/       # Agent 适配器：image / coding / environment / claude_code / pi / codex
│   ├── integrations/   # 外部轨迹导入：openai_agents / pi / codex_exec / otlp / claude_agent / atif
│   ├── llm/            # 控制面的 LLM 管道：定价 / token 用量提取 / 结构化输出
│   ├── sandbox/        # 隔离 workdir + 执行 + 收集（adapter 与 coding 域 grader 共用）
│   ├── report/         # console / html / site / analyzer / compare / leaderboard / insights
│   └── docs_index.py   # 随包分发的文档索引（撑起 compass docs）
├── tests/
├── docs/               # 专题文档（见"核心特性与文档导航"）
├── examples/
└── pyproject.toml
```

## 路线图

Phase 1（MVP）、Phase 2（核心功能）已全部完成，Phase 3（增强）只剩"更多图像 Adapter"未做——且已主动降优先级：自跑的 Agent 走**导入轨迹**比被外部驱动更合适。原路线图之后的那批基座化工作（轨迹接入、`compass grade` 执行/评分解耦、ToolCall 协议 2.0、观察标签与指标、静态站）归档在 Phase 5。

**当前未做**：Web Dashboard 的常驻服务形态（静态站 `compass site` 已交付）、现成的 CI/CD 流水线模板（退出码门禁已有）、"LLM 判官 vs 人工标注"的校准工具。详细清单见 [docs/roadmap.md](docs/roadmap.md)。

## 参考资料

- [Anthropic: Demystifying Evals for AI Agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- [OpenAI: Inside Our In-house Data Agent](https://openai.com/index/inside-our-in-house-data-agent/) — Data Agent 评估设计参考
- [OpenAI: Image Evals for Image Generation and Editing Use Cases](https://developers.openai.com/cookbook/examples/multimodal/image_evals)
- [OpenAI: Evals Framework](https://github.com/openai/evals)
- [Stripe: Can AI agents build real Stripe integrations?](https://stripe.com/blog/can-ai-agents-build-real-stripe-integrations)
- [LangChain: Evaluation](https://python.langchain.com/docs/guides/evaluation)

## 许可证

MIT License
