# 接入外部 Agent：Integrations 与 Adapter

> Compass 专题文档 · 返回 [README](../README.md)


## 接入外部 Agent 轨迹（无需写 Adapter）

> 设计目标：评估一个在 Compass **之外**运行的 Agent，不用为它写专门的 Adapter——直接把它自带的 trace 桥接成 Compass `Transcript`，再用现有的 Transcript 评分器打分。

Compass 的 Adapter 层适合"Compass 亲自驱动 Agent"的场景（图像/代码沙箱）。但现代 Agent SDK 大多**自带埋点**，为每个 Agent 写 Adapter 不现实。因此提供 `compass.integrations`——把外部 SDK 的原生轨迹**导入**为 Transcript。

**首个集成：OpenAI Agents SDK**（`compass.integrations.openai_agents`）

OpenAI Agents SDK 用类型化 span（`function`/`generation`/`response`/`agent`/`turn`/`handoff`…）记录执行，并开放可插拔的 `TracingProcessor`。Compass 实现了这样一个 processor，用户只加一行：

```python
from agents import Agent, Runner
from compass.integrations import install_openai_agents_processor

proc = install_openai_agents_processor()     # 注册到 SDK，仅此一行
await Runner.run(agent, "今天天气如何？")
transcript = proc.latest                      # 得到 Compass Transcript，可直接评分
```

拿到 `Transcript` 后走正常评分路径：

```python
from compass.graders import get_grader, GradeContext

grader = get_grader("cost_budget")({"max_cost_usd": 0.5})
result = await grader.grade(GradeContext(transcript=transcript, outcome=transcript.outcome))
```

**映射关系**

| SDK span | → Compass |
|---|---|
| `function`（工具/MCP 调用）| `ToolCall`（`tool_type=function`/`mcp`）|
| `generation` / `response`（LLM）| `ToolCall`（`tool_type=llm` + `tokens`；成本用可配置定价 `calculate_cost` 补算）|
| `agent` / `turn` | 通过父链回溯，给每个 `ToolCall` 写入一等字段 `agent_name` / `turn_index` |
| `handoff` / `guardrail` | 记入 `Transcript.metadata` |

**设计要点**
- **零硬依赖**：processor 是鸭子类型，Compass 不 import `agents`；只有 `install_openai_agents_processor()` 内部才惰性导入。
- **成本协同**：SDK span 只带 token、不带美元 → 复用 Compass 的可配置价格表补算。
- 其余框架（LangChain / LlamaIndex / CrewAI 等）后续通过离线 OTLP/OpenInference 导入或各自的 JSONL session 导入接入。

**第二个集成：pi（`@earendil-works/pi-*`）JSONL session 导入**（`compass.integrations.pi_sessions`）

与 OpenAI 的实时 processor 不同，pi 把一次运行**落盘为 JSONL session 树**（首行 session 头，之后每行一个 `SessionTreeEntry`，靠 `parentId` 连成树/分支）。这是一个**离线导入器**——读文件、重建 Transcript：

```python
from compass.integrations import import_pi_session, import_pi_sessions

t = import_pi_session("~/.pi/sessions/2026-01-01-weather.jsonl")
# 或批量导入一个目录
transcripts = import_pi_sessions("~/.pi/sessions/")
# t.tool_calls / t.reasoning_steps / t.outcome 均已就绪，可直接评分
```

**映射关系**

| pi 条目 | → Compass |
|---|---|
| session 头 `{id, cwd, timestamp}` | `trial_id` / `metadata` / 时间轴 |
| assistant 消息的 `usage` | 一个 `llm.generation` `ToolCall`（`tokens` + **原生 `cost`**；pi 没记成本时才用可配置价格表补算）|
| assistant 的 `toolCall` block | `ToolCall`（按 `toolCallId` 匹配对应的 `toolResult` 回填 output/status/duration）|
| assistant 的 `thinking` block | 一条 reasoning step |
| `user` 消息 | `input_prompt`（首条）/ 后续 reasoning |
| `compaction` / `branch_summary` / `model_change` / `custom` … | `metadata` / reasoning |

**设计要点**
- **只取活跃分支**：默认从 session 当前 leaf 回溯到 root，被放弃的 fork 不进入评估；线性会话等价于文件顺序，异常时回退文件顺序（`active_branch_only=False` 可取全量）。
- **成本优先用原生**：pi 的 `Usage` 自带美元成本，直接采用；缺失时才用 `calculate_cost` 补算。
- **鲁棒**：坏行跳过、缺 header 抛 `PiSessionError`、孤儿 `toolResult` 也保留。
- **零依赖**：纯 JSON 解析，不 import 任何 pi 包。

**第三个集成：OTLP / OpenInference 通用导入**（`compass.integrations.otlp`）

前两个集成针对具体 SDK。**多数其它框架**（LangChain / LlamaIndex / CrewAI / Haystack / DSPy…）没有 Compass 友好的原生 trace，但都能用 **OpenInference** 语义约定埋点、经 **OpenTelemetry（OTLP）** 导出（如通过 Arize Phoenix）。这个导入器读一份导出的 trace 文件，**每条 trace 重建一个 Transcript**：

```python
from compass.integrations import import_otlp_file

transcripts = import_otlp_file("phoenix_export.json")   # 一条 trace 一个 Transcript
t = transcripts[0]
```

**兼容的输入形态**（自动识别）：OTLP/JSON 信封（`resourceSpans` + `[{key,value:{stringValue}}]` 列表属性）、扁平 span 列表 / OTel SDK `ReadableSpan.to_json()`（属性为扁平 dict）、以及 JSON 或 JSONL（逐行）。

**映射关系**（由 `openinference.span.kind` 驱动）

| span kind | → Compass |
|---|---|
| `LLM` / `EMBEDDING` | `ToolCall`（`tool_type=llm` + `tokens`；成本优先用原生 `llm.cost.*`，缺失才补算）|
| `TOOL` | `ToolCall`（`tool_type=function`，`tool.parameters` 作为 input，JSON 自动解析）|
| `RETRIEVER` | `ToolCall`（`tool_type=search`，`retrieval.documents.*` 还原为文档列表）|
| `AGENT` / `CHAIN` | 结构性——父链回溯写入一等字段 `agent_name`；root chain 的 `input.value`/`output.value` 作为 prompt/outcome |
| `GUARDRAIL` / `RERANKER` / `EVALUATOR` | 记入 `Transcript.metadata` / reasoning |

**设计要点**
- **双属性编码**：同时兼容 OTLP 线格式（`[{key,value}]`，含 `intValue` 字符串编码）与 Phoenix/SDK 的扁平 dict。
- **成本协同**：与前两者一致，原生 `llm.cost.total` 优先，否则用可配置价格表 `calculate_cost` 补算。
- **鲁棒**：缺 id 的 span 跳过、空 payload / 全垃圾抛 `OTLPImportError`、时间戳兼容纳秒/微秒/毫秒/秒与 ISO 字符串。
- **零依赖**：不 import 任何 opentelemetry / openinference 包，纯 JSON 解析。

**第四个集成：Claude Agent SDK 消息流**（`compass.integrations.claude_agent`）

Claude Agent SDK（`claude_agent_sdk`）架构独特——它**自己不跑 agent loop**，而是把 Claude Code CLI 当子进程驱动，回吐一串类型化、Anthropic 原生形状的 `Message` 流（`query()` / `ClaudeSDKClient.receive_response()`）。这串消息是 SDK 的**稳定公开契约**，所以我们消费它（而非 CLI 那份刻意内部化的落盘 JSONL）：

```python
from claude_agent_sdk import query
from compass.integrations import reconstruct_transcript

messages = [m async for m in query(prompt="...")]
transcript = reconstruct_transcript(messages)   # 也可用 reconstruct_transcript_from_stream 直接吃异步流
```

也有**离线路径**：直接读 CLI 的 `--output-format stream-json` 落盘文件（每行一个 wire dict），无需 import SDK：

```python
from compass.integrations import import_claude_stream_json

transcript = import_claude_stream_json("run.stream.jsonl")   # 复用同一套映射
```

**映射关系**

| SDK 消息 | → Compass |
|---|---|
| `AssistantMessage`（`usage`/`model`）| `llm.generation` ToolCall（`tokens`；`turn_index` 按 assistant 消息递增）|
| 其中的 `ToolUseBlock` | `ToolCall`，按 `tool_use_id` 匹配结果 |
| **`UserMessage` 携带的 `ToolResultBlock`** | 回填对应调用的 output/status ← **这就是特殊点：Claude Code 约定工具结果是 user 角色消息** |
| `ServerToolUseBlock`/`ServerToolResultBlock`（web_search/web_fetch/advisor）| 服务端执行的 `ToolCall`（内联结果）|
| `ResultMessage` | outcome(`result`) + duration + **CLI 已算好的 `total_cost_usd`** |
| `parent_tool_use_id` + `Task` 调用 | 子 agent 归属 → 一等字段 `agent_name` |

**设计要点**
- **消费公开契约而非内部落盘**：SDK 明说磁盘 transcript 是"内部 discriminated union，当作不透明 blob"，所以走稳定的 `Message` 流。
- **成本用 CLI 权威总额**：CLI 只报一个 `total_cost_usd`（比逐调用美元更准），挂到终局 llm 调用上，`sum_cost`/`cost_budget` 即得全程真实成本——**无需价格表**。
- **子 agent 归属**：`Task` 工具的 `tool_use_id` 与后续消息的 `parent_tool_use_id` 对上，还原 `agent_name`。
- **零依赖**：鸭子类型读属性，不 import `claude_agent_sdk`。

至此，四个集成覆盖了「实时 span processor（OpenAI）+ 离线 SDK session（pi）+ 通用 OTLP/OpenInference（其余框架）+ 子进程消息流（Claude Agent SDK）」，评估外部 Agent 基本不再需要为每个框架写 Adapter。

**命令行统一入口：`compass import`**

三个**文件型**导入器（pi、OTLP/OpenInference、Claude stream-json）统一挂到了一个子命令上——自动识别格式、重建为 Compass transcript、打印摘要，并可保存后用 `compass trace` 查看：

```bash
compass import session.jsonl                    # 自动识别 + 摘要
compass import phoenix_export.json -o out/      # 多 trace：每条存一个文件
compass import run.stream.jsonl -f claude -o t.json   # Claude stream-json + 存单文件
compass import phoenix_export.json --json       # 打印重建后的 transcript JSON
```

格式自动识别（各用其首行不变量）：pi 会话首行是 `{"type":"session"}`；Claude stream-json 首行 `type` 是 CLI 消息类型（`assistant`/`user`/`result`/`system`…）；其余 JSON 按 OTLP/OpenInference 处理。（OpenAI Agents SDK 是实时集成，编程方式经 `compass.integrations` 使用，不走文件导入。）

**流式重建：`WireReconstructor`**

上面三个入口都是「跑完再解析」。要在 CLI **还在跑的时候**逐行喂，用增量版——同一套映射，只是换成 push 接口：

```python
from compass.integrations import WireReconstructor

r = WireReconstructor(task_id="add-rate-limit")
for line in proc.stdout:
    r.feed_line(line)          # 非 JSON 行返回 False，不抛
transcript = r.finish()        # 任何时刻调用都合法
```

两个好处直接落到评测上：**被 timeout 杀掉的运行仍然留下已完成的步骤**（`finish()` 中途可用），以及不必再写第二个解析器就能做实时进度视图。下面的 Claude Code Adapter 正是靠它工作的。

## Claude Code Adapter（在真实仓库上评编程 Agent）

`claude_code` adapter 面向一类具体需求：**同一批业务需求，换 Prompt / 换模型，看谁实现得更对、更省、更稳**。它把 `claude` CLI 跑在一个真实 git 仓库上，同时交出两样东西——**Agent 产出的 diff**（给 OUTCOME grader）和**它是怎么做到的完整轨迹**（给 TRANSCRIPT grader）。

### 为什么不用 `environment` adapter 直接 `agent_command: claude -p ...`

因为那样**过程侧全瞎**。`environment` 把整次运行记成**一条** `environment.agent` ToolCall（退出码 + stdout 长度），而 Compass 最值钱的东西——30 步 Edit/Bash/Read、每轮 token、CLI 报的成本、重试、循环、子 agent 扇出——全埋在那坨不透明的 stdout 里。`cost_budget` / `turn_count` / `loop_detection` / `tool_usage` / `efficiency` / `trajectory_judge` 一个都用不上。

`claude_code` 用 `--output-format stream-json --verbose` 起 CLI，边流边过 `WireReconstructor`，把还原出的调用**以对象形式**并进 runner 的 live transcript（不是拆成 kwargs 重建——那会丢掉 `call_id`/`turn_index`/`agent_name`，而这正是 `turn_count` 和子 agent 归属要读的字段）。

### 隔离靠 git worktree，不靠 sandbox

这是刻意的。`LocalSandbox` 是一个被洗过环境变量的临时目录；而 Claude Code CLI 要真实网络、真实凭证、真实 git 仓库、项目自己的工具链——四条全跟 sandbox 打架。所以 adapter 直接以 `cwd=<worktree>` 起进程，隔离交给 git：

| `isolation` | 行为 | 适用 |
|---|---|---|
| `worktree`（默认）| `git worktree add --detach` 从 `base_ref`（默认当前 HEAD）拉一份 | git 仓库；并行 trial 互不干扰，diff 基线明确 |
| `copy` | `copytree` 复制一份 | 非 git 目录 |
| `none` | 直接在原仓库里跑，**会修改它** | 单次手工调试 |

前两种下**源仓库全程只读**。

### 产出物是 diff，不是全量文件快照

收尾时跑 `git add -A -N`（intent-to-add，让新文件也出现在 diff 里）再 `git diff`，填进 `CodeArtifact.diff`；`files` 只装**改动过**的文件，不是整个仓库。

### keep_workspace 默认为 True

Grader 在 adapter 返回**之后**才跑，而 `integration_test` 靠对 Agent 产出的那棵树跑隐藏测试来打分——先把树删了，等于在任何人读到之前销毁证据。路径记在 `CodeArtifact.metadata["workspace"]`，`integration_test` 的 `workdir: "{workspace}"` 会解析到它。事后回收磁盘：在源仓库里 `git worktree prune`。

### 完整例子

```yaml
name: "编程 Agent — 业务需求实现"
agent:
  adapter: claude_code
  config:
    repo: "./fixtures/billing-service"
    base_ref: "main"
    permission_mode: acceptEdits
    allowed_tools: ["Edit", "Write", "Bash(pytest:*)"]
    max_turns: 40
    timeout: 900
    setup_commands: ["uv sync"]
    save_stream_to: "./streams"        # 存原始 stream-json，之后免费重放

cases:
  - id: add_rate_limit
    input:
      prompt: "给 /api/charge 加限流：每 IP 每分钟 60 次，超出返回 429。"
    graders:
      - {name: integration_test, type: code, gate: true,       # 对不对（隐藏测试）
         config: {script: "pytest tests/ -q", workdir: "{workspace}", output_format: pytest}}
      - {name: diff_size, type: code, config: {max_total_changes: 200}}   # 改动是否收敛
      - {name: cost_budget, type: code, config: {max_cost_usd: 1.5}}      # 以下四个：过程
      - {name: turn_count, type: code, config: {max_turns: 30}}
      - {name: loop_detection, type: code}
      - {name: tool_usage, type: code, config: {forbidden_tools: ["WebFetch"]}}
```

两条轴都只是 `agent.config` 里的一个键，所以扫哪条都是普通的多变体运行：

```bash
compass test coding.yaml -m claude-opus-4-6 -m claude-sonnet-5           # 模型轴
compass test coding.yaml --model-key append_system_prompt_file \
    -m prompts/terse.md -m prompts/thorough.md                           # Prompt 轴
```

跑一次很贵，所以配 `--trace-dir` 记录，之后改 rubric 用 `compass grade` 离线重评，不必重跑 Agent。

### 凭证：`env_allow`

`claude_code` 自己不走 sandbox，用宿主环境，所以没有这个问题。但如果你用 `environment` adapter 跑**别的**带凭证的 agent（goose、aider），会撞上 sandbox 的密钥过滤器：`ANTHROPIC_API_KEY` 同时命中 `ANTHROPIC_` 前缀和 `API_KEY` 子串，`env_passthrough` / `env_overrides` / 每次调用的 `env=` 三条路**全部**被拦——不给逃生舱的话，这类运行根本没法认证。

```yaml
sandbox_config:
  env_allow: ["ANTHROPIC_API_KEY"]   # 显式变量名，不支持通配
  preserve_home: true                # 订阅制凭证在 ~/.claude
```

`env_allow` 刻意做得很窄：只吃**变量名**（大小写不敏感、无 glob），放行一个凭证不会顺带放行一类。每个放行的名字都会在 setup 时打 WARNING 日志。代价要认：被测 agent 以及它跑起来的任何代码都能读到这个凭证——请用一把限定权限的 key，不要用生产 key。

## Environment Adapter（环境即代码）

受 [Stripe Agent Benchmark](https://github.com/stripe/ai/tree/main/benchmarks) 启发，Compass 提供了 **EnvironmentAdapter**——将 Agent 的运行环境视为代码来管理，实现可复现的端到端评估。

### 设计理念

在评估 Coding Agent 时，仅检查代码输出是不够的——Agent 需要在真实的软件环境中运行（安装依赖、启动服务、操作数据库）。EnvironmentAdapter 将这一过程标准化为 5 个阶段：

```
┌───────────────────────────────────────────────────────────┐
│                  EnvironmentAdapter 生命周期                │
│                                                           │
│  1. Copy     ──→ 将 environment/ 目录复制到沙箱            │
│  2. Setup    ──→ 运行 setup_commands（安装依赖、启动服务）  │
│  3. Agent    ──→ 执行 agent_command（调用 AI Agent）       │
│  4. Teardown ──→ 运行 teardown_commands（清理资源）        │
│  5. Collect  ──→ 收集输出文件，返回 CodeArtifact           │
└───────────────────────────────────────────────────────────┘
```

### YAML 配置示例

```yaml
name: "Stripe Integration Test"
agent:
  adapter: environment
  config:
    environment_dir: "./benchmarks/galtee-basic/environment"
    agent_command: "goose run --prompt '{prompt}'"
    problem_file: "PROBLEM.md"
    setup_commands:
      - "npm --prefix server install"
      - "npm --prefix server start &"
    teardown_commands:
      - "pkill -f 'node server' || true"
    timeout: 600
    env:
      STRIPE_SECRET_KEY: "sk_test_..."
      DATABASE_URL: "postgresql://localhost/test"

cases:
  - id: "basic_checkout"
    input:
      prompt: "Implement a Stripe checkout flow"
    graders:
      - name: integration_test
        config:
          test_command: "npm --prefix grader test"
          output_format: auto
```

### 核心特性

| 特性 | 说明 |
|------|------|
| **环境隔离** | 每次评估在独立沙箱中运行，互不干扰 |
| **Prompt 注入** | 支持 `{prompt}` 和 `{prompt_file}` 占位符替换 |
| **Problem 文件** | 自动读取 PROBLEM.md 并追加到 Agent Prompt |
| **Setup/Teardown** | 支持多步骤环境准备和清理命令 |
| **超时控制** | Agent 命令和 Setup 命令分别设置超时 |
| **环境变量** | 通过 `env` 传递密钥等敏感信息 |
| **Per-case 覆盖** | 每个 case 可通过 `params` 覆盖 `environment_dir` |
| **ToolCall 记录** | 自动记录 `environment.setup` 和 `environment.agent` 工具调用 |
| **文件收集** | 执行结束后自动收集输出文件到 CodeArtifact |

### 与 IntegrationGrader 配合

EnvironmentAdapter 和 IntegrationGrader 天然互补：

```
EnvironmentAdapter（运行 Agent）
        │
        ▼
   CodeArtifact（代码 + 执行结果）
        │
        ▼
IntegrationGrader（运行测试脚本评分）
```

这一组合完整复现了 Stripe Agent Benchmark 的评估模式：
1. **EnvironmentAdapter** 负责环境准备和 Agent 执行
2. **IntegrationGrader** 负责运行 `grader/` 中的测试脚本并计算得分

## 自定义 Adapter

```python
from compass.adapters import Adapter, register_adapter

@register_adapter("my_agent")
class MyAgentAdapter(Adapter):
    async def run(self, input: AgentInput) -> AgentOutput:
        # 实现 Agent 调用逻辑
        ...

    async def health_check(self) -> bool:
        # 实现健康检查
        ...
```

## 黑盒 Agent 的 ToolCall 获取机制

> **重要设计约束**：Compass 的过程事件（ToolCall）记录依赖于 Adapter 层的主动上报，而非自动拦截。

### 设计前提

当前架构要求被测 Agent 必须通过 Compass 的 **Adapter** 包装，由 Adapter 负责记录事件：

```
┌─────────────────────────────────────────────────────┐
│                  Compass Runner                      │
│  context={"transcript": transcript}  ← 注入记录器    │
└──────────────────────┬──────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────┐
│              你编写的 Adapter                        │
│  • 调用黑盒 agent                                    │
│  • 从响应中提取信息                                  │
│  • 调用 _record_tool_call() 记录事件                │
└──────────────────────┬──────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────┐
│           黑盒 Agent (外部 API/服务)                 │
│  • 内部 tool call 链路不可见                        │
│  • 只暴露输入/输出                                  │
└─────────────────────────────────────────────────────┘
```

### 三种黑盒场景的处理方式

**场景 1: LLM API 调用（OpenAI/Claude 等）**

可从 API 响应中提取部分信息：

```python
class MyLLMAdapter(Adapter, LLMToolCallMixin):
    async def run(self, input: AgentInput) -> AgentOutput:
        response = await openai.chat.completions.create(...)

        # 自动从响应提取 token 和成本
        self._record_llm_call(
            input,
            provider="openai",
            model="gpt-4o",
            response=response,  # ← 从 response.usage 提取
            duration_ms=elapsed,
        )
```

可获取：✅ Token 使用量、✅ 成本计算、✅ 延迟
不可获取：❌ Agent 内部的 tool call 链（除非 API 返回）

**场景 2: 完全黑盒的外部服务**

只能记录输入/输出级别的信息：

```python
class ExternalAgentAdapter(Adapter):
    async def run(self, input: AgentInput) -> AgentOutput:
        result = await external_api.call(input.prompt)

        # 只能记录"调用了一次外部服务"
        self._record_tool_call(
            input,
            tool_name="external.agent",
            input={"prompt": input.prompt},
            output={"result_summary": "..."},
            status="ok",
            duration_ms=elapsed,
        )
```

**场景 3: 支持事件流的 Agent**

如果外部 agent 支持 streaming 或事件回调，可以逐个记录：

```python
class StreamingAgentAdapter(Adapter):
    async def run(self, input: AgentInput) -> AgentOutput:
        async for event in external_agent.stream(input.prompt):
            if event.type == "tool_call":
                self._record_tool_call(
                    input,
                    tool_name=event.tool_name,
                    input=event.tool_input,
                    output=event.tool_output,
                    ...
                )
```

### 能力边界总结

| 黑盒类型 | 能否获取 tool call | 解决方案 |
|----------|-------------------|----------|
| 自己编写的 agent | ✅ 完全可控 | 直接调用 `_record_tool_call` |
| LLM API (OpenAI/Claude) | ⚠️ 部分 | 从 response 提取 token/cost |
| 外部 API 返回 tool_calls 字段 | ✅ 可以 | 解析响应并记录 |
| 纯黑盒 (无日志/无事件流) | ❌ 不可能 | 只能记录输入输出级别 |

### 设计权衡

**为什么不自动拦截？**

1. **侵入性**：自动拦截需要 monkey-patch SDK 或网络代理，对生产环境不友好
2. **可靠性**：不同 Agent 框架内部实现差异大，难以统一拦截
3. **显式优于隐式**：Adapter 明确声明记录了什么，便于理解和调试

**如果需要真正的黑盒 tool call 追踪**，可能的扩展方向：

- **代理层拦截**：在网络层拦截 HTTP 请求（类似 mitmproxy）
- **SDK Hook**：如果使用 OpenAI SDK，可以 monkey-patch 其方法
- **要求 Agent 暴露事件**：让被测 agent 支持 callback 或 streaming

**当前建议**：对于完全不透明的黑盒 Agent，接受只能记录外部可观测数据（输入、输出、延迟、状态）的限制，将过程评估聚焦于可控的 Agent 实现。
