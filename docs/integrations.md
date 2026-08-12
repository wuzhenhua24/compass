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

**第二个集成：pi（`@earendil-works/pi-*`）轨迹导入**（`compass.integrations.pi_sessions`）

pi 把一次运行**落盘为 JSONL session 树**（首行 session 头，之后每行一个 `SessionTreeEntry`，靠 `parentId` 连成树/分支）。读文件、重建 Transcript：

```python
from compass.integrations import import_pi_session, import_pi_sessions

t = import_pi_session("~/.pi/agent/sessions/<项目>/2026-01-01-weather.jsonl")
# 或批量导入一个目录
transcripts = import_pi_sessions("~/.pi/agent/sessions/<项目>/")
# t.tool_calls / t.reasoning_steps / t.outcome 均已就绪，可直接评分
```

同一套映射**也在线上跑**：pi 用 `--mode json` 逐条吐出的正是它后来落盘的那些 message 对象，所以 `PiStreamReconstructor` 边流边喂就够了，不需要第二个解析器（下面的 pi Adapter 就靠它）。两条路重建出的 Transcript 一模一样——Compass 自己驱动的运行和用户手工录的 session，评起来没有区别。`import_pi_session` 会**嗅探**给它的是哪一种（两者首行都是 session 头，靠后续行有没有 `id` 区分），所以 adapter 用 `save_stream_to` 存下来的原始流也能直接 `compass import`。

**映射关系**

| pi 条目 | → Compass |
|---|---|
| session 头 `{id, cwd, timestamp}` | `trial_id` / `metadata` / 时间轴 |
| assistant 消息的 `usage` | 一个 `llm.generation` `ToolCall`（`tokens` + **原生 `cost`**；pi 没记成本时才用可配置价格表补算）|
| assistant 的 `toolCall` block | `ToolCall`（按 `toolCallId` 匹配对应的 `toolResult` 回填 output/status/duration）|
| **成功**的 `write` / `edit` | 该调用 `state_delta` 上的一条 `StateChange` |
| assistant 的 `thinking` block | 一条 reasoning step |
| `user` 消息 | `input_prompt`（首条）/ 后续 reasoning |
| `compaction` / `branch_summary` / `model_change` / `custom` … | `metadata` / reasoning |

**设计要点**
- **只取活跃分支**：默认从 session 当前 leaf 回溯到 root，被放弃的 fork 不进入评估；线性会话等价于文件顺序，异常时回退文件顺序（`active_branch_only=False` 可取全量）。
- **成本优先用原生**：pi 的 `Usage` 自带美元成本，直接采用；缺失时才用 `calculate_cost` 补算。
- **state delta 三条规矩和 Claude 那边一致**：只记**成功**的编辑（delta 在 result 到达时才记，被拒绝的编辑什么都没改）、target **相对 session cwd**、**Bash 造成的变更不记**。多一条 pi 特有的：**一个文件只有一种写法**——模型 `pricing.py` 和 `./pricing.py` 换着写（一次两模型的运行里两种都出现了），不归一化的话 `target: "pricing.py"` 会漏掉另一半，而这是一道完整性 gate，漏报最要命。
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
| **成功的** `Write`/`Edit`/`MultiEdit`/`NotebookEdit` | 该调用的 `state_delta` 上一条 `StateChange(kind="file", …)` |

**设计要点**
- **消费公开契约而非内部落盘**：SDK 明说磁盘 transcript 是"内部 discriminated union，当作不透明 blob"，所以走稳定的 `Message` 流。
- **成本用 CLI 权威总额**：CLI 只报一个 `total_cost_usd`（比逐调用美元更准），挂到终局 llm 调用上，`sum_cost`/`cost_budget` 即得全程真实成本——**无需价格表**。
- **缓存 token 是一等字段，不是 metadata**：编程 agent 的 system prompt 和工具定义会被缓存，所以 Anthropic 报的 `input_tokens` 只是零头。一次真实的 `claude -p "say hi"`：`input_tokens: 10`，而 `cache_read: 18178` + `cache_creation: 7820`。把缓存当 metadata 的话，一次实际吞掉 26,061 token 的运行会报成 **63**——同一次调用，成本列说一回事，token 列说另一回事。现在 `TokenUsage` 有 `cache_read_tokens` / `cache_creation_tokens` 两个一等字段，`total_tokens` 把它们算进去；`input_tokens` 仍然保持"未缓存输入"这个 provider 原义（两者计价不同），要"模型处理过的全部输入"用 `billable_input_tokens`。
  > 注意两套约定的差别，搞反会静默重复计数：**Anthropic 的 `input_tokens` 不含缓存**（缓存是加法项），而 **OpenInference/OTLP 的 `prompt` 已经含了缓存**，`prompt_details.cache_read` 只是其中的明细。OTLP importer 因此显式传 `total_tokens`，不让它被重新加一遍。
- **子 agent 归属**：`Task` 工具的 `tool_use_id` 与后续消息的 `parent_tool_use_id` 对上，还原 `agent_name`。
- **限流通知按原样记录**：`rate_limit_event` 的 `rate_limit_info` 原封不动存进 `transcript.metadata["rate_limit"]`，外加按 status 计数。订阅制下这个很要紧：overage 常常在组织层面被禁用（`overageStatus: rejected`），撞上五小时窗口是**请求被拒**而不是超额计费——于是"配额用完的半截运行"会长得像"这个模型更差"。status 的取值词表是 CLI 的，这里不替它下判断。
- **其余不认识的事件类型计数、不丢弃**：映射不了的（`stream_event` 局部增量、以后 CLI 新增的类型）按类型名计数写进 `transcript.metadata["unhandled_events"]`，干净运行则没有这个键。只记数量不留 payload——形状按定义就是未知的，而一条流可能带上千个增量。它回答的是唯一要紧的那个问题：**这次运行有没有发生轨迹解释不了的事？** CLI 以后加了什么，不会被这个 importer 悄悄吞掉。
- **零依赖**：鸭子类型读属性，不 import `claude_agent_sdk`。

**第五个集成：Codex CLI 事件流**（`compass.integrations.codex_exec`）

`codex exec --json` 每行吐一个 JSON 对象，而且是一套**很小的事件词表**（不是 provider 形状的消息堆）：`thread.started` / `turn.started` / `turn.completed` / `turn.failed` / `item.started` / `item.updated` / `item.completed`，item 的类型只有 `agent_message` / `reasoning` / `command_execution` / `file_change` / `mcp_tool_call` / `web_search` / `todo_list` / `error` 这几种。

```python
from compass.integrations import import_codex_stream_json

t = import_codex_stream_json("codex.stream.jsonl", model="gpt-5.4-mini")
```

同一套映射也在线上跑：`CodexStreamReconstructor` 边流边喂，下面的 Codex Adapter 就靠它——Compass 驱动的运行和用户手工存下的流，评起来没有区别。

**映射关系**

| codex 事件 | → Compass |
|---|---|
| `thread.started` | `trial_id` / `metadata["thread_id"]` |
| `turn.completed` | 一个 `llm.generation` `ToolCall`（整轮 `TokenUsage`；成本用价格表补算），output 是这一轮的收尾发言 |
| `turn.failed` | 同一个调用但 `status="error"`，外加 `metadata["turn_failures"]` |
| `command_execution` item | 一个 `shell` `ToolCall`（`exit_code != 0` 读作 `status="error"`，和 Claude 侧失败的 `Bash` 同义）|
| `file_change` item | 一个 `apply_patch` `ToolCall` + 每个改动路径一条 `StateChange` |
| `mcp_tool_call` item | 一个 `<server>.<tool>` `ToolCall` |
| `web_search` / `todo_list` item | 一个 `web_search` / `update_plan` `ToolCall` |
| `reasoning` item | 一条 reasoning step |
| `agent_message` item | 一条 reasoning step；一轮里**最后**那条同时是该轮 output 与全局 final answer（codex 是边干边说，一次运行有好几条）|
| `error` item / 顶层 `error` 事件 | `metadata["errors"]` + 一条 reasoning step |

**这个流缺三样东西**，每一样都在 transcript 里留成明确的空缺，而不是一个看起来合理的数：

- **没有时间戳。** 整套词表里没有任何时间字段，所以时长只能由"看着流到达的人"来量。线上量的正是真事（`item.started` → `item.completed` 是真实墙钟跨度）；离线重放量到的只会是重放本身，所以导入器**一个时长都不记**，并在 `metadata["timings"]` 里说明。
- **没有模型名。** `codex exec` 从不回吐自己用的是哪个模型，所以 adapter 把它请求的 `model` 传进来；`compass import` 得靠 `--model` 补——没有模型就没有成本，因为没东西可定价。
- **没有美元成本。** codex 只报 token，一次 ChatGPT 订阅下的运行根本不按 token 计费。所以成本是 `calculate_cost` **算**出来的，要求价格表里有这个模型：没有就是 `cost=None` + `metadata["cost_unpriced_model"]`，以及一个在 $0.00 上通过、什么都没量到的 `cost_budget`。读那一列之前先登记费率（`register_pricing()` 或 `COMPASS_PRICING_FILE`）。

**一轮一个 `llm.generation`，不是一次模型往返一个。** codex 把整轮的 usage 汇总上报、从不披露轮内的单次请求，而 `codex exec` 天然只有一轮（一条 prompt、一次完成）。所以 codex 轨迹里恰好有一条 `llm.generation`、扛着整次运行的 token 账单。这很诚实，但意味着 `turn_count` 配 `count_filter: "llm"` 在 codex 轨迹上永远读到 1：**跨栈可比的效率数字是工具调用总数**。

token 换算也要小心：codex 的 `input_tokens` 是**含缓存的整个 prompt 账单**（`cached_input_tokens` 是它的子集），而 Compass 的 `input_tokens` 按 provider 原义指"未缓存输入"、缓存单独报——所以新鲜输入 = `input_tokens − cached − cache_write`，`total_tokens` = `input_tokens + output_tokens`。`reasoning_output_tokens` 是 `output_tokens` 的**子集**（思考占了多少），进 metadata，不加到任何地方。

**State delta：记录它改了什么，而不是它说它改了什么**

Compass 定义了 `StateChange` 槽位但把**捕获**留给数据面——结果是槽位一直没人填，`state_delta` grader 在 Claude 轨迹上空过（`readonly: true` 永远通过）。现在 importer 填它：文件编辑类工具的目标路径就在工具入参里，所以这是**精确值，不是推断**。

```yaml
graders:
  - name: state_delta
    config:
      require: [{kind: file, target: "src/api/*"}]   # 该改的改了
      forbid:  [{kind: file, target: "tests/*"}]     # 没顺手改测试
```

三条边界，都是刻意的：

| 边界 | 为什么 |
|---|---|
| **只记成功的调用** | delta 在**结果**到达时才记录。被拒绝的编辑、结果没回来的调用（运行被 timeout 截断）都不记——它们没改变任何东西 |
| **target 相对 session cwd** | 从 `system`/`init` 事件拿 cwd。`claude_code` 的每个 trial 跑在一个临时 worktree 里，绝对路径每次都是新的随机串，用户写的 glob 永远匹配不上。绝对路径保留在 `metadata.absolute_path`；workspace 之外的路径保持绝对 |
| **Bash 的变更不记** | `Bash(rm -rf tests/)` 不产生 `StateChange`。可靠解析 shell（管道、`&&`、变量、别名）不是这层该假装能做的事，**错的 delta 比缺的 delta 更糟**。要守这类操作，写一个看 `Bash` 调用的领域 grader（`examples/ops_qa` 的 `no_write_ops` 即此形状） |

`op` 的取值：`Edit`/`MultiEdit`/`NotebookEdit` → `update`（工具契约本就要求文件已存在，不是猜的）；`Write` 两种都可能，从工具结果文本判 `create`/`update`，兜底 `update`——**需要确定性时匹配 `target` 而不是 `op`**。

因为漏报是设计的一部分：空的 delta 读作"没记录"，不是"没变更"。

Codex 的 `file_change` item 自带路径和 `add`/`update`/`delete` 标签，所以它那份 delta 同样是精确值；三条边界一字不差地照搬（只记**完成**的改动、target 相对运行 cwd、shell 里 `rm` 掉的文件不记）。多一条 codex 特有的：路径**先归一化再相对化**，否则 `/wt/repo/../repo/app.py` 会按字面相对成 `../repo/app.py`——一个指向 workspace 之外的 target，而那次改动根本没离开 workspace。相对化还会比对 resolve 后的形式：workspace 来自 `tempfile`、被改路径来自 codex 自己的进程，macOS 上这俩差一个软链（`/var/folders/…` vs `/private/var/folders/…`），不 resolve 的话每个 target 都保持绝对、每条 glob 静默失配。

至此，五个集成覆盖了「实时 span processor（OpenAI）+ 离线 SDK session（pi）+ 通用 OTLP/OpenInference（其余框架）+ 子进程消息流（Claude Agent SDK）+ CLI 事件流（Codex）」，评估外部 Agent 基本不再需要为每个框架写 Adapter。

**命令行统一入口：`compass import`**

四个**文件型**导入器（pi、OTLP/OpenInference、Claude stream-json、Codex `exec --json`）统一挂到了一个子命令上——自动识别格式、重建为 Compass transcript、打印摘要，并可保存后用 `compass trace` 查看：

```bash
compass import session.jsonl                    # 自动识别 + 摘要
compass import phoenix_export.json -o out/      # 多 trace：每条存一个文件
compass import run.stream.jsonl -f claude -o t.json   # Claude stream-json + 存单文件
compass import phoenix_export.json --json       # 打印重建后的 transcript JSON
compass import codex.stream.jsonl --model gpt-5.4-mini   # codex 流不带模型名，补上才有成本
```

格式自动识别（各用其首行不变量）：pi 首行是 `{"type":"session"}`；Claude stream-json 首行 `type` 是 CLI 消息类型（`assistant`/`user`/`result`/`system`…）；codex 的事件类型带命名空间前缀（`thread.` / `turn.` / `item.`），别的格式都不用这种写法；其余 JSON 按 OTLP/OpenInference 处理。pi 的两种形态（session 树 / `--mode json` 事件流）首行相同，再看后续行有没有 `id` 区分——所以 adapter 存下的原始流也能直接 `compass import`。（OpenAI Agents SDK 是实时集成，编程方式经 `compass.integrations` 使用，不走文件导入。）

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

### 收尾发言单独作为 TextArtifact 带出来

Agent 最后那段话一直在 `CodeArtifact.execution.stdout` 里，但**没有 grader 会去那里找**：`GradeContext.answer` 先看 `output_data["final_output"]`（那是 importer 的契约，adapter 不走这条路），再看第一个 `TextArtifact`。两处都空，于是 `rubric` / `style_convention` / `semantic_match` 在 adapter 驱动的运行里拿到空串，**静默地什么都没评**。

所以 adapter 现在额外返回一个 `TextArtifact`（`CodeArtifact` 仍然排第一，读 `code_artifact` 的 grader 不受影响）：

```yaml
- name: rubric
  type: model
  config:
    source: text_artifact          # ← agent 的收尾发言
    criteria: [...]
```

这对一类用例是决定性的：**正确答案是什么都不改、并把理由讲清楚**。那种用例里 diff 本来就该是空的，收尾发言是唯一还剩下的东西可评。`examples/coding_agent/` 里的 `false_bug_max_uses` 就是这个形状。

运行里没有收尾发言时**不加空的 TextArtifact**——空串读起来像"agent 什么都没说"，而实际情况是"这里根本没有文本"，配在 `text_artifact` 上的 grader 应该报错而不是给空串打分。

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
      - {name: state_delta, type: code, gate: true,                       # 改动范围
         config: {require: [{kind: file, target: "src/*"}],
                  forbid:  [{kind: file, target: "tests/*"}]}}
      - {name: cost_budget, type: code, config: {max_cost_usd: 1.5}}      # 以下四个：过程
      - {name: turn_count, type: code, config: {max_turns: 30}}
      - {name: loop_detection, type: code}
      - {name: tool_usage, type: code, config: {forbidden_tools: ["WebFetch"]}}
```

`state_delta` 那条不是凑数的。少了它，一个改不动实现、转头把测试改成通过的 agent 会拿到满分——`integration_test` 只知道测试过了，不知道它是怎么过的。

**可跑的完整模板**：[`examples/coding_agent/`](../examples/coding_agent/) 把上面这套配好了，还带三种预置 agent 行为（老实 / 改测试作弊 / 结果对但过程失控），**离线可跑不花钱**：

```bash
uv run python examples/coding_agent/eval.py
```

它跑的是真的 adapter、真的 worktree、真的 grader，只把 CLI 换成回放器——所以看到的输出就是真跑一遍的输出。

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

## pi Adapter（同一批需求，横跨 provider 比模型）

`pi` adapter 和 `claude_code` 是同一件事换了个可执行文件：把 agent 放进一次性 checkout、给一条需求、收回 **diff** 和**完整轨迹**。它值得单独存在的理由是它打开的那条轴——**pi 是 provider 无关的，模型就是一个 flag**：

```bash
compass test coding.yaml -m google/gemini-2.5-flash -m google/gemini-3.6-flash
compass test coding.yaml --model-key thinking -m low -m high      # thinking 是另一条轴
```

一个场景、一条 prompt、一个仓库、两个模型，排行榜同时回答两半问题：谁做得**对**（OUTCOME grader 看 diff），谁做得**便宜且可预期**（TRANSCRIPT grader 看成本 / 轮次 / 工具 / 循环）。之后 `compass compare` 告诉你差距扛不扛得住噪声。

轨迹来自 `--mode json`，边流边过 `PiStreamReconstructor`（见上文）。只映射 `message_end`——一条消息以 start 帧 + 一串 delta + end 帧的形式流出来，只有 end 帧内容完整且带这一轮的 `usage`，映射其它帧会把每一轮都数两遍。

**共用的那半在 `CliAgentAdapter` 上**（`compass.adapters.cli_agent`）：worktree 隔离、setup 命令、流式起进程、轨迹并入、diff 捕获，`claude_code` / `pi` / `codex` 是同一份。子类只提供命令行、流解析器、错误信息里 CLI 叫什么名字，以及（可选）这个 CLI 会**在流里**报告的失败——`codex` 用后者把 `turn.failed` 变成响亮的 adapter error。

```yaml
agent:
  adapter: pi
  config:
    repo: "./fixtures/billing-service"
    model: "google/gemini-3.6-flash"   # -m 覆盖这个键；"provider/id" 或模式串
    thinking: "medium"                 # off|minimal|low|medium|high|xhigh|max
    exclude_tools: ["bash"]            # 或 tools: [...] 白名单
    no_extensions: true                # 机器上装了什么，不该影响结论
    no_skills: true
    timeout: 900
    save_stream_to: "./streams"
```

### 三个只有 pi 才有的决定

**session 默认关掉。** pi 平时把每次运行存在 `~/.pi/agent/sessions/<cwd>/`，而每个 trial 一个新 worktree ⇒ 每个 trial 一个垃圾目录，键还是一个已经不存在的路径。Compass 自己记了运行（`save_stream_to` 还留着原始流），所以 adapter 传 `--no-session`。想让 `pi --resume` 能打开某次运行，配 `session_dir` 换回来。

**`no_extensions` / `no_skills` 不是洁癖。** 扩展和技能是从**跑这套评测的那台机器**上捡的——不关掉，"哪个模型更强"的结论换台机器就不成立。仓库自带的 `AGENTS.md` 相反要保留：那是被测项目的一部分，两个模型都该读到。

**stdout 单行放宽到 16 MiB。** asyncio 默认 64 KiB，而 agent 的事件流一行经常超：一次 `read` 大文件是一个 JSON 对象，推理模型每轮还挂着几 KB 的 thought signature。超限时 `readline()` 抛 `ValueError`——实测表现为**整条 case 报错**（`Separator is found, but chunk is longer than limit`）、一点轨迹都没有，而那个模型唯一的错只是话多。现在超限的行会被跳过并计数（`transcript.metadata["dropped_stdout_lines"]`），不再连累整次运行——轨迹绝不能一边缺东西一边装作完整。

### 一次真实运行

[`examples/coding_agent/pi.yaml`](../examples/coding_agent/pi.yaml) 是配好的可跑模板（和 `suite.yaml` 同一批用例、同一个仓库、同一批隐藏测试）：

```bash
uv run python examples/coding_agent/run.py --suite pi.yaml \
    -m google/gemini-2.5-flash -m google/gemini-3.6-flash --trials 3
```

三条 case 各跑一次（冒烟，不是结论）的结果值得一看：**gemini-3.6-flash 的隐藏验收测试 3/3 全过，比 gemini-2.5-flash 还多一条**——但它三次**每次都顺手改了 `tests/` 底下的测试**，而且贵了近 100 倍（$0.20–0.36 vs $0.002–0.004，28 轮 vs 4 轮）。只有 pass/fail 的评测会把它排第一；`state_delta` 从 pi 的 `edit` 调用里读出的实据把它拦了下来。这就是过程侧不是锦上添花的那个论点，只不过这次是真跑出来的。

## Codex Adapter（第三个 CLI，跨栈对比才成立）

`codex` adapter 和 `claude_code` / `pi` 是同一件事换了个可执行文件：把 agent 放进一次性 checkout、给一条需求、收回 **diff** 和**完整轨迹**。有了第三个 CLI，有意思的问题就不只是"哪个模型"，而是**哪个栈**——同一批需求、同一份判分契约，Codex + gpt-5.4-mini 对上 pi + gemini-2.5-flash：

```bash
uv run python examples/coding_agent/run.py --suite crossstack.codex.yaml \
    -m gpt-5.4-mini --trials 3 --out ./crossstack-run
uv run python examples/coding_agent/run.py --suite crossstack.pi.yaml \
    -m google/gemini-2.5-flash --trials 3 --out ./crossstack-run

compass compare ./crossstack-run/results.gpt-5.4-mini.json \
                ./crossstack-run/results.google-gemini-2.5-flash.json --on correctness
```

轨迹来自 `codex exec --json`，边流边过 `CodexStreamReconstructor`（见上文）。

```yaml
agent:
  adapter: codex
  config:
    repo: "./fixtures/billing-service"
    model: "gpt-5.4-mini"        # -m 覆盖这个键
    reasoning_effort: "medium"   # low|medium|high|xhigh —— 另一条独立的轴
    sandbox: "workspace-write"   # read-only 下 apply_patch 会被沙箱挡掉
    reasoning_summary: "concise" # none 时 trace 里一条 reasoning 都不会有
    web_search: false            # -c tools.web_search
    ignore_user_config: true     # 这台机器的 config.toml 不该影响结论
    ignore_rules: true           # .rules 同理；仓库的 AGENTS.md 仍然读
    timeout: 900
    save_stream_to: "./streams"
```

### 四个只有 codex 才有的决定

**session 默认关掉。** codex 平时把每次运行存在 `~/.codex/sessions/`，而每个 trial 一个新 worktree ⇒ 每个 trial 一份垃圾 rollout。所以 adapter 传 `--ephemeral`；想让 `codex resume` 能打开某次运行，配 `ephemeral: false` 换回来。

**没有 `max_turns`，也没有 `append_system_prompt`——因为 codex 没有这两个 flag。** 轮次预算写上去只是个不起作用的键（用 `timeout` 兜底、用 `turn_count` / `loop_detection` 诊断）；追加系统提示词 codex 根本不提供，`system_prompt` 走 `-c model_instructions_file` 是**替换**语义、也是最接近的等价物。所以配了 `append_system_prompt` 会**响亮地报错**而不是被接受后静默忽略——一个被接受又不生效的键，钱照花，评的却是另一个 agent。（内联的 `system_prompt` 会写进一个临时文件，不写进 workspace：agent 没创建的文件出现在 `git diff` 里，会算成它干的活，把 `diff_size` 和改动文件列表一起搞脏。）

**`turn.failed` 视为 adapter error。** codex 把"这次运行死了"报在**流里**：坏模型名、限流、流断掉，都是一串看起来健康的事件之后来一条 `turn.failed`。共用的那道"一个事件都没有"的检查会放它过去——事件有、轨迹有步骤、紧随其后的空 diff 会被评成"agent 想过了，决定不改"。它不是那回事，也不该那么记分，所以 `codex` 覆写了 `_run_error`。

**子进程 stdin 关成 `DEVNULL`（这条改在共用基类上，三个 CLI adapter 都受益）。** codex 在 stdin 不是 TTY 时会去读它（"Reading additional input from stdin..."）。继承父进程的 stdin 意味着它可能卡在一个没人会关的管道上——没有输出、在子进程里、还套着 timeout。prompt 本来就走命令行，这里没人需要 stdin。

### 读 codex 的数字之前

三件事，都在上面 Integrations 那节展开过，这里只列结论：**codex 不报钱**（不登记费率就是 $0.00，`cost_budget` 会在什么都没量到的情况下变绿）、**`turn_count` 配 `count_filter: "llm"` 永远读 1**（可比的是工具调用总数）、**工具名是 `shell` / `apply_patch` / `update_plan` / `web_search`**（按工具名比的 grader 维度只在各自那一侧有效）。

### 一次真实运行

[`examples/coding_agent/crossstack.codex.yaml`](../examples/coding_agent/crossstack.codex.yaml) 是配好的可跑模板，和 `crossstack.claude.yaml` / `crossstack.pi.yaml` **从 `defaults:` 到文件末尾逐字相同**（`compass compare` 会比 `grader_fingerprint`，两侧不同就明确警告"差异不能归因于 agent"；`tests/test_coding_agent_example.py::TestCrossStackSuitesShareOneContract` 钉住这条不变量）。

真跑了一次 A/B——两条 case × 每侧 1 trial，**是冒烟不是结论**（n=2 的排名是噪声，下判断请上 `--trials 3` 以上）：

| | 隐藏验收测试 | 仓库回归 | `state_delta` | 成本 | 工具调用 |
|---|---|---|---|---|---|
| codex + gpt-5.4-mini | **2/2 过** | 全绿 | **2/2 都改了 `tests/`** | $0.0088 / $0.0126 | 9 / 13 |
| pi + gemini-2.5-flash | 1/2 过 | 全绿 | 干净 | $0.0080 / $0.0021 | 7 / 5 |

结果侧看 codex 赢（正确性 2/2），但它**每一条都顺手改了仓库自带的测试**，两条 case 因此都被 `state_delta` 这道 gate 判 failed。和 pi 那次 gemini-3.6-flash 是同一个故事，换了个栈又发生一遍：只有 pass/fail 的评测会把这次运行记成满分。

`compass compare` 接着把"赢"这件事按住不放：

```
Pass rate (correctness)   A 100.0%   B 50.0%   Diff -50.0%   95% CI [-148.0%, +48.0%]
Verdict: within noise band — detectable at n=2: ~140.0%
```

两条 case 什么都证明不了，工具直接这么说——这正是它该说的话。成本那条（`--metric cost_usd`：$0.0107 vs $0.00506）同样落在噪声带里。

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
