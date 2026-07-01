# ToolCall Protocol v1

> 目标：统一不同类型 Agent 的工具调用记录格式，便于跨 Agent 的过程评估、成本分析、合规审计与问题诊断。

## 1. 设计原则

- **通用性**：适配 LLM、检索、浏览器、数据库、代码执行、多模态等工具。
- **可观测**：支持链路追踪、时间/成本/错误分析。
- **安全性**：原始输入/输出可脱敏，避免泄露敏感信息。
- **可扩展**：核心字段稳定，扩展字段通过 `metadata` 承载。

## 2. Schema 规范（JSON）

### 2.1 必填字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `tool_name` | string | 标准化工具名（建议注册表维护） |
| `input` | object | 调用参数（原样保留，必要时脱敏） |
| `status` | string | 调用状态：`ok` / `error` / `blocked` |
| `duration_ms` | number | 调用耗时（毫秒） |
| `timestamp` | number | Unix epoch 秒（浮点） |

### 2.2 可选字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `output` | object \| string \| null | 调用结果（可截断/脱敏） |
| `error` | object | 错误信息：`{type, message, stack?}` |
| `cost` | object | 成本：`{total_usd, breakdown?}` |
| `tokens` | object | Token 用量：`{input, output, total}` |
| `tool_type` | string | 工具类型：`llm` / `retriever` / `browser` / `db` / `code` / `image` / `custom` |
| `retry_count` | number | 重试次数 |
| `trace` | object | 链路信息：`{span_id, parent_id}` |
| `metadata` | object | 扩展字段 |
| `redacted` | boolean | 是否已脱敏 |

## 3. 示例

```json
{
  "tool_name": "search_web",
  "tool_type": "retriever",
  "input": {"query": "agent eval framework"},
  "output": {"results": 10},
  "status": "ok",
  "duration_ms": 842,
  "timestamp": 1738051200.25,
  "cost": {"total_usd": 0.0021},
  "tokens": {"input": 120, "output": 480, "total": 600},
  "retry_count": 0,
  "metadata": {"provider": "serpapi"},
  "redacted": true
}
```

## 4. 标准化建议

- **tool_name 命名**：建议 `namespace.action` 形式，例如 `browser.open`、`db.query`、`llm.chat`。
- **status 规范**：
  - `ok`：正常返回。
  - `error`：执行失败（含异常）。
  - `blocked`：因安全/合规策略拒绝执行。
- **input/output 脱敏**：建议在 recorder 层进行字段级脱敏（如 key、cookie、PII）。
- **tokens 归一**：若工具是模型，建议统一输出 token 统计。
- **cost 归一**：建议统一为美元（USD），保留 breakdown 以支持多工具计费。

## 5. 与评估的关系

- **成本/延迟评估**：`cost` + `duration_ms` 支持预算与效率评分。
- **工具合规审计**：`tool_name` + `input` + `status` 可用于“必需/禁止工具”规则。
- **错误与重试分析**：`error` + `retry_count` 支持稳定性评分。
- **链路诊断**：`trace` 与 Span 对齐，支持跨工具的因果定位。

## 6. 兼容策略

- 旧字段可映射到新协议：
  - `ToolCall.tool` -> `tool_name`
  - `ToolCall.args` -> `input`
  - `ToolCall.result` -> `output`
  - `ToolCall.duration_ms` -> `duration_ms`
  - `ToolCall.error` -> `error.message`
- 未覆盖字段保持默认或置空，确保向后兼容。

