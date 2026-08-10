# 路线图（历史归档）

> Compass 专题文档 · 返回 [README](../README.md)


### Phase 1 - MVP ✅
- [x] 项目骨架搭建
- [x] CLI 基础命令
- [x] YAML 场景加载
- [x] 基础 Grader 实现
- [x] Transcript / Outcome 分离设计
  - [x] GraderScope 枚举（OUTCOME / TRANSCRIPT / BOTH）
  - [x] GradeContext 统一上下文（独立访问 Transcript 和 Outcome）
  - [x] validate_context 自动校验数据完整性
  - [x] Outcome 评分器：image_assertions, technical_quality, semantic_match, vlm_judge, aesthetic_score, safety_check, human_review
  - [x] Transcript 评分器：tool_usage, cost_budget, latency_budget
  - [x] 联合评分器：efficiency（BOTH scope）

- [x] 评估结果分析与可视化
  - [x] EvalResultAnalyzer：按 Transcript/Outcome 分维度分析
  - [x] 失败模式识别（自动聚合常见失败组合）
  - [x] Scope 对比诊断（Transcript vs Outcome 分数差距分析）
  - [x] 改进建议生成（瓶颈检测、波动预警、优先级排序）
  - [x] ConsoleReporter：Rich 终端可视化（表格、进度条、色彩编码）
  - [x] CLI `compass analyze` 命令（支持 JSON 输入/输出）

### Phase 2 - 核心功能 ✅
- [x] 多次 Trial 支持
  - [x] TrialManager 多次试验管理
  - [x] TrialResult / TaskResult 数据结构
  - [x] Runner 集成（单次/多次试验分支）
- [x] pass@k / pass^k 指标
  - [x] TrialMetrics 计算引擎
  - [x] pass_rate / score_mean / score_std
- [x] 正向/负向测试
  - [x] expect=pass/fail 机制
  - [x] required graders 必需评分器
  - [x] 两层 required 机制（场景级 + 用例级）
- [x] 结构化输出校验 Graders
  - [x] JsonSchemaGrader（JSON Schema 校验）
  - [x] SqlSyntaxGrader（SQL 语法 + 安全检查）
  - [x] StructureCheckGrader（JSON/YAML/XML/TOML）
- [x] Data Agent 评估 Graders（参考 OpenAI Kepler）
  - [x] SqlEquivalenceGrader（结果等价性比较）
  - [x] DataCorrectnessGrader（数据正确性校验）
  - [x] QueryQualityGrader（SQL 反模式检测）
  - [x] ReasoningTraceGrader（推理过程评估）
  - [x] SelfCorrectionGrader（自我纠错能力）
- [x] Coding Agent 评估 Graders
  - [x] ExitCodeGrader / TestRunnerGrader（功能测试）
  - [x] IntegrationGrader（集成测试评分，参考 Stripe Agent Benchmark）
  - [x] LintGrader / TypeCheckGrader（代码质量）
  - [x] DiffAccuracyGrader / DiffSizeGrader（代码差异）
  - [x] SecurityScanGrader（安全扫描）
- [x] Code Graders 目录重构（按 Agent 类型组织）
  - [x] common/（通用评分器）
  - [x] coding/（Coding Agent）
  - [x] data/（Data Agent）
  - [x] image/（Image Agent）
- [x] Transcript 完整记录与持久化
  - [x] `Transcript.save/load` 双格式（JSON 全量结构 + JSONL 事件流）、TranscriptRecorder
  - [x] `compass test --trace-dir/--trace-format`、`compass trace` 查看
- [x] HTML 测试报告增强
  - [x] HTMLReporter：Transcript/Outcome 双轴散点、分类分解、场景卡片
  - [x] 取数与渲染分离（`render_document`），同一份文档供 HTML 与静态站复用

### Phase 3 - 增强
- [x] EnvironmentAdapter（Environment-as-Code，参考 Stripe Agent Benchmark）
- [x] 分类聚合分析（Category & Tags，参考 Stripe Backend/Fullstack/Gym 分类）
- [x] TurnCountGrader（交互轮次评分，参考 Stripe turn count 指标）
- [x] Best-of-k 评分展示（参考 Stripe best-of-3 评估模式）
- [x] 答案泄漏检测（Leak Detection，参考 Stripe UUID 嵌入方案）
- [ ] 更多 Adapter (SD WebUI, DALL-E)
  - 已降优先级：自跑的 Agent 走**导入轨迹**（Integrations）比被外部驱动更合适，
    投入转向了 pi / OTLP / Claude / OpenAI Agents SDK 四种轨迹接入
- [x] Human Grader 接口完善
  - [x] 评分者一致性：Cohen's κ / Krippendorff's α（`graders/human/agreement.py`）
  - [x] 锚点校准会话（`graders/human/calibration.py`）、`pairwise_comparison` 成对比较
- [x] 环境隔离 (Sandbox)
- [x] 并行执行优化
  - [x] worker 上限由信号量约束；多试验用例走 round-major 调度
  - [x] 断点续跑按 trial 粒度补齐（top-up 而非重跑），中断后样本仍均衡

### Phase 4 - 完善
- [ ] Web Dashboard
  - 静态站那一半已交付：`compass site build/serve`（多仓库可 build 进同一目录，
    索引累积、带趋势线）；常驻服务/数据库形态未做
- [ ] CI/CD 集成
  - 门禁原语已有：`compass test` / `compass grade` 按全通过与否返回 0/1 退出码；
    尚无现成的 Action / 流水线模板
- [x] 基准测试对比
  - [x] `compass compare` 配对比较（case 翻转 + 置信区间 + MDE）
  - [x] `compass test -m` 多模型排行榜（每行标准误 + top 2 配对检验）
  - [x] `compass baseline set/compare/list` 产物级回归基线
- [ ] 模型校准工具
  - 已有测量原语（任意两个评分者之间的 κ / α），但尚未做成"LLM 判官 vs 人工标注"的校准工具

### Phase 5 - 基座化（原路线图之后交付）

把 Compass 从"一个评测工具"推到"评测基座"的那批工作，按主题归档：

- **轨迹接入**：OpenAI Agents SDK / pi JSONL / OTLP·OpenInference / Claude Agent SDK 四种轨迹归一成
  Transcript；`compass import` 自动识别格式（见 [integrations.md](integrations.md)）
- **执行与评分解耦**：`compass grade` 给已落盘轨迹打分，判分器内容指纹自动标记过期评分，
  多套 grader 可并排评同一批轨迹（见 [analysis.md](analysis.md)）
- **ToolCall 协议演进**：turn_index / agent_name 升为一等字段（多 Agent），
  `state_delta` 槽位 + 同名守卫 grader，run_id / config_hash 审计溯源（协议 2.0）
- **评测卫生**：harness 失败移出 pass rate 分母；`score=None` 表示"未测量"而非 0 分；
  原子写入 + 防御性读取
- **观察维度**：中性标签（tags）与定量指标（metrics）双轨聚合，LLM 判官支持受控词表
- **扩展面**：`external_checker` 让任何可执行文件成为 grader；grader 列表变成共享 workspace 的
  流水线（`creates:` / `required:` 中止）
- **过程侧判官**：`trajectory_judge`（调用链是否合理）、`groundedness`（答案是否被工具观察支撑）
- **结果分发**：`compass site build/serve` 静态站与趋势线；`compass docs` 在终端读框架自身文档
- **文档结构**：README 收敛为骨架，细节拆进 docs/ 专题文档 + 单页 cheatsheet
