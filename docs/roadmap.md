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
- [ ] Transcript 完整记录与持久化
- [ ] HTML 测试报告增强

### Phase 3 - 增强
- [x] EnvironmentAdapter（Environment-as-Code，参考 Stripe Agent Benchmark）
- [x] 分类聚合分析（Category & Tags，参考 Stripe Backend/Fullstack/Gym 分类）
- [x] TurnCountGrader（交互轮次评分，参考 Stripe turn count 指标）
- [x] Best-of-k 评分展示（参考 Stripe best-of-3 评估模式）
- [x] 答案泄漏检测（Leak Detection，参考 Stripe UUID 嵌入方案）
- [ ] 更多 Adapter (SD WebUI, DALL-E)
- [ ] Human Grader 接口完善
- [x] 环境隔离 (Sandbox)
- [ ] 并行执行优化

### Phase 4 - 完善
- [ ] Web Dashboard
- [ ] CI/CD 集成
- [ ] 基准测试对比
- [ ] 模型校准工具
