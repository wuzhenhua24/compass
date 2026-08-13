# Skill Eval — 证明 skill 的新版本真的更好了

改完一个 skill，要发版，怎么证明它是**优化**而不是**改坏了**？

这是 Compass 面向 Agent Skill 的完整模板：装哪个版本的 skill 是一个可扫的轴、
触发与否是一个可比的指标、v1→v2 的差是一个带置信区间的判断。**离线可跑，不需要
API key，不花钱。**

```bash
uv run python examples/skill_eval/eval.py
```

## 它输出什么

```
  baseline  — 不装 skill —      触发    0%   脚本    0%   $0.052/case   pass    0%
  v1        report-writer-v1   触发   33%   脚本    0%   $0.055/case   pass   33%
  v2        report-writer-v2   触发  100%   脚本  100%   $0.049/case   pass  100%

  按 `triggered` 判分：significant pass-rate improvement: +66.7% (95% CI [+1.3%, +132.0%])
      ↑ contextual_invocation
      ↑ implicit_invocation

  触发率（skill_triggered，越高越好）
      v1 0.3333 → v2 1
      skill_triggered: B is significantly higher — +0.6667 (95% CI [+0.01333, +1.32])
  bundled 脚本使用（skill_resources_used，越高越好）
      v1 0 → v2 1
  工具调用数（tool_calls，越低越好）
      v1 9.667 → v2 7
      tool_calls: B is significantly lower — -2.667 (95% CI [-3.973, -1.36])
```

再跑一遍触发率那套，会看到另一半：

```
  v2   pos_implicit         该触发      实测触发  100%
       pos_contextual       该触发      实测触发  100%
       neg_chart            不该触发     实测触发  100%   ← 不符合预期

  按 `triggered` 判分：within noise band: pass-rate diff +16.7%
      (95% CI [-43.6%, +76.9%]); detectable at n=6: ~86.0%
      ↑ pos_contextual
      ↑ pos_implicit
      ↓ neg_chart   ← 回退，即使总分涨了也要看这一行
```

**这才是这个例子的重点。** v2 的 description 写得更"主动"，正向召回从 17% 提到
67%——同时开始抢画图的活。只有正向用例的评测看不见这一行；一个只报均值的
benchmark 也看不见"6 条用例根本不够判定总体是否变好"（`detectable at n=6: ~86%`）。

## 三个可比的量

| 问的问题 | 谁回答 | 落到哪个指标 |
|---|---|---|
| skill 到底加载了没有 | `skill_trigger` | `skill_triggered`（多次 trial 求均值 = 触发率） |
| bundle 的脚本真的被用了吗 | `skill_trigger` + `required_resources` | `skill_resources_used` |
| 这一趟花了多少 | `cost_budget` / `turn_count` / `tool_usage` | `cost_usd` / `turns` / `tool_calls` |
| 差值是真的还是噪声 | `compass compare` | 配对差 + 95% CI + MDE |

**这里没有 correctness grader，是故意的。** 报告写得好不好是领域判断，属于用户
代码（像 pytest 里的 `tests/`）；Compass 管的是可复用的那半边。要加，就在 case
上挂自己的 grader——参见 `examples/coding_agent`，那边的 `integration_test` 就是
一个真正的正确性 gate。

## 三条支线是怎么切出来的

装哪个版本的 skill，就是 `agent.config.skill` 这个键：

```bash
compass test ab.yaml --model-key skill \
    -m "" \
    -m skills/report-writer-v1 \
    -m skills/report-writer-v2 \
    --report json -o out/results.json
```

`-m ""` 是不装 skill 的基线支线。adapter 会把指定目录装到 workspace 的
`.claude/skills/<skill 自己的 name>/`——注意是**frontmatter 里的 name**，不是目录
名，所以 `report-writer-v1` 和 `report-writer-v2` 到 agent 眼里都叫
`report-writer`，两条支线之间唯一的差别是内容。装进去的那份会记下内容 hash，写在
`transcript.metadata["skills"]` 里：半年后翻结果，"v2 更好"和"**这个** v2 更好"
是两回事。

### 一条容易踩空的线

```yaml
setting_sources: project
```

Claude Code 默认还会加载操作员自己的 `~/.claude/skills`。那里如果有个同名 skill，
它会盖掉、或者悄悄补上被测的这个——**baseline 支线于是不再是 baseline**，而结果
里看不出任何异常。装了 skill 的支线 Compass 会自动补上 `project`，但 baseline
支线没装东西、没得可补，所以要在共享的 `agent.config` 里显式写一次。

## 离线是怎么做到的

跑的是**真的** `claude_code` adapter、真的 skill 安装、真的 git worktree、真的
grader、真的配对比较。唯一被替换的是 CLI 本身：`replay_cli.py` 演一段运行，而不去
调 Anthropic。

而且它**读 workspace 里装好的那份 SKILL.md 来决定演什么**——不是查一张按 `--model`
索引的表。所以 skill 轴在这个例子里是真的被测着的：把安装逻辑改坏，回放就找不到
skill，报出基线运行，数字立刻变。一张查表会继续输出一份漂亮的、什么都没测的对比。

**接真实 agent = 删掉 `ab.yaml` 里 `cli_path` 那一行**，把 `repo` 指向你自己的
仓库。

## 换成你自己的 skill

1. `skills/` 换成你的两个版本（或者 `你的 skill` + `-m ""` 的基线）。
2. `ab.yaml` 的 cases 换成真实用户会怎么说话——正式的、口语的、只描述场景不提
   名字的。10~20 条足够抓回归。
3. `trigger.yaml` 的负例要写**近似的**：和 skill 沾边、关键词重叠、但正确做法是
   别用它。"写个 fibonacci 函数"这种测不出任何东西。
4. `trials: 3` 起步。触发是概率事件，一次跑出来的不是率，是轶事。
5. 出报告：

```bash
compass site compare out/results.<v1>.json out/results.<v2>.json \
    --label-a v1 --label-b v2 --on triggered \
    --metric skill_triggered --metric cost_usd -o site/
```

更详细的读法见 `compass docs skills`。
