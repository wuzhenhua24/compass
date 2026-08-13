# Skill 评测：证明新版本真的更好了

改完一个 Agent Skill 要发版，横在中间的问题是：**这是优化，还是改坏了？**

一个 skill 就是一段提示词加几个 bundle 的文件。它没有单元测试可跑，改动的效果
只体现在"agent 拿到它之后行为变了多少"上——而这正是 Compass 已经在做的事。

> 上手最快的路径：`examples/skill_eval/` 是一个能离线跑通的完整模板。
> `uv run python examples/skill_eval/eval.py`，不需要 API key。

## 一个 skill 会以三种方式坏掉

| 坏法 | 症状 | 谁能抓住 |
|---|---|---|
| **不触发了** | description 改得更"准确"了，于是不再从一堆 skill 里被选中。agent 照样把活干完（干得更笨），正确性指标看上去毫无异常 | `skill_trigger` |
| **抢别人的活** | description 改得更"主动"了，召回上去了，代价是它开始接不该接的请求 | `skill_trigger` + `should_trigger: false` 的负向控制 |
| **bundle 的脚本失联** | SKILL.md 里的脚本路径写错、或者指引不够明确，agent 每次都现场重造一遍轮子 | `skill_trigger` + `required_resources` |

三种都发生在**过程**里，三种都不改变最终答案的对错——所以只看 outcome 的评测
一个都抓不到。这也是为什么 skill 评测的重心在 Transcript 侧。

## 装哪个版本，是一个可扫的轴

`claude_code` adapter 的 `skill:` 配置项指向一个 skill 目录。跑之前，Compass 把
它装进**隔离的 workspace**：

```yaml
agent:
  adapter: claude_code
  config:
    repo: "./my-project"
    skill: ""                    # 由 -m 覆盖；"" = 不装 skill 的基线支线
    skills: []                   # 被测 skill 需要的伴随 skill（可选）
    setting_sources: project     # 见下面「一条容易踩空的线」
```

```bash
compass test ab.yaml --model-key skill \
    -m "" -m skills/report-writer-v1 -m skills/report-writer-v2 \
    --report json -o out/results.json
```

`--model-key skill` 把 `-m` 的值写进 `agent.config.skill`，其余部分（cases、
graders、阈值）逐字不变——所以三条支线之间唯一变动的量就是 skill 本身。这不是
为 skill 新造的机制：`-m` 扫模型、`--model-key append_system_prompt_file` 扫提示
词，用的是同一条路。

### 装进去的是「skill 自己的名字」

安装目录取 **SKILL.md frontmatter 里的 `name`**，不是源目录名：

```
./skills/report-writer-v1/  ──┐
                              ├──> <workspace>/.claude/skills/report-writer/
./skills/report-writer-v2/  ──┘
```

两个版本到 agent 眼里都叫 `report-writer`。按源目录名安装的话，v2 会变成一个
**改了名字的 skill**——名字和描述是触发机制本身的一部分，那样测的就不是同一个
实验了。frontmatter 读不出来时回退到目录名。

### 记的是内容 hash，不是路径

每次安装都会在 `AgentOutput.metadata["skills"]` 和 `transcript.metadata["skills"]`
里留一条：

```json
{"name": "report-writer", "source": "/abs/skills/report-writer-v2",
 "path": ".claude/skills/report-writer", "digest": "9f2a1c0b7e5d4a63", "files": 3}
```

`digest` 是整棵目录的内容哈希。半年后翻结果，"v2 更好"和"**这个** v2 更好"是
两回事——路径会复用，内容不会。

### 几条边界

- **目录里没有 SKILL.md 会直接报错**，不会当成"没装成"继续跑。这是唯一一种能让
  整个对比作废、却不留任何痕迹的失败：一个拼错的路径会安静地把 with-skill 支线
  变成又一条基线。
- **`isolation: none` 会被拒绝**。那种模式直接在你自己的仓库里跑，装 skill 意味着
  往你的 `.claude/skills/` 写文件并留在那儿。用 `worktree`（默认）或 `copy`。
- **装进去的 skill 不算 agent 的改动**：diff 和 `changed_files` 会排掉它，否则
  每条 with-skill 支线都会凭空多出几百行变更，`diff_size` 量的就成了测试框架。

### 一条容易踩空的线

```yaml
setting_sources: project
```

Claude Code 默认还会加载操作员自己的 `~/.claude/skills`。那里如果有个同名 skill，
它会盖掉、或者悄悄补上被测的这个——**基线支线于是不再是基线**，而结果里看不出
任何异常。

Compass 在装了 skill 的支线上会自动补 `--setting-sources project`（并在日志里说
一声）。但基线支线什么都没装，没得可补——所以把它显式写在共享的 `agent.config`
里，三条支线一起生效。

## skill_trigger：唯一无可替代的那个指标

```yaml
graders:
  - name: skill_trigger
    type: code
    label: triggered              # 有 label，compare --on 才选得中
    config:
      skill: report-writer        # 名字或目录都行；不写则读安装记录
      should_trigger: true        # false = 负向控制
      required_resources: ["scripts/summarize.py"]   # bundle 的脚本用了没有
```

**两种信号都算加载**：`Skill` 工具点名，或者任何工具碰到了 skill 目录下的文件
（读 SKILL.md、读 references/、跑 scripts/）。只认前者的话，一个工作得好好的
skill 会被报成 0% 触发率。

**只在正文里提到名字不算**。agent 在 Bash 里 `echo` 一句带 skill 名的话、在写出
的文件里提一嘴，都不构成加载——否则触发率会随着 agent 的话变多而虚高。

产出的指标：

| 指标 | 含义 |
|---|---|
| `skill_triggered` | 布尔；跨 trial 求均值就是**触发率** |
| `skill_trigger_turn` | 第几轮加载的——越早越好，晚加载意味着它先走了一段弯路 |
| `skill_touch_calls` | 碰到 skill 的调用数 |
| `skill_resources_used` | 用上了几个 `required_resources` |

`score=None` 的一种情况：既没配 `skill:`、安装记录里也没有。**基线支线属于这一
类**——它没装任何 skill——所以带基线的 sweep 一定要显式写 `skill: <name>`，让基线
支线诚实地记一笔"没加载"，而不是"没测"。

### 负向控制是评测的另一半

只有正向用例的触发率评测测不出坏 description：把 description 写成"任何时候都用
我"，正向全绿，代价是它开始抢别人的活——而所有正确性指标看上去都还好好的。

负向用例要写**近似的**：和 skill 沾边、关键词重叠、但正确做法是别用它。

```yaml
- id: neg_chart
  input: {prompt: "用 data/q3.csv 画个各区收入的柱状图"}
  graders:
    - {name: skill_trigger, type: code, label: triggered,
       config: {skill: report-writer, should_trigger: false}}
```

> 注意 `default_graders` 是**追加**不是覆盖。把 `should_trigger: true` 写进默认，
> 每条负例都会额外挂上一个必然失败的实例。期望相反的用例，grader 写在用例上。

## 两种 suite

| | 问什么 | 成本 | 什么时候跑 |
|---|---|---|---|
| **触发率** | description 该响的时候响、不该响的时候不响吗 | 低（`max_turns: 3`，agent 不用把活干完） | 每次改完 description |
| **A/B** | 换了这个版本，产出和代价变了多少 | 高（完整运行） | 发版前 |

两者都要 `trials: 3` 起步。触发是概率事件，一次跑出来的不是率，是轶事。

## 出报告

```bash
compass compare out/results.<v1>.json out/results.<v2>.json \
    --on triggered --metric skill_triggered --metric cost_usd

compass site build out/results.<v1>.json -o site/ --slug trigger-v1 --name "trigger · v1"
compass site build out/results.<v2>.json -o site/ --slug trigger-v2 --name "trigger · v2"
compass site compare out/results.<v1>.json out/results.<v2>.json \
    --label-a v1 --label-b v2 --on triggered -o site/
```

`site build` 不认领域，skill 的结果文件直接就能发。**一条支线一个 slug**，发进同一个
目录，清单自己累积——总览页因此是一张"哪个版本跑成什么样"的表。页面上和 skill 直接
相关的三处：

- **每次运行都写明被测的是谁**：`report-writer@9f2a1c0b`——名字加 digest 前 8 位，
  总览列表和运行页各一处。`digest` 是内容哈希，所以"v2 更好"和"**这个** v2 更好"在
  半年后仍然分得开。同一个 slug 重复 build 时，历史快照也各自记着当时那个版本，
  但趋势线**不**在版本变化处断开——那正是这条线要看的东西（判分契约变了才断，
  因为那是尺子变了）。
- **每个 case 的 `skill_triggered` 是跨 trial 的均值**，也就是这条 case 的触发率
  （3 次里中了 1 次 → `0.3333`），和它旁边那个同样是均值的分数对得上。
- 触发失败的 case 带着 `not_triggered` / `unexpected_trigger` / `resource_unused`
  标签，点一下就能筛出来；`skill_via_file` / `skill_via_skill_tool` 说明它是被哪种
  信号认定为"加载了"的。

`--include-details` 才会把 grader 的 `details`（证据：碰到的具体路径、工具名、轮次）
一起发布。skill 的安装记录里那条本机绝对路径（`source`）永远不发布，名字、digest、
文件数会。

`compass compare` 做的是**配对**比较：逐 case 配对、列出翻转（哪条从过变成不过）、
给出 95% 置信区间和当前样本量下的可检测效应（MDE）。这比"两边各报一个均值 ±
标准差"多回答一个问题——**这个差是真的，还是噪声？**

一份真实的输出长这样：

```
按 `triggered` 判分：within noise band: pass-rate diff +16.7%
    (95% CI [-43.6%, +76.9%]); detectable at n=6: ~86.0%
    ↑ pos_contextual
    ↑ pos_implicit
    ↓ neg_chart   ← 回退

触发率（skill_triggered）
    v1 0.1667 → v2 0.6667
    skill_triggered: B is significantly higher — +0.5 (95% CI [+0.06, +0.94])
```

读法：**触发率确实显著提高了**（+0.5，区间不跨 0）；但把正负例合起来看的总体
判定还在噪声带里，而且工具告诉你 6 条用例只能检测到 86% 以上的差异——想要结论，
得加用例，不是加解读。中间那条 `↓ neg_chart` 是均值永远藏得住、翻转永远藏不住的
那类信息。

## 边界：什么不在框架里

skill 干出来的活**对不对**——报告写得好不好、代码改得对不对——是领域判断，属于
用户代码，像 pytest 里的 `tests/`。Compass 提供的是可复用的那半边：它加载了没有、
bundle 的东西用了没有、花了多少、这个差是不是噪声。

要加正确性，就在 case 上挂自己的 grader（`integration_test` / `external_checker` /
自定义 `@register_grader`）。参见 [graders.md](graders.md) 和
`examples/coding_agent/`。

## 其它 CLI

目前只有 `claude_code` 声明了 skill 安装目录（`.claude/skills`）。`codex` 的
skill 约定（`.codex/skills`）没有等价的 hermetic 开关——`--setting-sources` 那种
"只认工作区配置"的保证，Compass 还没验证过怎么在 codex 上做到，所以宁可不做，
也不提供一个安静地不隔离的版本。给 adapter 加上 `skills_dir` 类属性即可接入，
其余（安装、改名、hash、diff 排除）都在共享基类里。

`skill_trigger` 本身不挑 CLI：它匹配的是 `skills/<name>/` 这样的路径片段和名叫
`Skill` 的工具，导入进来的 codex / OTLP 轨迹一样能评。
