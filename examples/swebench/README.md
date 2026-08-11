# SWE-bench × Compass

把 SWE-bench 实例喂给 `claude_code` adapter，用官方判定协议打分，同时拿到 Compass 的过程指标。

映射几乎就是改个名字——一条 SWE-bench 实例本来就带着 agent 评测需要的全部东西：

| SWE-bench | Compass |
|---|---|
| `repo` + `base_commit` | 每个 trial 的 worktree 从这里拉 |
| `problem_statement` | case 的 prompt |
| `FAIL_TO_PASS` | 必须由红转绿 |
| `PASS_TO_PASS` | 必须保持绿（回归守卫） |

## 先跑离线 demo（不用凭证、不用 Docker、不用 clone Django）

```bash
uv run python examples/swebench/demo.py
```

```
  solves        2/2   100%
  regresses     0/2   0%
  edits_tests   0/2   0%
```

除了 agent，全都是真的：真的 `claude_code` adapter、真的 git worktree、真的打 test patch、真的跑 pytest。`replay_cli.py` 顶替 CLI 回放三种预设行为。

> **`fixtures/instances.jsonl` 里的实例是编的。** 两条虚构实例，套 SWE-bench 的 schema，跑在一个十来行的本地仓库上。它们证明管道通，不说明任何模型的能力。

三种行为分别打在协议最容易出错的地方：

| 行为 | 结果 | 说明 |
|---|---|---|
| `solves` | resolved | 正常修好 |
| `regresses` | 不 resolved | 新测试过了，但弄坏了老的 → `PASS_TO_PASS` 抓住，打 `regression` 标签 |
| `edits_tests` | 不 resolved | 想靠删测试蒙混 → **重置步骤把测试文件恢复了**，一点便宜没占到 |

第三条是这套协议不可被 agent 操纵的原因，也是"直接在仓库里跑 pytest"那种朴素做法拿不到的保证。

## 换成真的 SWE-bench

**1. 拿数据**

```bash
uv run python -c "
from datasets import load_dataset
load_dataset('princeton-nlp/SWE-bench_Verified', split='test').to_json('swebench_verified.jsonl')
"
```

**2. clone 仓库**（一个仓库一次，不是一条实例一次）

```python
from loader import clone_commands, load_instances

instances = load_instances("swebench_verified.jsonl", limit=20)
print("\n".join(clone_commands(instances, "./repos")))
```

**3. 跑**

```python
import asyncio
from compass.core.runner import Compass
import graders                      # 注册 swebench_tests
from loader import build_scenario, load_instances

instances = load_instances("swebench_verified.jsonl", limit=20)
scenario = build_scenario(
    instances,
    repo_root="./repos",
    trials=3,
    agent_config={"model": "claude-opus-4-6"},
)
asyncio.run(Compass().run(scenario))
```

比 prompt 就换 `agent_config={"append_system_prompt_file": "prompts/a.md"}`，其余一个字不动——这是**结构上**保证的可比，不靠自觉。

## 三个必须知道的坑

**① 污染。** SWE-bench Verified 早就进了所有模型的训练集。**比 prompt 没问题**（两边污染程度一样），**比模型是硬伤**——你分不清"这个模型更强"和"这个模型见过这道题"。要比模型，用发布日期晚于模型训练截止的数据集，或者你自己仓库的样本。

**② 环境。** 每条实例要装好依赖才能跑测试。默认是在宿主机上直接跑 pytest，只有当仓库依赖恰好已装好时才成立。要用官方镜像：

```python
scenario = build_scenario(
    instances, repo_root="./repos",
    docker_image="swebench/sweb.eval.x86_64.{repo_underscored}-{instance_id}:latest",
)
```

镜像里仓库在 `/testbed`，grader 把 agent 的 worktree 挂到那个位置盖过去，正好是想要的效果。**这条路我没有实测过**（跑一遍要几十 GB 镜像），配置对不对请自己先验一条。

**③ 测试输出格式。** 只解析 pytest 的 `-rA` 短摘要。自带 runner 的仓库（Django 那种、tox 包装的）会解析出空——grader 把空**当成全挂**，并打 `no_tests_ran` 标签 + 附上输出尾巴，不会假装没事。用 `test_command` 换成能吐 pytest 格式的命令。

## 判定语义

`swebench_tests` 的 `passed` 是官方那个二元 **resolved**：两组测试**全过**才算。`score` 是通过比例——当头条数字没意义，排查时很有用（"23/24" 和 "0/24" 都是红灯，但完全是两回事）。

它是唯一的 gate。`cost_budget` / `turn_count` / `loop_detection` 一律不 gate：它们回答的是 resolution rate 回答不了的问题——**这个分数花了多少钱换来的**。两个都做到 45% 的配置，如果一个贵三倍，它们并不等价。

> `loader.default_process_graders()` 里的阈值是占位符。**用你自己观测到的分布去定**——抄别人仓库的阈值，只会得到一串关于别的项目的自信数字。

## 文件

```
examples/swebench/
├── loader.py                  # 实例 → Scenario（映射 + 防泄漏）
├── graders.py                 # swebench_tests：官方判定协议
├── demo.py                    # 离线演示
├── replay_cli.py              # 顶替 claude 的回放器（仅 demo 用）
└── fixtures/
    ├── instances.jsonl        # 两条虚构实例
    └── repo/                  # 它们指向的小仓库
```

`loader.py` / `graders.py` 放在 `examples/` 而不是核心里，是刻意的：Compass 的核心只保留可复用的骨架（Transcript 模型、过程 grader、指标），**correctness 是领域相关的、以用户代码接入**——一个 benchmark 的判定协议大概是最领域相关的东西了。要接自己的数据集，照着 `loader.py` 改就是了。
