# 跨栈评测站点 · Claude Code + haiku 4.5 vs pi + gemini-2.5-flash

一次真实运行的发布产物：两个 run + 一份配对比较。**浏览器不会从 `file://` 取数据**，
要起个静态服务：

```bash
python3 -m http.server -d reports/crossstack
```

打开后总览页列两个 run 和一份比较，点比较那行进对比页。

## 里面是什么

```
index.json                              runs 与 comparisons 两个数组
runs/claude-haiku/run.json              Claude Code + haiku 4.5，3 条用例 × 3 trial
runs/pi-gemini/run.json                 pi + gemini-2.5-flash，同上
comparisons/claude-haiku-vs-pi-gemini/  配对比较：总体 + 逐 grader + 过程指标
index.html                              viewer（单文件、零依赖）
```

**没有发布轨迹。** 一条 transcript 带着完整的 prompt 和模型输出，而且这份运行的轨迹
有 1.1MB、随时可以重跑出来——不属于版本库。要带上：给两条 `site build` 各加一个
`--trace-dir`。

## 怎么重新生成

用例套件是 [`examples/coding_agent/crossstack.claude.yaml`](../../examples/coding_agent/crossstack.claude.yaml)
和 [`crossstack.pi.yaml`](../../examples/coding_agent/crossstack.pi.yaml)——**同一份判分契约**，
只有 `agent:` 段不同。两次跑用同一个 `--out`，仓库基线因此是同一份：

```bash
OUT=./coding-eval-run
uv run python examples/coding_agent/run.py --suite crossstack.claude.yaml \
    -m haiku --trials 3 --out $OUT
cp $OUT/results.json $OUT/results.claude-haiku.json

uv run python examples/coding_agent/run.py --suite crossstack.pi.yaml \
    -m google/gemini-2.5-flash --trials 3 --out $OUT
cp $OUT/results.json $OUT/results.pi-gemini.json
```

单模型运行只写 `results.json`（per-model 文件只在 `-m` 给了多个模型时才有），所以
两次之间要各留一份副本。然后发布：

```bash
uv run compass site build $OUT/results.claude-haiku.json -o reports/crossstack \
    --slug claude-haiku --name "Claude Code + haiku 4.5"
uv run compass site build $OUT/results.pi-gemini.json -o reports/crossstack \
    --slug pi-gemini --name "pi + gemini-2.5-flash"
uv run compass site compare $OUT/results.claude-haiku.json $OUT/results.pi-gemini.json \
    -o reports/crossstack --slug claude-haiku-vs-pi-gemini \
    --name "Claude Code + haiku 4.5 vs pi + gemini-2.5-flash" \
    --label-a "Claude Code + haiku 4.5" --label-b "pi + gemini-2.5-flash"
```

成本：Claude 侧约 $0.61（9 次尝试），pi 侧约 $0.03。

## 这次跑出来的结论

两个栈三条用例都做了下来，pass^3 各 2/3。差别在**同一条需要定位的题**上，而且失败
方式正好相反：

| | 隐藏验收测试 | 改动范围（不许改 `tests/`） |
|---|---|---|
| Claude Code + haiku 4.5 | 3/3 全对 | **2/3** — 有一次改了 `tests/test_orders.py` |
| pi + gemini-2.5-flash | 2/3 | 3/3 干净 |

一个把答案做对了却越了界，一个守住了边界却没做对。过程侧差一个量级：每次尝试
$0.0682 对 $0.0037（18 倍），19.2 轮对 4.2 轮，两项 95% CI 都不跨 0。

**这比的是「栈」不是「模型」**：两个 CLI 的系统提示词、内置工具集、上下文策略都不同。
要比模型，在同一个 CLI 上换 `-m`。n=3，正确性那条差异的置信区间跨 0——**测不出来**，
不等于没差别。
