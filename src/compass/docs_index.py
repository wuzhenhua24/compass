"""Locating Compass's own documentation at runtime.

``compass docs`` exists so the terminal is enough: a person — or an agent
driving the CLI — can read how the framework works without leaving it, and
pipe it somewhere useful.

Two things this module is careful about:

- **It reads the files, not the package metadata.** ``importlib.metadata``
  carries the README, but only as it was at install time; in an editable
  checkout that snapshot goes stale the moment README.md is edited, and
  printing outdated docs is worse than printing none.
- **The topic list is curated, not a directory listing.** ``docs/`` also holds
  local working notes that are gitignored; globbing would surface them in a
  checkout and make the command's output depend on who is running it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Topic:
    """One documentation page."""

    name: str
    #: Path relative to the repository root.
    source: str
    summary: str

    @property
    def filename(self) -> str:
        return Path(self.source).name


TOPICS: tuple[Topic, ...] = (
    Topic(
        "readme",
        "README.md",
        "定位、核心概念、架构、CLI 总览、快速开始",
    ),
    Topic(
        "cheatsheet",
        "docs/cheatsheet.md",
        "一页速查：YAML 全字段、内置 grader、常用配方、语义坑（建议 agent 先读这个）",
    ),
    Topic(
        "core-design",
        "docs/core-design.md",
        "Transcript/Outcome 分离、GraderScope、ToolCall 协议、JSONL 事件流",
    ),
    Topic(
        "graders",
        "docs/graders.md",
        "评分流水线、子进程 Checker、观察标签、内置与自定义评分器",
    ),
    Topic(
        "scenario-config",
        "docs/scenario-config.md",
        "场景 YAML、多次试验、pass@k / pass^k、采样纪律、分类聚合",
    ),
    Topic(
        "analysis",
        "docs/analysis.md",
        "离线评分、评测卫生、多模型排行榜、analyze / compare、报告",
    ),
    Topic(
        "integrations",
        "docs/integrations.md",
        "轨迹导入（pi / codex / OTLP / Claude / OpenAI Agents）、Adapter 接入",
    ),
    Topic(
        "skills",
        "docs/skills.md",
        "Agent Skill 评测：skill 版本轴、触发率与负向控制、v1→v2 的配对判断",
    ),
    Topic(
        "roadmap",
        "docs/roadmap.md",
        "路线图与已落地能力",
    ),
)

_BY_NAME = {t.name: t for t in TOPICS}


def get_topic(name: str) -> Topic | None:
    return _BY_NAME.get(name.lower().removesuffix(".md"))


def topic_names() -> list[str]:
    return [t.name for t in TOPICS]


def read_topic(topic: Topic) -> str | None:
    """The topic's markdown, or None when it is not on disk.

    Looks in the packaged copy first, then in a source checkout. Both are
    legitimate: a wheel carries ``compass/docs/``, while an editable install
    points at ``src/`` and the real files live at the repository root — and in
    that case the checkout is the only copy that is actually up to date.
    """
    for candidate in _candidate_paths(topic):
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    return None


def _candidate_paths(topic: Topic) -> list[Path]:
    here = Path(__file__).resolve()
    return [
        # src/compass/docs/<file> — the packaged copy
        here.parent / "docs" / topic.filename,
        # <repo>/docs/<file> or <repo>/README.md — a source checkout
        here.parents[2] / topic.source,
    ]
