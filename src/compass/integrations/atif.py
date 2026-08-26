"""Import an ATIF trajectory (Harbor's interchange format) into a Compass Transcript.

**ATIF** — *Agent Trajectory Interchange Format* — is the versioned trajectory
schema from `Harbor <https://github.com/harbor-framework/harbor>`_, the harness
behind Terminal-Bench. It is one JSON document per agent run: a root ``agent``
block, a flat ``steps`` array, and optionally embedded ``subagent_trajectories``.
Harbor's agents write one to ``<trial_dir>/agent/trajectory.json``, and a growing
set of harnesses emit it directly.

Harbor scores a run with a *verifier* — a script inside the task's container that
writes a reward — and the trajectory itself carries **no verdict at all**. That
split is exactly the seam Compass fits into: the reward says whether the answer
was right, and this importer turns the trajectory into a Compass
:class:`~compass.core.transcript.Transcript` so the *process* can be graded with
the normal transcript-scope graders — cost, loops, tool usage, dangerous ops::

    from compass.integrations import import_atif_file

    t = import_atif_file("jobs/run-1/my-task/trial-1/agent/trajectory.json")
    # t.tool_calls / t.reasoning_steps / t.outcome are ready to grade

Mapping (ATIF -> Compass):

- ``agent``                     -> ``environment.adapter_version`` /
  ``environment.model_version``, and ``metadata["atif"]["agent"]``
- first ``source: "user"`` step -> ``input_prompt`` (later ones are reasoning
  steps, which is where Harbor's simulated-user turns land)
- first ``source: "system"`` step -> ``input_params["system_prompt"]``
- ``source: "agent"`` step      -> one ``llm.generation`` ``ToolCall`` carrying
  that step's ``TokenUsage`` + ``CostInfo``, with the step's message as output
- ``step.tool_calls[]``         -> one ``ToolCall`` each, keyed by
  ``tool_call_id`` and named by ``function_name``
- ``step.observation.results[]`` -> the output of the call named by
  ``source_call_id``; a result with no ``source_call_id`` is a system-initiated
  event and becomes a reasoning step
- ``reasoning_content``         -> a reasoning step
- ``subagent_trajectories`` / ``SubagentTrajectoryRef`` -> the subagent's calls,
  flattened into the same transcript with ``ToolCall.agent_name`` set (see below)
- last root-level agent message -> ``outcome.output_data["final_output"]``

**Copied context is dropped, and that is the point.** ATIF-v1.5 added
``is_copied_context``: steps carried over from a previous trajectory after a
summarization/compaction boundary. They are a *record-keeping* artifact, not work
the agent did on this run. Counting them would bill their tokens twice and make
every compaction look like a loop to a repeat-detection grader. This importer
skips them and reports how many it dropped in
``metadata["atif"]["copied_context_steps"]``.

**Deterministic dispatch emits no LLM call.** ATIF-v1.7's ``llm_call_count: 0``
marks an agent step that did no inference (a hard-coded orchestration hop), so no
``llm.generation`` is created for it — otherwise ``turn_count`` with
``count_filter: "llm"`` would count scaffolding as thinking. When
``llm_call_count > 1``, one step aggregates several inferences and the count is
kept on the call's metadata rather than split into calls that were never
separately measured.

**Subagents are flattened, not separated.** Compass's ``ToolCall.agent_name`` is
the slot for whose work a call was, and ATIF's ``total_cost_usd`` is defined to
*include* subagents — so a delegated trajectory's calls are appended to the
parent transcript right after the delegating call, tagged with the subagent's
name. The alternative (a transcript per agent) would make the run's cost and tool
usage unreadable from any single one of them. External refs
(``trajectory_path``) resolve relative to the source file's directory and only
inside it; anything else is left unresolved and listed in
``metadata["atif"]["unresolved_subagent_refs"]`` rather than read off disk.

**Four things ATIF does not carry**, each of which shows up as an explicit gap
rather than a plausible-looking number:

- **No tool-call failure signal.** ATIF models an observation as content, with no
  status, exit code, or error flag. Every imported call therefore reads
  ``status="ok"``, and a grader keyed on ``status="error"`` measures *nothing* on
  an ATIF trace. Read an all-``ok`` transcript as "not recorded", never as
  "nothing failed".
- **No per-call durations.** ``timestamp`` dates a *step*, not a call, and one
  step may hold several parallel tool calls. Durations are left at ``0.0`` and
  the note lives in ``metadata["atif"]["timings"]``; the run's wall clock is
  still recovered from the first and last step timestamps when they are present.
- **No state deltas.** Nothing in the schema says which files a call changed, so
  ``state_delta`` is always empty here — unlike the Claude/codex importers, which
  have a structured file-change record to map. A ``state_delta`` grader on an
  ATIF trace passes vacuously.
- **No verdict.** Correctness lives in Harbor's ``VerifierResult``, in a sibling
  file. When importing from a Harbor trial directory this module reads that
  sibling ``results.json`` and parks its rewards in ``metadata["harbor"]`` so the
  two signals sit side by side — it never turns them into a Compass score.

**Cost.** ``metrics.cost_usd`` is used verbatim when the producer recorded it.
Otherwise cost is computed from :func:`~compass.llm.calculate_cost`,
which needs a pricing entry for the step's model: no entry means ``cost=None``
and a ``cost_budget`` grader that passes on $0.00 having measured nothing.
Register a rate before reading that grader's verdict::

    from compass.llm import register_pricing
    register_pricing("claude-opus-4-1", {"input": 15.0, "output": 75.0})

ATIF's ``prompt_tokens`` includes cache hits; Compass keeps cached traffic out of
``input_tokens`` and reports it separately, because it is priced differently.

Zero dependency on Harbor: the format is plain JSON, parsed directly.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from compass.core.transcript import CostInfo, TokenUsage, ToolCall, Transcript
from compass.integrations._common import MAX_STEP_CHARS, as_dict, token_count
from compass.llm import calculate_cost

logger = logging.getLogger(__name__)

# Max characters kept for a single reasoning step / for a captured prompt.
_MAX_PROMPT_CHARS = 8000

# Delegation can nest; a malformed file could nest it in a cycle. `visited`
# stops the cycle, this stops a pathologically deep one.
_MAX_SUBAGENT_DEPTH = 8

# Harbor writes one trajectory per trial under `<trial_dir>/agent/`.
_DEFAULT_DIR_PATTERN = "**/trajectory.json"

_SCHEMA_PREFIX = "ATIF-"

# function_name -> Compass tool_type. ATIF does not type its tool calls, so this
# is a name heuristic over what harnesses actually call things; anything
# unrecognized is a plain "function", never a guess.
_CODE_TOOLS = frozenset(
    {"bash", "shell", "sh", "exec", "run", "python", "run_command", "execute_command",
     "terminal", "command_execution"}
)
_FILE_TOOLS = frozenset(
    {"read", "write", "edit", "read_file", "write_file", "edit_file", "create_file",
     "apply_patch", "str_replace_editor", "str_replace_based_edit_tool", "ls"}
)
_SEARCH_TOOLS = frozenset(
    {"grep", "glob", "find", "search", "web_search", "websearch", "browse",
     "codebase_search", "fetch", "web_fetch"}
)


class ATIFImportError(ValueError):
    """Raised when a payload is not a recognizable ATIF trajectory."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def import_atif_file(
    path: str | Path,
    *,
    task_id: str | None = None,
    trial_id: str | None = None,
    model: str | None = None,
    include_subagents: bool = True,
) -> Transcript:
    """Import an ATIF trajectory file into a Transcript.

    Args:
        path: Path to an ATIF ``.json`` document.
        task_id: Task id for the transcript. Defaults to the Harbor
            ``results.json`` sibling's ``task_name`` if there is one, else the
            trajectory's ``session_id`` / ``trajectory_id``.
        trial_id: Trial id. Same defaulting, via ``trial_name``.
        model: Model that ran, overriding ``agent.model_name`` and any
            per-step ``model_name``. Only needed when the trajectory does not
            name a model and you want cost computed.
        include_subagents: Flatten delegated subagent trajectories into this
            transcript (see the module docstring). When False, subagent calls
            are omitted and only the delegating call remains.

    Returns:
        A reconstructed :class:`~compass.core.transcript.Transcript`.

    Raises:
        ATIFImportError: If the payload is not an ATIF trajectory.
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path).expanduser()
    return load_atif(
        path.read_text(encoding="utf-8"),
        source=str(path),
        task_id=task_id,
        trial_id=trial_id,
        model=model,
        base_dir=path.parent,
        harbor_sidecar=_read_harbor_sidecar(path),
        include_subagents=include_subagents,
    )


def import_atif_dir(
    path: str | Path,
    *,
    pattern: str = _DEFAULT_DIR_PATTERN,
    model: str | None = None,
    include_subagents: bool = True,
) -> list[Transcript]:
    """Import every ATIF trajectory under a directory — e.g. a whole Harbor job.

    A Harbor job directory holds one trajectory per trial at
    ``<job>/<task>/<trial>/agent/trajectory.json`` (and, for multi-step trials,
    ``<trial>/steps/<step>/agent/trajectory.json``), so the default pattern
    picks up every trial in a run.

    Files that match the pattern but are not ATIF are skipped with a warning
    rather than failing the import — a directory sweep should not die on one
    stray file.

    Args:
        path: Directory to sweep.
        pattern: Glob relative to *path*.
        model: Model override, applied to every trajectory.
        include_subagents: See :func:`import_atif_file`.

    Returns:
        Transcripts ordered by start time (may be empty).

    Raises:
        ATIFImportError: If *path* is not a directory.
    """
    root = Path(path).expanduser()
    if not root.is_dir():
        raise ATIFImportError(f"{root}: not a directory")

    transcripts: list[Transcript] = []
    for f in sorted(root.glob(pattern)):
        if not f.is_file():
            continue
        try:
            transcripts.append(
                import_atif_file(f, model=model, include_subagents=include_subagents)
            )
        except (ATIFImportError, OSError, ValueError) as e:
            logger.warning("ATIF import: skipping %s (%s)", f, e)
    transcripts.sort(key=lambda t: t.start_time)
    return transcripts


def load_atif(
    content: str,
    *,
    source: str = "<string>",
    task_id: str | None = None,
    trial_id: str | None = None,
    model: str | None = None,
    base_dir: Path | None = None,
    harbor_sidecar: dict[str, Any] | None = None,
    include_subagents: bool = True,
) -> Transcript:
    """Reconstruct a Transcript from the raw text of an ATIF document.

    Same as :func:`import_atif_file` but takes the file *content* directly,
    which is convenient for tests and in-memory pipelines. ``base_dir`` is the
    directory external ``trajectory_path`` refs resolve against; without it,
    external refs are left unresolved.
    """
    stripped = content.strip()
    if not stripped:
        raise ATIFImportError(f"{source}: empty payload")
    try:
        data = json.loads(stripped)
    except (ValueError, TypeError) as e:
        raise ATIFImportError(f"{source}: not valid JSON ({e})") from e
    return atif_to_transcript(
        data,
        source=source,
        task_id=task_id,
        trial_id=trial_id,
        model=model,
        base_dir=base_dir,
        harbor_sidecar=harbor_sidecar,
        include_subagents=include_subagents,
    )


def atif_to_transcript(
    data: Any,
    *,
    source: str = "<string>",
    task_id: str | None = None,
    trial_id: str | None = None,
    model: str | None = None,
    base_dir: Path | None = None,
    harbor_sidecar: dict[str, Any] | None = None,
    include_subagents: bool = True,
) -> Transcript:
    """Convert an already-parsed ATIF trajectory dict into a Transcript."""
    if not looks_like_atif(data):
        raise ATIFImportError(
            f"{source}: not an ATIF trajectory. Expected an object with a "
            "'steps' array and an 'agent' object (schema_version 'ATIF-*')."
        )

    agent = data.get("agent") or {}
    ids = _identity(data, harbor_sidecar)
    t = Transcript(
        task_id=task_id or ids["task_id"],
        trial_id=trial_id or ids["trial_id"],
    )

    resolved_model = model or agent.get("model_name") or ""
    if resolved_model:
        t.environment.model_version = str(resolved_model)
    if agent.get("name"):
        name, version = str(agent["name"]), agent.get("version") or ""
        t.environment.adapter_version = f"{name}@{version}" if version else name

    ctx = _Ctx(
        transcript=t,
        model_override=model,
        base_dir=base_dir,
        include_subagents=include_subagents,
    )
    _ingest(ctx, data, agent_name=None, parent_turn=None, depth=0)

    _finalize(ctx, data, agent, source, harbor_sidecar)
    return t


def looks_like_atif(obj: Any) -> bool:
    """Whether a parsed JSON object is an ATIF trajectory (used for sniffing).

    An explicit ``schema_version`` of ``ATIF-*`` settles it. Producers that omit
    it are still recognizable by the shape ATIF requires and no other Compass
    trace format has: an ``agent`` object beside a non-empty ``steps`` array
    whose entries carry a ``step_id``.
    """
    if not isinstance(obj, dict):
        return False
    schema = obj.get("schema_version")
    if isinstance(schema, str) and schema.startswith(_SCHEMA_PREFIX):
        return True
    steps = obj.get("steps")
    if not isinstance(obj.get("agent"), dict) or not isinstance(steps, list) or not steps:
        return False
    first = steps[0]
    return isinstance(first, dict) and "step_id" in first and "source" in first


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


@dataclass
class _Ctx:
    """Mutable state threaded through a (possibly nested) trajectory walk."""

    transcript: Transcript
    model_override: str | None
    base_dir: Path | None
    include_subagents: bool

    # Cycle guard for subagent refs, keyed by trajectory_id / resolved path.
    visited: set[str] = field(default_factory=set)

    copied_steps: int = 0
    images: int = 0
    llm_steps: int = 0
    subagents: list[str] = field(default_factory=list)
    unresolved: list[dict[str, Any]] = field(default_factory=list)
    timestamps: list[datetime] = field(default_factory=list)
    final_text: str = ""
    saw_user: bool = False
    root_steps: int = 0


def _ingest(
    ctx: _Ctx,
    traj: dict[str, Any],
    *,
    agent_name: str | None,
    parent_turn: int | None,
    depth: int,
) -> None:
    """Walk one trajectory's steps into ``ctx.transcript``.

    ``agent_name`` is None for the root agent and the subagent's name below it;
    ``parent_turn`` is None at the root (each step is its own turn) and the
    delegating step's turn for a subagent, so a delegation reads as *one* turn of
    the parent no matter how many turns happened inside it.
    """
    embedded = _embedded_index(traj)
    root = parent_turn is None

    for raw in traj.get("steps") or []:
        if not isinstance(raw, dict):
            continue
        if raw.get("is_copied_context"):
            ctx.copied_steps += 1
            continue

        step_id = raw.get("step_id")
        turn = parent_turn if parent_turn is not None else (
            int(step_id) if isinstance(step_id, int) else None
        )
        _record_timestamp(ctx, raw.get("timestamp"))

        text, images = _flatten_content(raw.get("message"))
        ctx.images += images
        srcname = raw.get("source")
        first_step = root and ctx.root_steps == 0
        if root:
            ctx.root_steps += 1

        calls: dict[str, ToolCall] = {}
        if srcname == "user":
            _on_user_step(ctx, text, root=root)
        elif srcname == "system":
            _on_system_step(ctx, text, first_step=first_step)
        else:
            # source == "agent" (the schema allows nothing else).
            reasoning = raw.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning.strip():
                ctx.transcript.add_reasoning_step(f"[reasoning] {_truncate(reasoning)}")
            if text.strip():
                ctx.transcript.add_reasoning_step(f"[message] {_truncate(text)}")
                if root:
                    ctx.final_text = text

            _emit_llm_call(ctx, raw, text, agent_name=agent_name, turn=turn)
            calls = _emit_tool_calls(
                ctx, raw, agent_name=agent_name, turn=turn, step_id=step_id
            )

        # Every source can carry an observation — ATIF reserves only
        # model_name/reasoning_effort/reasoning_content/tool_calls/metrics for
        # agent steps. A `system` step with an observation is how a compaction
        # handoff records the subagents it delegated to, so skipping non-agent
        # observations would drop that work (and its whole token bill) silently.
        _apply_observation(ctx, raw, calls, embedded=embedded, turn=turn, depth=depth)


def _on_user_step(ctx: _Ctx, text: str, *, root: bool) -> None:
    """The first user turn is the task; later ones are follow-ups mid-run."""
    if root and not ctx.saw_user:
        ctx.saw_user = True
        ctx.transcript.input_prompt = text
        return
    if text.strip():
        ctx.transcript.add_reasoning_step(f"[user] {_truncate(text)}")


def _on_system_step(ctx: _Ctx, text: str, *, first_step: bool) -> None:
    """A system step is the system prompt only if it opens the run.

    Anything later is an injected event mid-run — a compaction handoff, a
    timeout notice — and reads as a trace step, not as configuration. The
    distinction matters: parking a handoff notice in ``system_prompt`` would
    make every summarizing agent look like it was configured mid-flight.
    """
    if first_step:
        if text:
            ctx.transcript.input_params["system_prompt"] = _truncate(
                text, _MAX_PROMPT_CHARS
            )
            if len(text) > _MAX_PROMPT_CHARS:
                ctx.transcript.input_params["system_prompt_truncated"] = True
        return
    if text.strip():
        ctx.transcript.add_reasoning_step(f"[system] {_truncate(text)}")


def _emit_llm_call(
    ctx: _Ctx,
    step: dict[str, Any],
    text: str,
    *,
    agent_name: str | None,
    turn: int | None,
) -> None:
    """One ``llm.generation`` per agent step that actually ran an inference."""
    count = step.get("llm_call_count")
    if count == 0:  # ATIF-v1.7 deterministic dispatch — no model was called.
        return

    model = str(
        ctx.model_override
        or step.get("model_name")
        or ctx.transcript.environment.model_version
        or ""
    )
    tokens, cost = _step_tokens_cost(step.get("metrics"), model)

    meta: dict[str, Any] = {}
    if model:
        meta["model"] = model
    if isinstance(count, int) and count > 1:
        # Several inferences billed as one step; splitting them would invent
        # per-call numbers the producer never measured.
        meta["llm_call_count"] = count
    if step.get("reasoning_effort") is not None:
        meta["reasoning_effort"] = step["reasoning_effort"]

    ctx.llm_steps += 1
    ctx.transcript.tool_calls.append(
        ToolCall(
            tool_name="llm.generation",
            input={},
            output=_truncate(text) if text else None,
            status="ok",
            tokens=tokens,
            cost=cost,
            tool_type="llm",
            metadata=meta,
            turn_index=turn,
            agent_name=agent_name,
        )
    )


def _emit_tool_calls(
    ctx: _Ctx,
    step: dict[str, Any],
    *,
    agent_name: str | None,
    turn: int | None,
    step_id: Any,
) -> dict[str, ToolCall]:
    """Map a step's ``tool_calls`` array; return them keyed by tool_call_id."""
    by_id: dict[str, ToolCall] = {}
    for raw in step.get("tool_calls") or []:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("function_name") or "unknown_tool")
        call_id = str(raw.get("tool_call_id") or "")

        meta: dict[str, Any] = {}
        if isinstance(step_id, int):
            meta["atif_step_id"] = step_id
        if isinstance(raw.get("extra"), dict):
            meta["extra"] = raw["extra"]

        kwargs: dict[str, Any] = dict(
            tool_name=name,
            input=as_dict(raw.get("arguments")),
            # ATIF records no failure signal — see the module docstring.
            status="ok",
            tool_type=_tool_type(name),
            metadata=meta,
            turn_index=turn,
            agent_name=agent_name,
        )
        if call_id:
            kwargs["call_id"] = call_id
        call = ToolCall(**kwargs)
        ctx.transcript.tool_calls.append(call)
        if call_id:
            by_id[call_id] = call
    return by_id


def _apply_observation(
    ctx: _Ctx,
    step: dict[str, Any],
    calls: dict[str, ToolCall],
    *,
    embedded: dict[str, dict[str, Any]],
    turn: int | None,
    depth: int,
) -> None:
    """Attach each observation result to its call, and follow any delegation.

    ATIF requires ``source_call_id`` to name a tool call in *this same step*, so
    a step is self-contained: action and result arrive together.
    """
    observation = step.get("observation")
    if not isinstance(observation, dict):
        return

    for result in observation.get("results") or []:
        if not isinstance(result, dict):
            continue
        text, images = _flatten_content(result.get("content"))
        ctx.images += images

        call_id = result.get("source_call_id")
        call = calls.get(str(call_id)) if call_id is not None else None
        if call is not None:
            call.output = _join_output(call.output, text)
            if isinstance(result.get("extra"), dict):
                call.metadata["result_extra"] = result["extra"]
        elif text.strip():
            # No source_call_id: a system-initiated event, not a tool result.
            ctx.transcript.add_reasoning_step(f"[observation] {_truncate(text)}")

        for ref in result.get("subagent_trajectory_ref") or []:
            if isinstance(ref, dict):
                _follow_subagent(ctx, ref, embedded=embedded, turn=turn, depth=depth)


def _follow_subagent(
    ctx: _Ctx,
    ref: dict[str, Any],
    *,
    embedded: dict[str, dict[str, Any]],
    turn: int | None,
    depth: int,
) -> None:
    """Resolve one delegation and fold the subagent's work into the transcript."""
    if not ctx.include_subagents:
        return
    if depth >= _MAX_SUBAGENT_DEPTH:
        ctx.unresolved.append({**_ref_key(ref), "reason": "max depth exceeded"})
        return

    sub, key, reason = _resolve_subagent(ctx, ref, embedded)
    if sub is None:
        ctx.unresolved.append({**_ref_key(ref), "reason": reason})
        return
    if key in ctx.visited:
        ctx.unresolved.append({**_ref_key(ref), "reason": "already visited (cycle)"})
        return
    ctx.visited.add(key)

    name = str((sub.get("agent") or {}).get("name") or "subagent")
    if name not in ctx.subagents:
        ctx.subagents.append(name)
    _ingest(ctx, sub, agent_name=name, parent_turn=turn, depth=depth + 1)


def _resolve_subagent(
    ctx: _Ctx, ref: dict[str, Any], embedded: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any] | None, str, str]:
    """(trajectory, cycle-key, failure reason) for one subagent reference.

    Embedded refs (``trajectory_id``) resolve in-memory. External refs
    (``trajectory_path``) are read only when they resolve *inside* the source
    file's directory — a trajectory is data, and data does not get to name
    arbitrary paths on the importing machine.
    """
    traj_id = ref.get("trajectory_id")
    if isinstance(traj_id, str) and traj_id in embedded:
        return embedded[traj_id], f"id:{traj_id}", ""

    raw_path = ref.get("trajectory_path")
    if not isinstance(raw_path, str) or not raw_path:
        if traj_id:
            return None, "", f"trajectory_id {traj_id!r} not found in subagent_trajectories"
        return None, "", "ref names neither trajectory_id nor trajectory_path"

    if ctx.base_dir is None:
        return None, "", "external trajectory_path, but the source has no directory"

    base = ctx.base_dir.resolve()
    target = (base / raw_path).resolve()
    if not target.is_relative_to(base):
        return None, "", "trajectory_path resolves outside the source directory"
    if not target.is_file():
        return None, "", "trajectory_path does not exist"

    try:
        sub = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return None, "", f"trajectory_path unreadable ({e})"
    if not looks_like_atif(sub):
        return None, "", "trajectory_path is not an ATIF trajectory"
    return sub, f"path:{target}", ""


def _finalize(
    ctx: _Ctx,
    data: dict[str, Any],
    agent: dict[str, Any],
    source: str,
    harbor_sidecar: dict[str, Any] | None,
) -> None:
    """Timing, provenance and outcome — everything that needs the whole walk."""
    t = ctx.transcript

    if ctx.timestamps:
        t.start_time = min(ctx.timestamps)
        end = max(ctx.timestamps)
        if end > t.start_time:
            t.end_time = end
            t.total_duration_ms = (end - t.start_time).total_seconds() * 1000.0

    atif: dict[str, Any] = {
        "source": source,
        "schema_version": data.get("schema_version"),
        "session_id": data.get("session_id"),
        "trajectory_id": data.get("trajectory_id"),
        "agent": {k: agent.get(k) for k in ("name", "version", "model_name") if agent.get(k)},
        "llm_steps": ctx.llm_steps,
        # Durations are per-step in ATIF and per-call in Compass; see the docstring.
        "timings": "per-call durations not recorded: ATIF timestamps steps, not calls",
    }
    if data.get("notes"):
        atif["notes"] = _truncate(str(data["notes"]))
    if data.get("continued_trajectory_ref"):
        atif["continued_trajectory_ref"] = data["continued_trajectory_ref"]
    if isinstance(data.get("final_metrics"), dict):
        # Reported totals, kept beside the sums Compass computes from the steps.
        # They are allowed to disagree: ATIF's totals include subagents and
        # copied-context steps, which this importer deliberately does not count.
        atif["final_metrics_reported"] = data["final_metrics"]
    if isinstance(agent.get("tool_definitions"), list):
        atif["available_tools"] = sorted(
            {n for d in agent["tool_definitions"] if (n := _tool_def_name(d))}
        )
    if ctx.copied_steps:
        atif["copied_context_steps"] = ctx.copied_steps
    if ctx.images:
        atif["image_content_parts"] = ctx.images
    if ctx.subagents:
        atif["subagents"] = ctx.subagents
    if ctx.unresolved:
        atif["unresolved_subagent_refs"] = ctx.unresolved
    if not ctx.include_subagents:
        atif["subagents_excluded"] = True
    t.metadata["atif"] = atif

    if harbor_sidecar:
        t.metadata["harbor"] = harbor_sidecar

    if ctx.final_text:
        t.set_outcome(output_data={"final_output": ctx.final_text})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identity(
    data: dict[str, Any], harbor_sidecar: dict[str, Any] | None
) -> dict[str, str]:
    """Best available (task_id, trial_id) — Harbor's names beat opaque uuids."""
    session = str(data.get("session_id") or data.get("trajectory_id") or "atif-import")
    task = str((harbor_sidecar or {}).get("task_name") or session)
    trial = str((harbor_sidecar or {}).get("trial_name") or session)
    return {"task_id": task, "trial_id": trial}


def _read_harbor_sidecar(trajectory_path: Path) -> dict[str, Any] | None:
    """Harbor's ``results.json`` for the trial this trajectory belongs to.

    Layout is ``<trial>/agent/trajectory.json`` (single-step) or
    ``<trial>/steps/<step>/agent/trajectory.json`` (multi-step), so the trial
    root is at most four levels up. Only the verdict-shaped fields are kept —
    the whole TrialResult would drag a Harbor config into every transcript.
    """
    for parent in list(trajectory_path.resolve().parents)[:4]:
        candidate = parent / "results.json"
        if not candidate.is_file():
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict) or "trial_name" not in data:
            return None
        out: dict[str, Any] = {"results_path": str(candidate)}
        for key in ("task_name", "trial_name", "trial_uri", "task_checksum"):
            if data.get(key):
                out[key] = data[key]
        verifier = data.get("verifier_result")
        if isinstance(verifier, dict) and verifier.get("rewards") is not None:
            out["rewards"] = verifier["rewards"]
        exc = data.get("exception_info")
        if isinstance(exc, dict) and exc.get("exception_message"):
            out["exception"] = _truncate(str(exc["exception_message"]), 500)
        return out
    return None


def _embedded_index(traj: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """``trajectory_id`` -> embedded subagent trajectory, for ref resolution."""
    index: dict[str, dict[str, Any]] = {}
    for sub in traj.get("subagent_trajectories") or []:
        if isinstance(sub, dict) and isinstance(sub.get("trajectory_id"), str):
            index[sub["trajectory_id"]] = sub
    return index


def _ref_key(ref: dict[str, Any]) -> dict[str, Any]:
    """The identifying fields of a ref, for reporting one we could not follow."""
    return {
        k: ref[k]
        for k in ("trajectory_id", "trajectory_path", "session_id")
        if ref.get(k)
    }


def _step_tokens_cost(
    metrics: Any, model: str
) -> tuple[TokenUsage | None, CostInfo | None]:
    """Map a step's ``metrics`` to Compass ``TokenUsage`` + ``CostInfo``.

    ATIF's ``prompt_tokens`` is the whole input bill, cache hits included;
    Compass keeps cached traffic out of ``input_tokens`` and reports it
    separately, because it is priced differently. So the fresh input is what is
    left after the cache reads are taken out.
    """
    if not isinstance(metrics, dict):
        return None, None

    prompt = token_count(metrics.get("prompt_tokens"))
    completion = token_count(metrics.get("completion_tokens"))
    cached = token_count(metrics.get("cached_tokens"))
    fresh_input = max(0, prompt - cached)

    tokens: TokenUsage | None = None
    if prompt or completion or cached:
        tokens = TokenUsage(
            input_tokens=fresh_input,
            output_tokens=completion,
            total_tokens=prompt + completion,
            cache_read_tokens=cached,
            metadata={"prompt_tokens_including_cache": prompt} if prompt else {},
        )

    reported = metrics.get("cost_usd")
    if isinstance(reported, (int, float)):
        # The producer priced the call; Compass's table does not get a vote.
        return tokens, CostInfo(total_usd=float(reported), metadata={"source": "atif"})
    cost = calculate_cost(model, fresh_input, completion, cached) if model else None
    return tokens, cost


def _flatten_content(value: Any) -> tuple[str, int]:
    """(text, image_count) for an ATIF message or observation content.

    ATIF-v1.6 allows a string or an array of ContentParts. Images are counted
    and named, not fetched — the parts hold a path or URL, and reading either
    one off a trace file is not this importer's job.
    """
    if value is None:
        return "", 0
    if isinstance(value, str):
        return value, 0
    if not isinstance(value, list):
        return str(value), 0

    chunks: list[str] = []
    images = 0
    for part in value:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "image":
            images += 1
            src = part.get("source") or {}
            path = src.get("path") if isinstance(src, dict) else None
            chunks.append(f"[image: {path}]" if path else "[image]")
        elif part.get("text"):
            chunks.append(str(part["text"]))
    return "\n".join(chunks), images


def _join_output(existing: Any, text: str) -> Any:
    """Several observation results can name one call; keep them all, in order."""
    if not text:
        return existing
    if existing is None or existing == "":
        return text
    return f"{existing}\n{text}"


def _tool_def_name(definition: Any) -> str | None:
    """The name out of an OpenAI-shaped tool definition, nested or flat."""
    if not isinstance(definition, dict):
        return None
    fn = definition.get("function")
    if isinstance(fn, dict) and fn.get("name"):
        return str(fn["name"])
    return str(definition["name"]) if definition.get("name") else None


def _tool_type(name: str) -> str:
    lowered = name.lower()
    if lowered.startswith("mcp__") or lowered.startswith("mcp."):
        return "mcp"
    if lowered in _SEARCH_TOOLS:
        return "search"
    if lowered in _CODE_TOOLS:
        return "code"
    if lowered in _FILE_TOOLS:
        return "file"
    return "function"


def _record_timestamp(ctx: _Ctx, value: Any) -> None:
    """Collect a step timestamp as a *local* naive datetime.

    Local, not UTC, and naive, not aware, for the same two reasons pi's
    ``_parse_iso_local`` is: the rest of Compass dates transcripts with naive local
    datetimes (``datetime.now()``), and a producer that mixes offset-carrying and
    offset-free timestamps in one file would otherwise make ``min()``/``max()``
    raise on the comparison. Dropping the offset instead of converting through it
    would shift every duration by the host's UTC offset.
    """
    if not isinstance(value, str) or not value:
        return
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        logger.debug("ATIF import: unparseable timestamp %r", value)
        return
    ctx.timestamps.append(
        parsed if parsed.tzinfo is None else parsed.astimezone().replace(tzinfo=None)
    )






def _truncate(text: str, limit: int = MAX_STEP_CHARS) -> str:
    """Like :func:`_common.truncate`, but says how much was cut.

    ATIF is the interchange format, so a step that was shortened on the way
    in should say by how much — a reader comparing an ATIF import against
    the vendor trace it came from can otherwise only see that text is gone.
    """
    return text if len(text) <= limit else text[:limit] + f"… (+{len(text) - limit} chars)"
