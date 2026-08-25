"""``compass import`` — ingest an external agent trace."""

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
from rich.panel import Panel
from rich.table import Table

from compass.cli.app import cli, console

if TYPE_CHECKING:
    from compass.core.transcript import Transcript


@cli.command(name="import")
@click.argument("source", type=click.Path(exists=True))
@click.option(
    "--format", "-f", "fmt",
    type=click.Choice(["auto", "pi", "otlp", "claude", "codex", "atif"]), default="auto",
    help="Trace format (default: auto-detect)",
)
@click.option(
    "--output", "-o", type=click.Path(),
    help="Save reconstructed transcript(s). Use a directory for multi-trace sources.",
)
@click.option(
    "--output-format", type=click.Choice(["json", "jsonl"]), default="json",
    help="Saved transcript format",
)
@click.option(
    "--model", "-m", default=None,
    help="Model that produced the trace (codex streams do not record it; "
         "without it there is no cost to compute)",
)
@click.option("--json", "as_json", is_flag=True, help="Print reconstructed transcript(s) as JSON")
def import_(
    source: str,
    fmt: str,
    output: str | None,
    output_format: str,
    model: str | None,
    as_json: bool,
) -> None:
    """Import an external agent trace into Compass transcript(s).

    SOURCE is an offline trace file (or a directory of pi sessions):

    \b
      pi      a pi (@earendil-works/pi-*) JSONL session file, a saved `--mode json`
              event stream, or a directory of either
      otlp    an OTLP / OpenInference trace JSON (LangChain / LlamaIndex / CrewAI …
              exported via Arize Phoenix); a single file may hold many traces
      claude  a Claude Code `--output-format stream-json` file
      codex   a Codex CLI `codex exec --json` event stream
      atif    a Harbor ATIF trajectory (`trajectory.json`), or a directory of
              them — point it at a Harbor job dir to import every trial at once

    Format is auto-detected by default. The OpenAI Agents SDK integration is
    live/streaming (used programmatically via ``compass.integrations``); the
    Claude Agent SDK can be imported from its stream-json output.

    A codex stream records no model id and no cost — pass ``--model`` to name the
    model (which is also what makes cost computable from the pricing table). An
    ATIF trajectory usually names its own model; ``--model`` overrides it.

    Examples:
        compass import session.jsonl                  # auto-detect + summarize
        compass import phoenix_export.json -o out/    # one saved file per trace
        compass import run.stream.jsonl -f claude -o t.json
        compass import codex.stream.jsonl --model gpt-5.4-mini
        compass import jobs/my-run -f atif -o out/    # every trial in a Harbor job
    """
    import json as json_mod

    from compass.integrations import (
        import_atif_dir,
        import_atif_file,
        import_claude_stream_json,
        import_codex_stream_json,
        import_otlp_file,
        import_pi_session,
        import_pi_sessions,
    )

    src = Path(source)
    detected = fmt if fmt != "auto" else _detect_trace_format(src)
    if detected is None:
        console.print(
            "[red]Could not auto-detect trace format.[/red] "
            "Pass [bold]--format pi|otlp|claude|codex|atif[/bold]."
        )
        sys.exit(1)

    try:
        if detected == "pi":
            transcripts = (
                import_pi_sessions(src) if src.is_dir() else [import_pi_session(src)]
            )
        elif detected == "atif":
            transcripts = (
                import_atif_dir(src, model=model)
                if src.is_dir()
                else [import_atif_file(src, model=model)]
            )
        elif detected == "claude":
            transcripts = [import_claude_stream_json(src)]
        elif detected == "codex":
            transcripts = [import_codex_stream_json(src, model=model)]
        else:  # otlp
            transcripts = import_otlp_file(src)
    except Exception as e:
        console.print(f"[red]Failed to import trace: {e}[/red]")
        sys.exit(1)

    if not transcripts:
        console.print("[yellow]No transcripts reconstructed from source.[/yellow]")
        return

    console.print(Panel(
        f"[bold]Compass Import[/bold]  |  format: {detected}  |  "
        f"{len(transcripts)} transcript(s)"
    ))

    if as_json:
        payload = [t.to_dict() for t in transcripts]
        console.print_json(
            json_mod.dumps(payload if len(payload) > 1 else payload[0], ensure_ascii=False)
        )

    _print_import_summary(transcripts)

    if output:
        saved = _save_transcripts(transcripts, Path(output), output_format)
        console.print(f"\n[green]Saved {len(saved)} transcript(s):[/green]")
        for p in saved:
            console.print(f"  {p}")
        console.print(f"\n[dim]View with:[/dim] compass trace {saved[0]}")


_CLAUDE_WIRE_TYPES = frozenset(
    {"assistant", "user", "result", "system", "stream_event", "rate_limit_event"}
)


def _detect_trace_format(path: Path) -> str | None:
    """Sniff the trace format from a file (or directory) without full parsing.

    The line-oriented formats are told apart by their first-line invariant:
      - pi:     a ``{"type": "session", ...}`` session header
      - claude: a stream-json line whose ``type`` is a Claude Code message type
      - codex:  an event whose ``type`` is namespaced ``thread.``/``turn.``/``item.``
      - otlp:   anything else that is JSON (OTLP / OpenInference)

    ATIF is checked separately, because it is the one format that is normally
    *pretty-printed*: its first line is a bare ``{``, which parses as nothing and
    would otherwise fall through to the ``otlp`` catch-all.
    """
    import json as json_mod

    from compass.integrations import looks_like_codex_stream

    if path.is_dir():
        # A Harbor job dir holds <trial>/agent/trajectory.json; anything else
        # with a directory of traces in it is pi (import_pi_sessions globs it).
        for candidate in path.glob("**/trajectory.json"):
            if _looks_like_atif_file(candidate):
                return "atif"
        return "pi"

    first_line = ""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    first_line = line.strip()
                    break
    except OSError:
        return None

    try:
        obj = json_mod.loads(first_line)
    except (ValueError, TypeError):
        obj = None
    if isinstance(obj, dict):
        if obj.get("type") == "session":
            return "pi"
        if obj.get("type") in _CLAUDE_WIRE_TYPES:
            return "claude"
        if looks_like_codex_stream(obj):
            return "codex"
    if _looks_like_atif_file(path, first_line=first_line, parsed=obj):
        return "atif"
    if first_line[:1] in ("{", "["):
        return "otlp"
    return None


# `schema_version` is ATIF's first field, so it lands well inside the head of any
# machine-written trajectory, pretty-printed or not.
_ATIF_HEAD_BYTES = 8192


def _looks_like_atif_file(
    path: Path, *, first_line: str | None = None, parsed: Any = None
) -> bool:
    """Whether a file is an ATIF trajectory, without parsing all of it.

    A one-line document is settled by the parsed object. A pretty-printed one is
    settled by finding the ``schema_version`` declaration in the file's head.
    """
    import re

    from compass.integrations import looks_like_atif

    if parsed is not None and looks_like_atif(parsed):
        return True
    if first_line is not None and first_line[:1] not in ("{", ""):
        return False
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            head = f.read(_ATIF_HEAD_BYTES)
    except OSError:
        return False
    return bool(re.search(r'"schema_version"\s*:\s*"ATIF-', head))


def _print_import_summary(transcripts: list["Transcript"]) -> None:
    """Print a compact per-transcript summary table."""
    table = Table(show_header=True, header_style="bold")
    table.add_column("#", justify="right", width=3)
    table.add_column("Trial / Task", style="cyan")
    table.add_column("Calls", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Final output", overflow="fold")

    for i, t in enumerate(transcripts, 1):
        tokens = t.sum_tokens().total_tokens
        cost = t.sum_cost().total_usd
        final = ""
        if t.outcome and t.outcome.output_data:
            fo = t.outcome.output_data.get("final_output")
            if isinstance(fo, str):
                final = fo[:60] + ("…" if len(fo) > 60 else "")
        table.add_row(
            str(i),
            f"{t.trial_id or '-'}\n[dim]{t.task_id}[/dim]",
            str(len(t.tool_calls)),
            str(tokens) if tokens else "-",
            f"${cost:.4f}" if cost else "-",
            final or "[dim](none)[/dim]",
        )
    console.print(table)


def _save_transcripts(
    transcripts: list["Transcript"], output: Path, output_format: str
) -> list[Any]:
    """Save transcript(s) to a file or directory; return the written paths."""
    import re

    def _write(t: "Transcript", path: Path) -> None:
        if output_format == "jsonl":
            t.save_jsonl(path)
        else:
            t.save(path)

    # Single transcript to an explicit file target.
    if len(transcripts) == 1 and output.suffix in (".json", ".jsonl"):
        _write(transcripts[0], output)
        return [output]

    # Otherwise treat the target as a directory, one file per transcript.
    ext = ".jsonl" if output_format == "jsonl" else ".json"
    output.mkdir(parents=True, exist_ok=True)
    saved = []
    for t in transcripts:
        stem = re.sub(r"[^A-Za-z0-9._-]", "_", t.trial_id or t.task_id or "transcript")[:100]
        path = output / f"{stem or 'transcript'}{ext}"
        _write(t, path)
        saved.append(path)
    return saved
