"""``compass site`` — publish results as a shareable static site."""

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click
from rich.panel import Panel
from rich.table import Table

from compass.cli.app import cli, console
from compass.report.site import DEFAULT_HISTORY

if TYPE_CHECKING:
    pass


@cli.group()
def site() -> None:
    """Publish results as a shareable static site."""
    pass


@site.command("build")
@click.argument("results", type=click.Path(exists=True), nargs=-1, required=True)
@click.option("--output", "-o", type=click.Path(), default="site", show_default=True,
              help="Site directory; merged into if it already exists")
@click.option("--slug", help="Name to index this run under (default: derived from the results)")
@click.option("--name", help="Display name for the run (default: the slug)")
@click.option("--trace-dir", type=click.Path(exists=True),
              help="Publish these transcripts alongside the run (they carry prompts and outputs)")
@click.option("--include-details", is_flag=True,
              help="Publish grader details/metadata too (redacted by default)")
@click.option("--history", default=DEFAULT_HISTORY, show_default=True,
              help="Previous builds of this slug to remember for the trend line")
def site_build(
    results: tuple[str, ...],
    output: str,
    slug: str | None,
    name: str | None,
    trace_dir: str | None,
    include_details: bool,
    history: int,
) -> None:
    """Build (or refresh) a run in a static site.

    RESULTS is a results JSON written by `compass test --report json` or
    `compass grade -o` — the same shapes `analyze` and `compare` read.

    Only the named run is written; every other run already in the site is left
    alone. That is what lets separate repositories build into one shared
    directory (a gh-pages branch, a bucket prefix) and have the index
    accumulate — no server, no database.

    \b
      compass site build results.json -o site/
      compass site build results.json -o site/ --slug image-evals --trace-dir traces/
      python -m http.server -d site/          # browsers will not fetch from file://

    Grader details are left out unless --include-details: they are free-form
    and routinely hold model output. Transcripts are opt-in the same way —
    they are published only when --trace-dir names them.
    """
    import json

    from compass.report.site import build_site, collect_run_payload, slugify

    if slug and len(results) > 1:
        console.print("[red]--slug takes a single results file[/red]")
        sys.exit(1)

    built = []
    for path_str in results:
        path = Path(path_str)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            console.print(f"[red]Could not read {path}: {e}[/red]")
            sys.exit(1)

        run_slug = slugify(slug or path.stem)
        doc = collect_run_payload(payload, name=name or slug or path.stem)
        if not doc["run"]["total_cases"]:
            console.print(f"[yellow]No cases found in {path} — skipped[/yellow]")
            continue

        result = build_site(
            doc,
            output,
            slug=run_slug,
            trace_dir=trace_dir,
            include_details=include_details,
            history=history,
        )
        built.append(result)

    if not built:
        console.print("[red]Nothing to publish[/red]")
        sys.exit(1)

    last = built[-1]
    table = Table(title=f"Published to {last.site_dir}")
    table.add_column("Run", style="bold cyan")
    table.add_column("Cases", justify="right")
    table.add_column("Pass Rate", justify="right")
    table.add_column("Traces", justify="right")
    table.add_column("History", justify="right")
    # Only when something installed one — an image eval has no column to fill.
    show_skills = any(r.entry.get("skills") for r in built)
    if show_skills:
        table.add_column("Skill")
    for result in built:
        entry = result.entry
        row = [
            result.slug,
            f"{entry['passed_cases']}/{entry['evaluated_cases']}",
            f"{entry['pass_rate'] * 100:.1f}%",
            _format_bytes(result.trace_bytes) if result.trace_files else "—",
            str(result.history) if result.history else "—",
        ]
        if show_skills:
            row.append(
                ", ".join(
                    f"{s['name']}@{str(s.get('digest') or '')[:8]}"
                    for s in entry.get("skills") or []
                ) or "—"
            )
        table.add_row(*row)
    console.print(table)

    console.print(f"[green]{last.runs} run(s) in this site[/green]")
    if not include_details:
        console.print(
            "[dim]Grader details redacted — rebuild with --include-details to keep them.[/dim]"
        )
    if last.trace_files:
        console.print(
            f"[yellow]Published {last.trace_files} trace file(s): "
            f"transcripts carry prompts and model output.[/yellow]"
        )
    from collections import Counter

    from compass.report.redaction import summarize

    scrubbed: Counter[str] = Counter()
    for result in built:
        scrubbed.update(result.redactions)
    if scrubbed:
        # Worth a line of its own: whoever ran this has a key to rotate, and
        # whoever reads the page is looking at evidence with holes in it.
        console.print(f"[yellow]{summarize(scrubbed).capitalize()} before publishing.[/yellow]")
    console.print(f"[dim]Preview with:[/dim] python -m http.server -d {last.site_dir}")


@site.command("compare")
@click.argument("results_a", type=click.Path(exists=True))
@click.argument("results_b", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), default="site", show_default=True,
              help="Site directory; merged into if it already exists")
@click.option("--slug", help="Name to index this comparison under (default: from the filenames)")
@click.option("--name", help="Display name for the comparison (default: the slug)")
@click.option("--label-a", help="What to call the baseline run (default: its filename)")
@click.option("--label-b", help="What to call the candidate run (default: its filename)")
@click.option("--on", "on", multiple=True,
              help="Grader to scope to, repeatable (default: every grader both runs measured)")
@click.option("--metric", "metrics", multiple=True,
              help="Process metric to compare, repeatable (default: cost_usd, turns, tool_calls)")
@click.option("--missing-as-zero", is_flag=True,
              help="Treat an absent metric as a measured 0 — see `compass compare`")
def site_compare(
    results_a: str,
    results_b: str,
    output: str,
    slug: str | None,
    name: str | None,
    label_a: str | None,
    label_b: str | None,
    on: tuple[str, ...],
    metrics: tuple[str, ...],
    missing_as_zero: bool,
) -> None:
    """Publish a paired comparison of two runs into a static site.

    RESULTS_A is the baseline, RESULTS_B the candidate — the same inputs
    `compass compare` takes, and the same statistics. What the page adds is all
    of them at once: the overall score, then *every* grader both runs measured,
    then the process metrics.

    That per-grader table is not a nicety. Gate scores are excluded from the
    case score by design, so when correctness is decided by gates the overall
    comparison is measuring process cost and the correctness signal never
    reaches it.

    \b
      compass site build a.json -o site/ --slug baseline
      compass site build b.json -o site/ --slug candidate
      compass site compare a.json b.json -o site/ --slug baseline-vs-candidate
      python -m http.server -d site/

    Runs and comparisons accumulate in the same directory: publishing either
    one leaves the other alone.

    A comparison is only meaningful when both runs were graded the same way.
    Cases whose grading contract differs are counted and shown as *regraded* —
    their difference is not attributable to the agent.
    """
    from compass.report.site import build_comparison, collect_comparison, slugify

    path_a, path_b = Path(results_a), Path(results_b)
    comparison_slug = slugify(slug or f"{path_a.stem}-vs-{path_b.stem}")

    try:
        doc = collect_comparison(
            path_a,
            path_b,
            name=name or slug or comparison_slug,
            label_a=label_a or path_a.stem,
            label_b=label_b or path_b.stem,
            on=list(on) or None,
            metrics=list(metrics) or None,
            missing_as_zero=missing_as_zero,
        )
    except Exception as e:  # noqa: BLE001 — surfaced, not swallowed
        console.print(f"[red]Could not compare: {e}[/red]")
        sys.exit(1)

    body = doc["comparison"]
    if not body.get("n_paired"):
        console.print(
            "[red]No cases paired between the two runs[/red] — they share no case ids."
        )
        sys.exit(1)

    result = build_comparison(doc, output, slug=comparison_slug)

    table = Table(title=f"Published to {result.site_dir}")
    table.add_column("Comparison", style="bold cyan")
    table.add_column("Paired", justify="right")
    table.add_column("Graders", justify="right")
    table.add_column("Metrics", justify="right")
    table.add_column("Flips", justify="right")
    table.add_row(
        result.slug,
        str(body["n_paired"]),
        str(len(doc["scoped"])),
        str(len(doc["metrics"])),
        f"+{len(body['improved'])}/-{len(body['regressed'])}",
    )
    console.print(table)
    console.print(f"[dim]{body['verdict']}[/dim]")

    if body.get("regraded"):
        console.print(
            f"[yellow]⚠ {len(body['regraded'])} case(s) were graded by a different "
            f"grader spec in B than in A — their diff is not attributable to the "
            f"agent:[/yellow] {', '.join(body['regraded'][:10])}"
        )
    for skip in doc["skipped"]:
        console.print(
            f"[yellow]Skipped {skip.get('on') or skip.get('metric')}:[/yellow] {skip['reason']}"
        )

    console.print(f"[green]{result.comparisons} comparison(s) in this site[/green]")
    console.print(f"[dim]Preview with:[/dim] python -m http.server -d {result.site_dir}")


@site.command("serve")
@click.argument("sources", type=click.Path(exists=True), nargs=-1, required=True)
@click.option("--port", "-p", default=7001, show_default=True, help="Port to listen on")
@click.option("--host", default="127.0.0.1", show_default=True, help="Address to bind")
@click.option("--slug", help="Name to serve a single results file under")
@click.option("--name", help="Display name for the run")
@click.option("--trace-dir", type=click.Path(exists=True),
              help="Serve these transcripts alongside the run")
@click.option("--include-details/--redact", "include_details", default=None,
              help="Publish grader details (default: on for localhost, off otherwise)")
def site_serve(
    sources: tuple[str, ...],
    port: int,
    host: str,
    slug: str | None,
    name: str | None,
    trace_dir: str | None,
    include_details: bool | None,
) -> None:
    """Serve results as a site, reading from disk on every request.

    SOURCES are results JSON files, directories of them (``*.json``, not
    recursive), or one already-built site directory.

    Nothing is written. Each response is recomputed from the files, and the
    page polls, so a run that is still being written updates as it goes —
    which is also the answer to "why can't I just open index.html": browsers
    will not fetch JSON from file://.

    \b
      compass site serve results.json
      compass site serve results/ -p 8000
      compass site serve site/                # an already-built site

    Grader details are served by default on localhost, where this is local
    debugging. Bind anywhere else and they are redacted unless you pass
    --include-details, because that is publishing.
    """
    from compass.report.site import (
        LiveSource,
        is_built_site,
        is_loopback,
        make_server,
        make_static_server,
        slugify,
    )

    loopback = is_loopback(host)
    if include_details is None:
        include_details = loopback

    paths = [Path(s) for s in sources]
    if len(paths) == 1 and is_built_site(paths[0]):
        server = make_static_server(paths[0], host=host, port=port)
        console.print(Panel(f"[bold]Serving built site[/bold]  {paths[0]}"))
    else:
        if (slug or trace_dir) and len(paths) > 1:
            console.print("[red]--slug and --trace-dir take a single source[/red]")
            sys.exit(1)

        files: list[Path] = []
        for path in paths:
            files.extend(sorted(path.glob("*.json")) if path.is_dir() else [path])
        if not files:
            console.print("[red]No results files found[/red]")
            sys.exit(1)
        if len(files) > 1 and (slug or trace_dir):
            console.print("[red]--slug and --trace-dir take a single results file[/red]")
            sys.exit(1)

        live = [
            LiveSource(
                slug=slugify(slug or path.stem),
                path=path,
                name=name or slug or path.stem,
                trace_dir=Path(trace_dir) if trace_dir else None,
            )
            for path in files
        ]
        server = make_server(live, host=host, port=port, include_details=include_details)

        table = Table(title="Serving live")
        table.add_column("Run", style="bold cyan")
        table.add_column("Source")
        for source in live:
            table.add_row(source.slug, str(source.path))
        console.print(table)

    shown = "localhost" if host in ("127.0.0.1", "::1", "") else host
    console.print(f"[green]http://{shown}:{server.server_port}[/green]  (Ctrl-C to stop)")
    if not loopback:
        console.print(
            f"[yellow]Bound to {host} — anyone who can reach this machine can read it.[/yellow]"
        )
    if not include_details:
        console.print("[dim]Grader details redacted — pass --include-details to serve them.[/dim]")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped.[/dim]")
    finally:
        server.server_close()


def _format_bytes(size: int) -> str:
    """Human-readable byte count — the site's weight has to be visible."""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f}{unit}" if unit == "B" else f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}GB"


if __name__ == "__main__":
    cli()
