"""``vstash`` inspection commands (#284): list, export, stats.

Imported by ``vstash/cli/__init__.py`` so the ``@app.command()`` decorators
register on the shared Typer ``app``.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.panel import Panel
from rich.table import Table

from ..embed import embed_query
from ._app import _get_store, _profile_from_ctx, _safe_exc, app, console


@app.command(name="list")
def list_docs(
    ctx: typer.Context,
    collection: str | None = typer.Option(None, "--collection", "-c", help="Filter by collection"),
    project: str | None = typer.Option(None, "--project", "-p", help="Filter by project"),
    layer: str | None = typer.Option(None, "--layer", "-l", help="Filter by layer"),
) -> None:
    """List all documents in memory."""
    _cfg, store = _get_store(profile=_profile_from_ctx(ctx))

    with store:
        docs = store.list_documents(
            collection=collection,
            project=project,
            layer=layer,
        )

        if not docs:
            msg = (
                "Memory is empty."
                if not collection
                else f'No documents in collection "{collection}".'
            )
            console.print(f"[yellow]{msg} Add documents with [bold]vstash add[/bold].[/yellow]")
            return

        table = Table(show_header=True, header_style="bold cyan", border_style="dim")
        table.add_column("Title", style="bold")
        table.add_column("Collection", style="cyan")
        table.add_column("Type", style="dim")
        table.add_column("Chunks", justify="right")
        table.add_column("Added", style="dim")

        for doc in docs:
            added = doc.added_at[:10]  # just the date
            table.add_row(
                doc.title,
                doc.collection,
                doc.source_type,
                str(doc.chunk_count),
                added,
            )

        console.print(table)


# ------------------------------------------------------------------ #
# vstash export                                                       #
# ------------------------------------------------------------------ #


@app.command()
def export(
    ctx: typer.Context,
    output: str = typer.Option(..., "--output", "-o", help="Output file path"),
    fmt: str = typer.Option("jsonl", "--format", "-f", help="Output format: jsonl or csv"),
    collection: str | None = typer.Option(None, "--collection", "-c", help="Filter by collection"),
    project: str | None = typer.Option(None, "--project", "-p", help="Filter by project"),
    layer: str | None = typer.Option(None, "--layer", "-l", help="Filter by layer"),
    tags: str | None = typer.Option(None, "--tags", "-t", help="Filter by tag"),
    include_embeddings: bool = typer.Option(
        False, "--include-embeddings", help="Include embedding vectors"
    ),
) -> None:
    """Export chunks as JSONL or CSV for training data curation."""
    import csv

    _cfg, store = _get_store(profile=_profile_from_ctx(ctx))

    with store:
        chunks = store.export_chunks(
            collection=collection,
            project=project,
            layer=layer,
            tags=tags,
            include_embeddings=include_embeddings,
        )

    if not chunks:
        console.print("[yellow]No chunks match the given filters.[/yellow]")
        raise typer.Exit(1)

    out_path = Path(output)
    fmt = fmt.lower()

    if fmt not in ("jsonl", "csv"):
        console.print(f"[red]✗ Unsupported format '{fmt}'. Use 'jsonl' or 'csv'.[/red]")
        raise typer.Exit(1)

    if fmt == "csv":
        fieldnames = ["text", "title", "path", "chunk", "collection", "project", "layer", "tags"]
        if include_embeddings:
            fieldnames.append("embedding")
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for chunk in chunks:
                row = {}
                for k in fieldnames:
                    value = chunk.get(k, "")
                    if value is None:
                        value = ""
                    row[k] = value
                if include_embeddings and "embedding" in chunk:
                    row["embedding"] = json.dumps(chunk["embedding"])
                writer.writerow(row)
    else:  # jsonl
        with out_path.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    console.print(f"[green]✓[/green] Exported {len(chunks)} chunks → [bold]{output}[/bold] ({fmt})")


# ------------------------------------------------------------------ #
# vstash stats                                                        #
# ------------------------------------------------------------------ #


@app.command()
def stats(
    ctx: typer.Context,
    detailed: bool = typer.Option(
        False,
        "--detailed",
        "-d",
        help="Show the full observability metrics registry (counters, gauges, histograms)",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON (machine-readable, good for scrapers)",
    ),
) -> None:
    """Show memory statistics."""
    import json as _json

    from ..metrics import registry

    cfg, store = _get_store(profile=_profile_from_ctx(ctx))

    with store:
        s = store.stats()

        if json_output:
            payload = {
                "documents": s.documents,
                "chunks": s.chunks,
                "collections": s.collections,
                "db_path": s.db_path,
                "db_size_mb": s.db_size_mb,
                "inference_backend": cfg.inference.backend,
                "inference_model": cfg.inference.model,
                "embeddings_model": cfg.embeddings.model,
            }
            if detailed:
                payload["metrics"] = registry.snapshot()
            print(_json.dumps(payload, indent=2))
            return

        console.print(
            Panel(
                f"[bold]Documents:[/bold] {s.documents}\n"
                f"[bold]Chunks:[/bold] {s.chunks}\n"
                f"[bold]Collections:[/bold] {s.collections}\n"
                f"[bold]Database:[/bold] {s.db_path}\n"
                f"[bold]Size:[/bold] {s.db_size_mb} MB\n"
                f"[bold]Backend:[/bold] {cfg.inference.backend} / {cfg.inference.model}\n"
                f"[bold]Embeddings:[/bold] {cfg.embeddings.model}",
                title="[bold cyan]vstash Memory[/bold cyan]",
                border_style="cyan",
            )
        )

        if detailed:
            snap = registry.snapshot()
            console.print()
            # The metrics registry is in-memory and per-process.  A CLI
            # invocation creates a fresh process, so uptime is always
            # ~0s and counters/histograms reflect only this single
            # invocation.  Make that explicit so the reader does not
            # mistake the values for cumulative state — the long-running
            # picture lives behind ``vstash serve``'s /metrics endpoint.
            console.print(
                f"[bold cyan]Observability metrics[/bold cyan] "
                f"[dim](this CLI process only — uptime "
                f"{snap['uptime_seconds']:.1f}s; for cumulative state "
                f"see [bold]vstash serve[/bold]'s /metrics)[/dim]"
            )
            if snap["counters"]:
                console.print("\n[bold]Counters:[/bold]")
                for name, value in sorted(snap["counters"].items()):
                    console.print(f"  {name} = {value}")
            if snap["gauges"]:
                console.print("\n[bold]Gauges:[/bold]")
                for name, value in sorted(snap["gauges"].items()):
                    console.print(f"  {name} = {value}")
            if snap["histograms"]:
                console.print("\n[bold]Histograms:[/bold]")
                for name, h in sorted(snap["histograms"].items()):
                    console.print(
                        f"  {name}: count={h['count']} "
                        f"mean={h['mean_ms']:.1f}ms sum={h['sum_ms']:.1f}ms"
                    )


@app.command()
def why(
    ctx: typer.Context,
    query: str | None = typer.Argument(
        None,
        help="The search query that missed. Omit when using --recent.",
    ),
    recent: int = typer.Option(
        0,
        "--recent",
        "-r",
        help="Instead of running a new miss analysis, list the N most "
        "recent search events that carry an auto-logged miss_hint "
        "(empty / all-low results). 0 = disabled. Issue #157 part 3.",
    ),
    expect: str | None = typer.Option(
        None,
        "--expect",
        "-e",
        help="Path of the document you expected to see. Either this or "
        "--expect-chunk-id must be given.",
    ),
    expect_chunk_id: int | None = typer.Option(
        None,
        "--expect-chunk-id",
        help="Specific chunk id to track instead of resolving from a path. "
        "Useful when you already have a chunk id from an earlier search.",
    ),
    top_k: int = typer.Option(
        0, "--top-k", "-k", help="Top-k window to check against (0 = from config)"
    ),
    collection: str | None = typer.Option(
        None, "--collection", "-c", help="Restrict to collection"
    ),
    project: str | None = typer.Option(None, "--project", "-p", help="Restrict to project"),
    layer: str | None = typer.Option(None, "--layer", "-l", help="Restrict to layer"),
    json_out: bool = typer.Option(
        False,
        "--json",
        help="Emit the raw MissAnalysis as JSON (for scripting / piping). "
        "Default output is a rich table plus a suggestions panel.",
    ),
) -> None:
    """Diagnose why an expected document did not appear in search results.

    Runs ``VstashStore.miss_analysis()`` and prints a stage-by-stage trace
    showing where the expected chunk was eliminated from the ranking, plus
    rule-based suggestions for fixing the query.

    Examples:

        vstash why "rate limits" --expect notes/api-design.md

        vstash why "renewable energy" --expect papers/2024/clean.pdf --top-k 10

        vstash why "quota" --expect-chunk-id 4213 --json | jq .suggestions
    """
    import json as _json
    import sqlite3 as _sqlite3

    def _why_error(msg: str, exit_code: int = 1) -> typer.Exit:
        """Emit an error consistently for both --json and pretty modes.
        Mirrors the pattern in ``vstash search --miss`` so script
        consumers piping ``vstash why --json`` always see JSON, even on
        the error path. ``_safe_exc`` strips Rich markup so paths like
        ``foo[bar].md`` do not get mangled by the markup parser."""
        if json_out:
            print(_json.dumps({"error": msg}))
        else:
            console.print(f"[red]x[/red] {_safe_exc(msg)}")
        return typer.Exit(exit_code)

    # --recent mode: dump the N most recent miss_hint rows and exit.
    # Issue #157 part 3: surfaces recent miss hints stored in the DB
    # for this profile (the ``search_events`` table keeps the last
    # 1000 rows, including prior runs). Users see which queries
    # recently missed without having to remember them.
    if recent < 0:
        raise _why_error(f"--recent must be >= 0 (0 = disabled), got {recent}.")
    if recent > 0:
        _, store = _get_store(warm=False, profile=_profile_from_ctx(ctx))
        with store:
            hints = store.recent_miss_hints(limit=recent)
        if json_out:
            print(_json.dumps({"recent_miss_hints": hints}, indent=2, default=str))
            raise typer.Exit(0)
        if not hints:
            console.print(
                "[dim]No recent miss_hints recorded. "
                "Run a search that returns empty / all-low results, "
                "or verify [observability] auto_miss_hint is enabled.[/dim]"
            )
            raise typer.Exit(0)
        console.print(f"[bold]{len(hints)} recent miss hint(s)[/bold]")
        table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
        table.add_column("When", style="dim")
        table.add_column("Query", overflow="fold")
        table.add_column("Reason")
        table.add_column("Tier")
        table.add_column("Best dist", justify="right")
        table.add_column("Results", justify="right")
        for h in hints:
            reason = h["miss_hint"].get("reason", "-")
            table.add_row(
                h["created_at"][:19],  # truncate to seconds
                h["query"][:80],
                reason,
                h["relevance_tier"],
                f"{h['best_distance']:.4f}",
                str(h["result_count"]),
            )
        console.print(table)
        console.print()
        console.print(
            "[dim]Drill into any of these with:[/dim] "
            '[bold]vstash why "<query>" --expect <path>[/bold]'
        )
        raise typer.Exit(0)

    if query is None:
        raise _why_error("Missing argument QUERY. Pass a query or use --recent N.")

    if expect is None and expect_chunk_id is None:
        raise _why_error("Provide either --expect <path> or --expect-chunk-id <id>.")

    # ``why`` is a diagnostic tool; don't eagerly warm the embedder.
    # ``embed_query`` will load on demand if needed. Users running
    # ``vstash why`` against a misconfigured model get the actual error
    # instead of a cold-start failure that masks the diagnostic.
    cfg, store = _get_store(warm=False, profile=_profile_from_ctx(ctx))

    # Normalize --expect the same way ``vstash search --miss`` and
    # ``Memory.miss_analysis`` do: http(s)/text URIs pass through,
    # everything else resolves to an absolute path. Without this,
    # ``vstash why "q" --expect notes/doc.md`` would lookup the
    # literal relative string against a DB that stores absolute paths
    # and always report "No chunks found".
    expected_path_arg: str | None = None
    if expect is not None:
        if expect.startswith(("http://", "https://", "text://")):
            expected_path_arg = expect
        else:
            expected_path_arg = str(Path(expect).resolve(strict=False))

    effective_top_k = top_k or cfg.chunking.top_k

    with store:
        try:
            q_embedding = embed_query(query, cfg.embeddings.model)
            analysis = store.miss_analysis(
                query_embedding=q_embedding,
                query_text=query,
                expected_path=expected_path_arg,
                expected_chunk_id=expect_chunk_id,
                top_k=effective_top_k,
                collection=collection,
                project=project,
                layer=layer,
            )
        except ValueError as exc:
            # ValueError covers the documented miss_analysis failures
            # (unknown path / chunk id, no expected provided) plus
            # LimitError which subclasses ValueError.
            raise _why_error(str(exc)) from exc
        except _sqlite3.Error as exc:
            # Corrupt / locked DB surfaces here from the search() step
            # inside miss_analysis. Keep the user on the --json contract.
            raise _why_error(f"database error: {exc}") from exc
        except (RuntimeError, OSError) as exc:
            # embed_query failure modes: ONNX load failure, daemon
            # socket error, model download failure, etc.
            raise _why_error(f"embedder error: {exc}") from exc

    if json_out:
        # exclude_none strips empty-but-optional fields; indent=2 matches
        # the search --miss JSON contract. Exit code mirrors pretty-mode
        # (2 when dropped) so script consumers get reliable CI checks.
        print(_json.dumps(analysis.model_dump(exclude_none=True), indent=2, default=str))
        raise typer.Exit(0 if analysis.appeared_in_results else 2)

    # Header.
    # Ranks are 0-indexed internally; display 1-indexed to match ``vstash search``.
    if analysis.appeared_in_results:
        assert analysis.final_rank is not None
        console.print(
            f"[green]✓[/green] [bold]Expected doc IS in the top-{analysis.top_k_requested}[/bold] "
            f"(rank {analysis.final_rank + 1})."
        )
    else:
        console.print(
            f"[red]✗[/red] [bold]Expected doc NOT in top-{analysis.top_k_requested}.[/bold]"
        )
        if analysis.dropped_at:
            console.print(f"  Dropped at stage: [yellow bold]{analysis.dropped_at}[/yellow bold]")

    console.print()
    console.print(f"  [dim]Query:[/dim]          {analysis.query!r}")
    if analysis.expected_path:
        console.print(f"  [dim]Expected path:[/dim]  {analysis.expected_path}")
    if analysis.expected_chunk_id is not None:
        console.print(
            f"  [dim]Chunk id:[/dim]       {analysis.expected_chunk_id} "
            f"([dim]{analysis.target_resolution}[/dim])"
        )
    if analysis.total_chunks_in_doc > 1:
        console.print(f"  [dim]Doc has[/dim]         {analysis.total_chunks_in_doc} chunks")

    # Per-stage trace.
    console.print()
    console.print("[bold]Pipeline trace[/bold]")
    trace_table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    trace_table.add_column("Stage", style="cyan")
    trace_table.add_column("Passed")
    trace_table.add_column("Rank", justify="right")
    trace_table.add_column("Score", justify="right")
    trace_table.add_column("Detail", overflow="fold")
    for v in analysis.stage_verdicts:
        passed_cell = "[green]yes[/green]" if v.passed else "[red]no[/red]"
        drop_style = "red bold" if v.stage == analysis.dropped_at else ""
        stage_cell = f"[{drop_style}]{v.stage}[/{drop_style}]" if drop_style else v.stage
        trace_table.add_row(
            stage_cell,
            passed_cell,
            "-" if v.rank is None else str(v.rank + 1),
            "-" if v.score is None else f"{v.score:.4f}",
            v.detail,
        )
    console.print(trace_table)

    # Actual top-k for contrast.
    if analysis.actual_top_k:
        console.print()
        console.print("[bold]Actual top-k[/bold]")
        top_table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
        top_table.add_column("Rank", justify="right")
        top_table.add_column("Path", overflow="fold")
        top_table.add_column("Title", overflow="fold")
        top_table.add_column("Score", justify="right")
        for r in analysis.actual_top_k:
            top_table.add_row(str(r.rank + 1), r.path, r.title or "-", f"{r.score:.4f}")
        console.print(top_table)

    # Suggestions panel last so it lands near the user's prompt.
    if analysis.suggestions:
        console.print()
        console.print(
            Panel.fit(
                "\n".join(f"- {s}" for s in analysis.suggestions),
                title="[bold]Suggestions[/bold]",
                border_style="yellow",
            )
        )

    if not analysis.appeared_in_results:
        raise typer.Exit(code=2)
