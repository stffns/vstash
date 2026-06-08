"""``vstash ask`` / ``search`` / ``chat`` commands (#284).

The retrieval-facing CLI surface. Imported by ``vstash/cli/__init__.py`` so the
``@app.command()`` decorators register on the shared Typer ``app``.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .. import chat as chat_module
from ..embed import embed_query
from ..services import search_with_embedding
from ..store import relevance_tier
from ..validation import LimitError, validate_search_input
from ._app import (
    _get_store,
    _inference_hint,
    _profile_from_ctx,
    _record_miss_event,
    _safe_exc,
    app,
    console,
)


@app.command()
def ask(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="Your question"),
    top_k: int = typer.Option(
        0, "--top-k", "-k", help="Number of chunks to retrieve (0 = from config)"
    ),
    collection: str | None = typer.Option(
        None, "--collection", "-c", help="Restrict to collection"
    ),
    project: str | None = typer.Option(None, "--project", "-p", help="Restrict to project"),
    layer: str | None = typer.Option(None, "--layer", "-l", help="Restrict to layer"),
    sources: bool = typer.Option(True, "--sources/--no-sources", help="Show source citations"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream the response"),
    all_profiles: bool = typer.Option(
        False, "--all-profiles", "-A", help="Search across all profiles"
    ),
) -> None:
    """Ask a question about your documents."""
    cfg, store = _get_store(warm=True, profile=_profile_from_ctx(ctx))

    with store:
        k = top_k or cfg.chunking.top_k

        # Embed + search via the services layer for the single-store
        # path; federated still embeds once up front because
        # federated_search wants the embedding pre-computed.
        with console.status("[dim]Searching memory...[/dim]", spinner="dots"):
            if all_profiles:
                from ..profile import federated_search

                q_embedding = embed_query(query, cfg.embeddings.model)
                tagged = federated_search(
                    query_embedding=q_embedding,
                    query_text=query,
                    cfg=cfg,
                    top_k=k,
                    collection=collection,
                    project=project,
                    layer=layer,
                    expand_window=1,
                )
                chunks = [r for _, r in tagged]
            else:
                chunks = search_with_embedding(
                    cfg=cfg,
                    store=store,
                    query=query,
                    top_k=k,
                    expand_window=0,
                    collection=collection,
                    project=project,
                    layer=layer,
                )

        if not chunks:
            if not all_profiles:
                # Log the empty-result event with a miss_hint so
                # ``vstash why --recent`` can surface it later. Use
                # best_distance=1.0 (max) as the sentinel for "no
                # vector ever matched" -- federated mode has no single
                # best_distance and is skipped.
                _record_miss_event(
                    store=store,
                    cfg=cfg,
                    query=query,
                    best_distance=1.0,
                    tier="low",
                    result_count=0,
                    top_k_requested=k,
                )
            console.print(
                "[yellow]No relevant documents found. "
                "Try adding some with [bold]vstash add[/bold].[/yellow]"
            )
            raise typer.Exit()

        # Tiered relevance signal (skip for federated — no single best_distance)
        if not all_profiles:
            tier = relevance_tier(store.last_best_distance)
            _record_miss_event(
                store=store,
                cfg=cfg,
                query=query,
                best_distance=store.last_best_distance,
                tier=tier,
                result_count=len(chunks),
                top_k_requested=k,
            )
            if tier == "low":
                console.print(
                    "[dim]⚠ Low relevance — context may not match your question well.[/dim]"
                )
            elif tier == "medium":
                console.print("[dim]? Uncertain relevance — results may be tangential.[/dim]")

        # Expand context: include adjacent chunks for richer LLM context
        # For federated mode, expansion happens inside federated_search
        # (per-store, before stores are closed).
        if not all_profiles:
            chunks = store.expand_context(chunks, window=1)

        # Show sources
        if sources:
            source_list = list({c.title for c in chunks})
            console.print(f"\n[dim]Sources: {', '.join(source_list)}[/dim]")

        # Stream response
        console.print()
        if stream:
            try:
                token_count = 0
                for token in chat_module.stream(query, chunks, cfg):
                    print(token, end="", flush=True)
                    token_count += 1
                print()  # newline after stream
            except ConnectionError as exc:
                if token_count > 0:
                    console.print(
                        f"\n[yellow]⚠ Stream interrupted after {token_count} tokens.[/yellow]"
                    )
                console.print(f"[red]✗ Inference error: {_safe_exc(exc)}[/red]")
                hint = _inference_hint(exc, cfg)
                if hint:
                    console.print(f"[dim]  Hint: {hint}[/dim]")
                raise typer.Exit(1) from exc
        else:
            with console.status("[dim]Thinking...[/dim]", spinner="dots"):
                try:
                    response = chat_module.ask(query, chunks, cfg)
                except ConnectionError as exc:
                    console.print(f"[red]✗ Inference error: {_safe_exc(exc)}[/red]")
                    hint = _inference_hint(exc, cfg)
                    if hint:
                        console.print(f"[dim]  Hint: {hint}[/dim]")
                    raise typer.Exit(1) from exc
            console.print(Markdown(response))


# ------------------------------------------------------------------ #
# vstash search                                                       #
# ------------------------------------------------------------------ #


@app.command()
def search(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="Your search query"),
    top_k: int = typer.Option(
        0, "--top-k", "-k", help="Number of chunks to retrieve (0 = from config)"
    ),
    collection: str | None = typer.Option(
        None, "--collection", "-c", help="Restrict to collection"
    ),
    project: str | None = typer.Option(None, "--project", "-p", help="Restrict to project"),
    layer: str | None = typer.Option(None, "--layer", "-l", help="Restrict to layer"),
    tag: list[str] | None = typer.Option(
        None,
        "--tag",
        "-t",
        help=(
            "Restrict to documents tagged with this value. Repeat the flag "
            "for OR semantics (``--tag alpha --tag beta``), or pass a "
            "comma-separated string in a single flag (``--tag 'alpha,beta'``). "
            "Comma-anchored match -- ``alpha`` does NOT false-match ``alphabet``."
        ),
    ),
    added_after: str | None = typer.Option(
        None,
        "--after",
        help="ISO date filter: only documents added on or after this date (e.g. 2026-01-15).",
    ),
    added_before: str | None = typer.Option(
        None,
        "--before",
        help="ISO date filter: only documents added strictly before this date.",
    ),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output results as JSON"),
    all_profiles: bool = typer.Option(
        False, "--all-profiles", "-A", help="Search across all profiles"
    ),
    explain: bool = typer.Option(
        False, "--explain", "-E", help="Show why each chunk ranked where it did"
    ),
    miss: str | None = typer.Option(
        None,
        "--miss",
        help="Diagnose why this expected document path did not appear in results",
    ),
    miss_chunk: int | None = typer.Option(
        None,
        "--miss-chunk",
        help="Diagnose why this expected chunk id did not appear in results",
    ),
    exact_match: str | None = typer.Option(
        None,
        "--exact-match",
        help="Post-filter: each returned chunk's text must contain this "
        "literal substring. Bypasses FTS5 tokenization so punctuation / "
        "identifiers / code survive verbatim. Case-insensitive by default "
        "-- pair with --exact-match-case-sensitive for a strict compare.",
    ),
    exact_match_case_sensitive: bool = typer.Option(
        False,
        "--exact-match-case-sensitive/--no-exact-match-case-sensitive",
        help="Toggle case sensitivity of --exact-match.",
    ),
) -> None:
    """Semantic search without LLM (free, local)."""
    cfg, store = _get_store(warm=True, profile=_profile_from_ctx(ctx))
    _search_tagged: list[tuple[str, object]] | None = None

    # Miss analysis branch — short-circuits the normal search flow.
    # Exit code conventions for scripting:
    #   0 — expected document IS in top-k (no miss)
    #   1 — argument error (missing args, invalid path, etc.)
    #   2 — expected document did NOT appear (analysis completed successfully,
    #       but the miss is real — lets CI / monitoring scripts alert)
    if miss is not None or miss_chunk is not None:
        import json as _json

        def _miss_error(msg: str, exit_code: int = 1) -> typer.Exit:
            """Emit an error consistently for both JSON and pretty output modes.

            Returns the Exit exception so the caller can ``raise`` it,
            which keeps ``raise ... from None`` chaining clean.
            """
            if json_output:
                print(_json.dumps({"error": msg}))
            else:
                console.print(f"[red]✗ {msg}[/red]")
            return typer.Exit(exit_code)

        if all_profiles:
            raise _miss_error("Miss analysis is not supported with --all-profiles.")
        with store:
            k = top_k or cfg.chunking.top_k
            # Normalize the path the same way add() does
            path_arg: str | None = None
            if miss is not None:
                if miss.startswith(("http://", "https://", "text://")):
                    path_arg = miss
                else:
                    path_arg = str(Path(miss).resolve(strict=False))
            try:
                with console.status("[dim]Analyzing miss...[/dim]", spinner="dots"):
                    q_embedding = embed_query(query, cfg.embeddings.model)
                    analysis = store.miss_analysis(
                        q_embedding,
                        query,
                        expected_path=path_arg,
                        expected_chunk_id=miss_chunk,
                        top_k=k,
                        collection=collection,
                        project=project,
                        layer=layer,
                    )
            except ValueError as exc:
                raise _miss_error(str(exc)) from None

        # JSON output path (honored for both the no-miss and real-miss cases)
        if json_output:
            print(_json.dumps(analysis.model_dump(), indent=2))
            raise typer.Exit(0 if analysis.appeared_in_results else 2)

        # Pretty print (Rich-styled)
        console.print()
        console.print(f"[bold]Query:[/bold] {query}")
        expected_label = analysis.expected_path or f"chunk #{analysis.expected_chunk_id}"
        console.print(f"[bold]Expected:[/bold] {expected_label}")
        if analysis.target_resolution == "best_of_n":
            console.print(
                f"[dim]  → tracing best-matching chunk of "
                f"{analysis.total_chunks_in_doc} in the document (bias warning: "
                f"other chunks may fail differently)[/dim]"
            )
        console.print(f"[bold]top_k:[/bold] {analysis.top_k_requested}")
        console.print()
        if analysis.appeared_in_results:
            console.print(
                f"[green]✓ Expected document IS in results at rank "
                f"{(analysis.final_rank or 0) + 1}. No miss to analyze.[/green]"
            )
            raise typer.Exit(0)
        console.print(
            f"[red]✗ Expected document did NOT appear. Dropped at: "
            f"[bold]{analysis.dropped_at}[/bold][/red]\n"
        )
        console.print("[bold cyan]Pipeline trace:[/bold cyan]")
        for v in analysis.stage_verdicts:
            mark = "[green]✓[/green]" if v.passed else "[red]✗[/red]"
            console.print(f"  {mark} [bold]{v.stage}[/bold]: {v.detail}")
            if v.counterfactual:
                console.print(f"     [dim]→ {v.counterfactual}[/dim]")
        console.print()
        console.print("[bold cyan]Suggestions:[/bold cyan]")
        for s in analysis.suggestions:
            console.print(f"  • {s}")
        console.print()
        if analysis.actual_top_k:
            console.print("[bold cyan]Actual top results (for comparison):[/bold cyan]")
            for r in analysis.actual_top_k:
                console.print(f"  {r.rank + 1}. {r.title} ({r.path}) — score {r.score:.4f}")
        raise typer.Exit(2)

    with store:
        k = top_k or cfg.chunking.top_k

        # Validate at the CLI boundary (parity with the SDK/MCP/web adapters,
        # which validate via services) so a pathological top_k or oversized query
        # raises a clean LimitError here instead of crashing deep inside SQLite.
        try:
            validate_search_input(
                query_text=query,
                top_k=k,
                distance_cutoff=cfg.limits.max_distance_cutoff,
                recency_boost=0.0,
                limits=cfg.limits,
            )
        except LimitError as exc:
            console.print(f"[red]✗[/red] {_safe_exc(exc)}")
            raise typer.Exit(code=2) from exc

        with console.status("[dim]Searching memory...[/dim]", spinner="dots"):
            q_embedding = embed_query(query, cfg.embeddings.model)

            if all_profiles:
                from ..profile import federated_search

                tagged = federated_search(
                    query_embedding=q_embedding,
                    query_text=query,
                    cfg=cfg,
                    top_k=k,
                    collection=collection,
                    project=project,
                    layer=layer,
                    tags=tag,
                    added_after=added_after,
                    added_before=added_before,
                    expand_window=1,
                )
                chunks = [r for _, r in tagged]
                _search_tagged = tagged
                # Post-filter federated results the same way VstashStore.search
                # does so --exact-match works across profiles too. #106.
                if exact_match:
                    if exact_match_case_sensitive:
                        _search_tagged = [
                            (name, r) for (name, r) in _search_tagged if exact_match in r.text
                        ]
                    else:
                        _needle = exact_match.casefold()
                        _search_tagged = [
                            (name, r)
                            for (name, r) in _search_tagged
                            if _needle in r.text.casefold()
                        ]
                    chunks = [r for _, r in _search_tagged]
            else:
                chunks = store.search(
                    q_embedding,
                    query,
                    top_k=k,
                    collection=collection,
                    project=project,
                    layer=layer,
                    tags=tag,
                    added_after=added_after,
                    added_before=added_before,
                    explain=explain,
                    exact_match=exact_match,
                    exact_match_case_sensitive=exact_match_case_sensitive,
                )

        if not chunks:
            if not all_profiles:
                _record_miss_event(
                    store=store,
                    cfg=cfg,
                    query=query,
                    best_distance=1.0,
                    tier="low",
                    result_count=0,
                    top_k_requested=k,
                )
            if json_output:
                print("[]")
                raise typer.Exit()
            console.print(
                "[yellow]No relevant documents found. "
                "Try adding some with [bold]vstash add[/bold].[/yellow]"
            )
            raise typer.Exit()

        # Relevance signal (skip for federated — no single best_distance)
        tier = "high"
        if not all_profiles:
            best_distance = store.last_best_distance
            tier = relevance_tier(best_distance)

            # Telemetry: record search event for discard rate analysis,
            # attaching a miss_hint when the tier is "low" (issue #157 part 3).
            _record_miss_event(
                store=store,
                cfg=cfg,
                query=query,
                best_distance=best_distance,
                tier=tier,
                result_count=len(chunks),
                top_k_requested=k,
            )

        if json_output:
            import json

            out: dict = {"chunks": [c.model_dump(exclude_none=True) for c in chunks]}
            if not all_profiles:
                out["relevance"] = tier
                out["best_distance"] = round(best_distance, 4)
            if _search_tagged:
                out["profiles"] = [name for name, _ in _search_tagged]
            print(json.dumps(out, indent=2))
            raise typer.Exit()

        if tier == "low":
            console.print("[dim]⚠ Low relevance — results may not match your query.[/dim]\n")

        # Normalize scores to [0, 1] for display
        scores = [c.score for c in chunks]
        max_score = max(scores) if scores else 1.0
        min_score = min(scores) if len(scores) > 1 else 0.0
        score_range = max_score - min_score

        table = Table(show_header=True, header_style="bold cyan", padding=(0, 1))
        table.add_column("#", style="dim", width=3)
        if all_profiles and _search_tagged:
            table.add_column("Profile", style="magenta", max_width=15)
        table.add_column("Score", width=6)
        table.add_column("Source", style="green", max_width=30)
        table.add_column("Text", max_width=80)

        for i, c in enumerate(chunks, 1):
            display_score = (c.score - min_score) / score_range if score_range > 0 else 1.0
            rank_label = f"{i}?" if tier == "medium" else str(i)
            text_preview = c.text.replace("\n", " ").strip()
            if len(text_preview) > 120:
                text_preview = text_preview[:120] + "..."
            if all_profiles and _search_tagged:
                profile_name = _search_tagged[i - 1][0] if i <= len(_search_tagged) else ""
                table.add_row(
                    rank_label,
                    profile_name,
                    f"{display_score:.2f}",
                    c.title,
                    text_preview,
                )
            else:
                table.add_row(
                    rank_label,
                    f"{display_score:.2f}",
                    c.title,
                    text_preview,
                )

        console.print(table)

        # --- Explain: diagnostic breakdown per chunk ---
        if explain:
            from rich.markup import escape

            console.print()
            for i, c in enumerate(chunks, 1):
                ex = c.explain
                if ex is None:
                    continue

                console.print(
                    f"[bold cyan]#{i}[/bold cyan] [bold]{escape(c.title)}[/bold] — {escape(c.path)}"
                )

                lines = []
                # Vector
                if ex.vec_rank is not None:
                    tier_label = (
                        relevance_tier(ex.vec_distance) if ex.vec_distance is not None else "?"
                    )
                    lines.append(
                        f"  Vector:  rank {ex.vec_rank + 1}, "
                        f"distance {ex.vec_distance} ({tier_label} relevance)"
                    )
                else:
                    lines.append("  Vector:  not in vector results (FTS-only match)")

                # FTS
                if ex.fts_rank is not None:
                    terms = ", ".join(escape(t) for t in ex.fts_terms) if ex.fts_terms else "n/a"
                    lines.append(f"  FTS:     rank {ex.fts_rank + 1}, terms \\[{terms}]")
                else:
                    lines.append("  FTS:     not in keyword results (vector-only match)")

                # RRF
                weight_info = ""
                if ex.rrf_vec_weight is not None:
                    weight_info = f" w={ex.rrf_vec_weight:.1f}/{ex.rrf_fts_weight:.1f}"
                lines.append(
                    f"  RRF:     {ex.rrf_total:.4f} "
                    f"(vec: {ex.rrf_vec:.4f} + fts: {ex.rrf_fts:.4f}){weight_info}"
                )

                # MMR
                if ex.mmr_penalty > 0:
                    lines.append(f"  MMR:     -{ex.mmr_penalty:.2f} penalty (same-doc similarity)")
                else:
                    lines.append("  MMR:     no penalty (unique document)")

                for line in lines:
                    console.print(f"[dim]{line}[/dim]")
                console.print()


# ------------------------------------------------------------------ #
# vstash chat                                                         #
# ------------------------------------------------------------------ #


@app.command()
def chat(
    ctx: typer.Context,
    top_k: int = typer.Option(0, "--top-k", "-k"),
) -> None:
    """Interactive chat mode. Type 'exit' or Ctrl+C to quit."""
    cfg, store = _get_store(warm=True, profile=_profile_from_ctx(ctx))

    with store:
        k = top_k or cfg.chunking.top_k

        console.print(
            Panel(
                "[bold cyan]vstash[/bold cyan] · Interactive mode\n"
                "[dim]Type your question. 'exit' to quit.[/dim]",
                border_style="cyan",
            )
        )

        history: list[dict[str, str]] = []
        import tiktoken

        _enc = tiktoken.get_encoding("cl100k_base")
        _MAX_HISTORY_TOKENS = 8192

        def _trim_history(hist: list[dict[str, str]]) -> list[dict[str, str]]:
            """Keep only the most recent turns that fit within the token budget."""
            total = 0
            trimmed: list[dict[str, str]] = []
            for msg in reversed(hist):
                msg_tokens = len(_enc.encode(msg["content"]))
                if total + msg_tokens > _MAX_HISTORY_TOKENS:
                    break
                trimmed.append(msg)
                total += msg_tokens
            return list(reversed(trimmed))

        # Telemetry: track whether user engages after non-high results
        last_event_id: int | None = None
        last_tier: str = "high"

        def _maybe_dismiss() -> None:
            """Mark the last search as dismissed if it was non-high relevance."""
            if last_event_id is not None and last_tier != "high":
                store.mark_search_dismissed(last_event_id)

        try:
            while True:
                console.print()
                try:
                    query = console.input("[bold cyan]>[/bold cyan] ").strip()
                except (EOFError, KeyboardInterrupt):
                    _maybe_dismiss()
                    break

                if not query:
                    continue
                if query.lower() in ("exit", "quit", "q"):
                    _maybe_dismiss()
                    break

                # Search via the services layer: embed + validate +
                # search + expand_context in one call. expand_window=1
                # gives the LLM adjacent context, same as before.
                chunks = search_with_embedding(
                    cfg=cfg,
                    store=store,
                    query=query,
                    top_k=k,
                    expand_window=1,
                )

                if not chunks:
                    console.print("[yellow]No relevant context found.[/yellow]")
                    continue

                # Tiered relevance signal (post-expand, mirroring the
                # MCP migration pattern -- semantically identical to
                # the old pre-expand placement).
                tier = relevance_tier(store.last_best_distance)
                last_event_id = store.record_search_event(
                    query=query,
                    best_distance=store.last_best_distance,
                    relevance_tier=tier,
                    result_count=len(chunks),
                )
                last_tier = tier
                if tier == "low":
                    console.print("[dim]⚠ Low relevance — context may not match well.[/dim]")
                elif tier == "medium":
                    console.print("[dim]? Uncertain relevance — results may be tangential.[/dim]")

                source_list = list({c.title for c in chunks})
                console.print(f"[dim]Sources: {', '.join(source_list)}[/dim]\n")

                # Stream with history (trimmed to token budget)
                trimmed = _trim_history(history)
                full_response = ""
                token_count = 0
                try:
                    for token in chat_module.stream(query, chunks, cfg, history=trimmed):
                        print(token, end="", flush=True)
                        full_response += token
                        token_count += 1
                    print()
                except ConnectionError as exc:
                    if token_count > 0:
                        console.print(
                            f"\n[yellow]⚠ Stream interrupted after {token_count} tokens.[/yellow]"
                        )
                    console.print(f"[red]✗ Inference error: {_safe_exc(exc)}[/red]")
                    hint = _inference_hint(exc, cfg)
                    if hint:
                        console.print(f"[dim]  Hint: {hint}[/dim]")
                    continue

                # Accumulate history for multi-turn context
                history.append({"role": "user", "content": query})
                history.append({"role": "assistant", "content": full_response})

        except KeyboardInterrupt:
            pass

        console.print("\n[dim]Goodbye.[/dim]")
