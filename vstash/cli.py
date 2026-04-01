"""
cli.py — vstash command line interface.

Commands:
  vstash add <file/url>   → ingest document
  vstash remember "text"  → ingest text directly (no file needed)
  vstash search "<query>" → semantic search (no LLM, free)
  vstash ask "<query>"    → single question (requires LLM)
  vstash chat             → interactive mode
  vstash list             → show ingested documents
  vstash stats            → memory statistics
  vstash forget <file>    → remove document
  vstash reindex           → re-embed chunks with new model
  vstash config           → show current config
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from . import chat as chat_module
from .config import VstashConfig, load_config
from .embed import embed_query, get_embedding_dim, warmup
from .ingest import ingest, ingest_directory
from .store import VstashStore, relevance_tier

from . import __version__


def _inference_hint(exc: ConnectionError, cfg: VstashConfig) -> str:
    """Return a user-friendly hint based on the inference error and backend."""
    msg = str(exc).lower()
    backend = cfg.inference.backend.lower()

    if "connection refused" in msg or "connect" in msg:
        if backend == "ollama":
            return "Is Ollama running? Try: ollama serve"
        return f"Could not reach {backend} server. Check your connection."
    if "rate limit" in msg or "429" in msg:
        return "Rate limited — wait a moment and try again."
    if "api key" in msg or "unauthorized" in msg or "401" in msg:
        return f"Check your API key for {backend}."
    if "timeout" in msg or "timed out" in msg:
        return "Request timed out — the server may be overloaded."
    return ""


def _version_callback(value: bool) -> None:
    if value:
        print(f"vstash {__version__}")
        raise typer.Exit()


app = typer.Typer(
    name="vstash",
    help="Local document memory with instant semantic search.",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


@app.callback()
def _app_callback(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Local document memory with instant semantic search."""


def _get_store(
    cfg: VstashConfig | None = None,
    *,
    warm: bool = False,
) -> tuple[VstashConfig, VstashStore]:
    """Initialize config and create a VstashStore instance.

    Args:
        cfg: Optional pre-loaded config. If None, loads from vstash.toml.
        warm: If True, eagerly load the ONNX embedding model to
            eliminate cold-start latency on the first query.

    Returns:
        Tuple of (config, store).
    """
    if cfg is None:
        cfg = load_config()
    if warm:
        warmup(cfg.embeddings.model)
    dim = get_embedding_dim(cfg.embeddings.model)
    store = VstashStore(
        cfg.storage.db_path,
        embedding_dim=dim,
        vector_backend=cfg.storage.vector_backend,
        snapvec_bits=cfg.storage.snapvec_bits,
    )
    return cfg, store


# ------------------------------------------------------------------ #
# vstash add                                                          #
# ------------------------------------------------------------------ #


@app.command()
def add(
    sources: list[str] = typer.Argument(..., help="Files, directories, or URLs to ingest"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-ingest even if already in memory"),
    collection: str = typer.Option("default", "--collection", "-c", help="Collection to add to"),
    project: str | None = typer.Option(
        None, "--project", "-p", help="Project tag (overrides frontmatter)"
    ),
    layer: str | None = typer.Option(
        None, "--layer", "-l", help="Layer tag (overrides frontmatter)"
    ),
    tags: str | None = typer.Option(
        None, "--tags", "-t", help="Comma-separated tags (overrides frontmatter)"
    ),
) -> None:
    """Add documents or URLs to memory."""
    cfg, store = _get_store()
    meta = {"project": project, "layer": layer, "tags": tags}

    with store:
        for source in sources:
            path = Path(source)

            # Directory
            if path.is_dir():
                console.print(f"\n[bold]Scanning[/bold] {source}")
                results = ingest_directory(
                    source,
                    cfg,
                    store,
                    force=force,
                    collection=collection,
                    **meta,
                )
                ok = [r for r in results if r.status == "ok"]
                skipped = [r for r in results if r.status == "skipped"]
                msg = f"\n[green]✓[/green] {len(ok)}/{len(results)} files ingested from [bold]{source}[/bold]"
                if skipped:
                    msg += f" [dim]({len(skipped)} skipped, use --force to re-ingest)[/dim]"
                if collection != "default":
                    msg += f" [cyan]→ {collection}[/cyan]"
                console.print(msg)
                continue

            # File or URL
            source_str = str(path.resolve()) if path.exists() else source
            result = ingest(
                source_str,
                cfg,
                store,
                force=force,
                collection=collection,
                **meta,
            )

            if result.status == "ok":
                parts = (
                    f"[green]✓[/green] [bold]{result.title}[/bold] — "
                    f"{result.chunks} chunks in {result.elapsed_s}s"
                )
                if collection != "default":
                    parts += f" [cyan]→ {collection}[/cyan]"
                console.print(parts)
            elif result.status == "skipped":
                console.print(f"[dim]⊊ {source} already in memory (use --force to re-ingest)[/dim]")
            elif result.status == "empty":
                console.print(f"[yellow]⚠[/yellow] No content extracted from {source}")
            elif result.status == "error":
                console.print(f"[red]✗[/red] Error: {result.error}")


# ------------------------------------------------------------------ #
# vstash ask                                                          #
# ------------------------------------------------------------------ #


@app.command()
def ask(
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
) -> None:
    """Ask a question about your documents."""
    cfg, store = _get_store(warm=True)

    with store:
        k = top_k or cfg.chunking.top_k

        # Embed query
        with console.status("[dim]Searching memory...[/dim]", spinner="dots"):
            q_embedding = embed_query(query, cfg.embeddings.model)
            chunks = store.search(
                q_embedding,
                query,
                top_k=k,
                collection=collection,
                project=project,
                layer=layer,
                scoring=cfg.scoring,
            )

        if not chunks:
            console.print(
                "[yellow]No relevant documents found. "
                "Try adding some with [bold]vstash add[/bold].[/yellow]"
            )
            raise typer.Exit()

        # Tiered relevance signal
        tier = relevance_tier(store.last_best_distance)
        store.record_search_event(
            query=query,
            best_distance=store.last_best_distance,
            relevance_tier=tier,
            result_count=len(chunks),
        )
        if tier == "low":
            console.print("[dim]⚠ Low relevance — context may not match your question well.[/dim]")
        elif tier == "medium":
            console.print("[dim]? Uncertain relevance — results may be tangential.[/dim]")

        # Expand context: include adjacent chunks for richer LLM context
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
                console.print(f"[red]✗ Inference error: {exc}[/red]")
                hint = _inference_hint(exc, cfg)
                if hint:
                    console.print(f"[dim]  Hint: {hint}[/dim]")
                raise typer.Exit(1) from exc
        else:
            with console.status("[dim]Thinking...[/dim]", spinner="dots"):
                try:
                    response = chat_module.ask(query, chunks, cfg)
                except ConnectionError as exc:
                    console.print(f"[red]✗ Inference error: {exc}[/red]")
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
    query: str = typer.Argument(..., help="Your search query"),
    top_k: int = typer.Option(
        0, "--top-k", "-k", help="Number of chunks to retrieve (0 = from config)"
    ),
    collection: str | None = typer.Option(
        None, "--collection", "-c", help="Restrict to collection"
    ),
    project: str | None = typer.Option(None, "--project", "-p", help="Restrict to project"),
    layer: str | None = typer.Option(None, "--layer", "-l", help="Restrict to layer"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output results as JSON"),
) -> None:
    """Semantic search without LLM (free, local)."""
    cfg, store = _get_store(warm=True)

    with store:
        k = top_k or cfg.chunking.top_k

        with console.status("[dim]Searching memory...[/dim]", spinner="dots"):
            q_embedding = embed_query(query, cfg.embeddings.model)
            chunks = store.search(
                q_embedding,
                query,
                top_k=k,
                collection=collection,
                project=project,
                layer=layer,
                scoring=cfg.scoring,
            )

        if not chunks:
            if json_output:
                print("[]")
                raise typer.Exit()
            console.print(
                "[yellow]No relevant documents found. "
                "Try adding some with [bold]vstash add[/bold].[/yellow]"
            )
            raise typer.Exit()

        # Relevance signal: tiered ghost warning based on vector distance.
        # high (<=0.95): confident, no warning. medium (0.95-0.98): subtle ?.
        # low (>0.98): full warning. Works from day zero, no scoring needed.
        best_distance = store.last_best_distance
        tier = relevance_tier(best_distance)
        scoring_enabled = cfg.scoring is not None and cfg.scoring.enabled

        # Telemetry: record search event for discard rate analysis
        _event_id = store.record_search_event(
            query=query,
            best_distance=best_distance,
            relevance_tier=tier,
            result_count=len(chunks),
        )

        if json_output:
            import json

            out = {
                "chunks": [c.model_dump() for c in chunks],
                "relevance": tier,
                "best_distance": round(best_distance, 4),
            }
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
        table.add_column("Score", width=6)
        table.add_column("Source", style="green", max_width=30)
        table.add_column("Text", max_width=80)

        for i, c in enumerate(chunks, 1):
            display_score = (c.score - min_score) / score_range if score_range > 0 else 1.0
            # Ghost warning: medium tier gets a subtle ? next to the rank
            rank_label = f"{i}?" if tier == "medium" else str(i)
            text_preview = c.text.replace("\n", " ").strip()
            if len(text_preview) > 120:
                text_preview = text_preview[:120] + "..."
            table.add_row(
                rank_label,
                f"{display_score:.2f}",
                c.title,
                text_preview,
            )

        console.print(table)

        # Show scoring warm-up progress when scoring is disabled
        if not scoring_enabled:
            total_accesses = store.total_access_count()
            target = 500  # ~100 searches × 5 results
            if total_accesses >= target:
                console.print(
                    "\n[dim]💡 Scoring ready! You have enough usage history. "
                    "Enable in vstash.toml: [bold]scoring.enabled = true[/bold][/dim]"
                )
            elif total_accesses >= 50:
                # Show progress once user has done at least ~10 searches
                pct = min(100, int(total_accesses / target * 100))
                bar_filled = pct // 5  # 20-char bar
                bar = "█" * bar_filled + "░" * (20 - bar_filled)
                console.print(
                    f"\n[dim]Learning preferences: {bar} {pct}% ({total_accesses}/{target})[/dim]"
                )


# ------------------------------------------------------------------ #
# vstash chat                                                         #
# ------------------------------------------------------------------ #


@app.command()
def chat(
    top_k: int = typer.Option(0, "--top-k", "-k"),
) -> None:
    """Interactive chat mode. Type 'exit' or Ctrl+C to quit."""
    cfg, store = _get_store(warm=True)

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
        import tiktoken  # noqa: PLC0415 — lazy import, only needed for chat

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

                # Search
                q_embedding = embed_query(query, cfg.embeddings.model)
                chunks = store.search(q_embedding, query, top_k=k, scoring=cfg.scoring)

                if not chunks:
                    console.print("[yellow]No relevant context found.[/yellow]")
                    continue

                # Tiered relevance signal
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

                # Expand context with adjacent chunks
                chunks = store.expand_context(chunks, window=1)

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
                    console.print(f"[red]✗ Inference error: {exc}[/red]")
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


# ------------------------------------------------------------------ #
# vstash list                                                         #
# ------------------------------------------------------------------ #


@app.command(name="list")
def list_docs(
    collection: str | None = typer.Option(None, "--collection", "-c", help="Filter by collection"),
    project: str | None = typer.Option(None, "--project", "-p", help="Filter by project"),
    layer: str | None = typer.Option(None, "--layer", "-l", help="Filter by layer"),
) -> None:
    """List all documents in memory."""
    cfg, store = _get_store()

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
    import json

    cfg, store = _get_store()

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
def stats() -> None:
    """Show memory statistics."""
    cfg, store = _get_store()

    with store:
        s = store.stats()

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


# ------------------------------------------------------------------ #
# vstash forget                                                       #
# ------------------------------------------------------------------ #


@app.command()
def forget(
    path: str = typer.Argument(..., help="File path or URL to remove from memory"),
) -> None:
    """Remove a document from memory."""
    cfg, store = _get_store()

    with store:
        source = str(Path(path).resolve()) if Path(path).exists() else path
        deleted = store.delete_document(source)

        if deleted:
            console.print(f"[green]✓[/green] Removed [bold]{path}[/bold] from memory.")
        else:
            console.print(f"[yellow]Not found:[/yellow] {path}")


# ------------------------------------------------------------------ #
# vstash config                                                       #
# ------------------------------------------------------------------ #


@app.command(name="config")
def show_config() -> None:
    """Show current configuration."""
    cfg = load_config()
    from .embed import resolve_backend

    resolved = resolve_backend(cfg.embeddings.backend)
    console.print(
        Panel(
            f"[bold]Inference backend:[/bold] {cfg.inference.backend}\n"
            f"[bold]Model:[/bold] {cfg.inference.model}\n"
            f"[bold]Embedding model:[/bold] {cfg.embeddings.model}\n"
            f"[bold]Embedding backend:[/bold] {resolved} "
            f"({'Apple Silicon GPU' if resolved == 'mlx' else 'ONNX Runtime'})\n"
            f"[bold]Chunk size:[/bold] {cfg.chunking.size} tokens\n"
            f"[bold]Chunk overlap:[/bold] {cfg.chunking.overlap} tokens\n"
            f"[bold]Top-k retrieval:[/bold] {cfg.chunking.top_k}\n"
            f"[bold]Database:[/bold] {cfg.storage.db_path}\n"
            f"[bold]Cerebras key:[/bold] {'set ✓' if cfg.cerebras_api_key else 'not set ✗'}\n"
            f"[bold]OpenAI key:[/bold] {'set ✓' if cfg.openai_api_key else 'not set ✗'}",
            title="[bold cyan]vstash Config[/bold cyan]",
            border_style="cyan",
        )
    )


# ------------------------------------------------------------------ #
# vstash reindex                                                       #
# ------------------------------------------------------------------ #


@app.command()
def reindex(
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="New embedding model (default: use vstash.toml setting)",
    ),
    batch_size: int = typer.Option(256, "--batch-size", help="Chunks per embedding batch"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Re-embed all chunks with a different embedding model.

    Use this after changing the embedding model in vstash.toml
    (e.g., switching to a multilingual model). All existing vector
    embeddings are recomputed — text, FTS, and metadata are preserved.
    """
    from .embed import embed_texts, get_embedding_dim

    cfg = load_config()
    target_model = model or cfg.embeddings.model
    new_dim = get_embedding_dim(target_model)

    # Warm up to get accurate model name display
    warmup(target_model, cfg.embeddings.backend)

    # Open store with current dim (to read existing data)
    current_dim = get_embedding_dim(cfg.embeddings.model) if model else new_dim
    store = VstashStore(
        cfg.storage.db_path,
        embedding_dim=current_dim,
        vector_backend=cfg.storage.vector_backend,
        snapvec_bits=cfg.storage.snapvec_bits,
    )

    with store:
        total = store._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        if total == 0:
            console.print("[yellow]No chunks to reindex.[/yellow]")
            raise typer.Exit()

        console.print(
            f"[bold]Reindex plan:[/bold]\n"
            f"  Model:  {target_model}\n"
            f"  Dims:   {new_dim}\n"
            f"  Chunks: {total}\n"
            f"  Batch:  {batch_size}"
        )

        if not yes:
            typer.confirm("This will re-embed all chunks. Continue?", abort=True)

        from rich.progress import Progress

        with Progress(console=console) as progress:
            task = progress.add_task("[cyan]Re-embedding chunks...", total=total)

            def _progress(processed: int, _total: int) -> None:
                progress.update(task, completed=processed)

            def _embed_batch(texts: list[str]) -> list[list[float]]:
                return embed_texts(texts, target_model, cfg.embeddings.backend)

            count = store.reindex(
                embed_fn=_embed_batch,
                new_dim=new_dim,
                batch_size=batch_size,
                progress_cb=_progress,
            )

        console.print(
            f"[bold green]Done![/bold green] Re-embedded {count} chunks "
            f"with {target_model} ({new_dim} dims)"
        )


# ------------------------------------------------------------------ #
# vstash watch                                                         #
# ------------------------------------------------------------------ #


@app.command()
def watch(
    paths: list[str] = typer.Argument(..., help="Directories to watch"),
    collection: str = typer.Option(
        "default", "--collection", "-c", help="Collection for ingested files"
    ),
    ext: str | None = typer.Option(
        None, "--ext", help="Comma-separated extensions, e.g. .md,.txt,.py"
    ),
    debounce: float = typer.Option(2.0, "--debounce", help="Seconds to wait before re-ingesting"),
) -> None:
    """Watch directories for changes and auto-ingest files."""
    from .watch import start_watch

    cfg, store = _get_store()

    extensions: frozenset[str] | None = None
    if ext:
        extensions = frozenset(
            (part if part.startswith(".") else f".{part}")
            for raw in ext.split(",")
            if (part := raw.strip().lower())
        )

    with store:
        start_watch(
            paths,
            cfg,
            store,
            collection=collection,
            extensions=extensions,
            debounce_s=debounce,
        )


@app.command()
def remember(
    text: str | None = typer.Argument(None, help="Text to ingest (or pipe via stdin)"),
    title: str | None = typer.Option(
        None, "--title", "-t", help="Title for the document (auto-generated if omitted)"
    ),
    collection: str = typer.Option("default", "--collection", "-c", help="Collection to add to"),
    project: str | None = typer.Option(None, "--project", "-p", help="Project tag"),
    layer: str | None = typer.Option(None, "--layer", "-l", help="Layer tag"),
    tags: str | None = typer.Option(None, "--tags", help="Comma-separated tags"),
) -> None:
    """Ingest text directly — no file needed.

    Pass text as an argument, or pipe it via stdin:

        vstash remember "The API uses OAuth2 with PKCE" --title "auth-notes"
        echo "deployment steps..." | vstash remember --title "deploy"
        cat notes.md | vstash remember --title "meeting-notes" --project myproj
    """
    import sys

    # Read from argument or stdin
    if text is None:
        if sys.stdin.isatty():
            console.print("[red]✗ No text provided. Pass as argument or pipe via stdin.[/red]")
            raise typer.Exit(1)
        text = sys.stdin.read()

    if not text.strip():
        console.print("[yellow]⚠ Empty text, nothing to ingest.[/yellow]")
        raise typer.Exit(1)

    from .ingest import ingest_text

    cfg, store = _get_store()
    meta = {"project": project, "layer": layer, "tags": tags}

    with store:
        result = ingest_text(
            text,
            cfg,
            store,
            title=title,
            collection=collection,
            **meta,
        )

        if result.status == "ok":
            parts = (
                f"[green]✓[/green] [bold]{result.title}[/bold] — "
                f"{result.chunks} chunks in {result.elapsed_s}s"
            )
            if collection != "default":
                parts += f" [cyan]→ {collection}[/cyan]"
            console.print(parts)
        else:
            console.print(f"[yellow]⚠ {result.status}: {result.source}[/yellow]")


# ------------------------------------------------------------------ #
# Entry point                                                         #
# ------------------------------------------------------------------ #


def main() -> None:
    """Main entry point for the vstash CLI."""
    app()


if __name__ == "__main__":
    main()
