"""
cli.py — vstash command line interface.

Commands:
  vstash add <file/url>   → ingest document
  vstash ask "<query>"    → single question
  vstash chat             → interactive mode
  vstash list             → show ingested documents
  vstash stats            → memory statistics
  vstash forget <file>    → remove document
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
from .embed import embed_query, get_embedding_dim
from .ingest import ingest, ingest_directory
from .store import VstashStore

app = typer.Typer(
    name="vstash",
    help="Local document memory with instant semantic search.",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


def _get_store(cfg: VstashConfig | None = None) -> tuple[VstashConfig, VstashStore]:
    """Initialize config and create a VstashStore instance.

    Args:
        cfg: Optional pre-loaded config. If None, loads from vstash.toml.

    Returns:
        Tuple of (config, store).
    """
    if cfg is None:
        cfg = load_config()
    dim = get_embedding_dim(cfg.embeddings.model)
    store = VstashStore(cfg.storage.db_path, embedding_dim=dim)
    return cfg, store


# ------------------------------------------------------------------ #
# vstash add                                                          #
# ------------------------------------------------------------------ #


@app.command()
def add(
    sources: list[str] = typer.Argument(..., help="Files, directories, or URLs to ingest"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-ingest even if already in memory"),
) -> None:
    """Add documents or URLs to memory."""
    cfg, store = _get_store()

    with store:
        for source in sources:
            path = Path(source)

            # Directory
            if path.is_dir():
                console.print(f"\n[bold]Scanning[/bold] {source}")
                results = ingest_directory(source, cfg, store, force=force)
                ok = [r for r in results if r.status == "ok"]
                skipped = [r for r in results if r.status == "skipped"]
                msg = f"\n[green]✓[/green] {len(ok)}/{len(results)} files ingested from [bold]{source}[/bold]"
                if skipped:
                    msg += f" [dim]({len(skipped)} skipped, use --force to re-ingest)[/dim]"
                console.print(msg)
                continue

            # File or URL
            source_str = str(path.resolve()) if path.exists() else source
            result = ingest(source_str, cfg, store, force=force)

            if result.status == "ok":
                console.print(
                    f"[green]✓[/green] [bold]{result.title}[/bold] — "
                    f"{result.chunks} chunks in {result.elapsed_s}s"
                )
            elif result.status == "skipped":
                console.print(
                    f"[dim]⊘ {source} already in memory (use --force to re-ingest)[/dim]"
                )
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
    top_k: int = typer.Option(0, "--top-k", "-k", help="Number of chunks to retrieve (0 = from config)"),
    sources: bool = typer.Option(True, "--sources/--no-sources", help="Show source citations"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream the response"),
) -> None:
    """Ask a question about your documents."""
    cfg, store = _get_store()

    with store:
        k = top_k or cfg.chunking.top_k

        # Embed query
        with console.status("[dim]Searching memory...[/dim]", spinner="dots"):
            q_embedding = embed_query(query, cfg.embeddings.model)
            chunks = store.search(q_embedding, query, top_k=k)

        if not chunks:
            console.print(
                "[yellow]No relevant documents found. "
                "Try adding some with [bold]vstash add[/bold].[/yellow]"
            )
            raise typer.Exit()

        # Show sources
        if sources:
            source_list = list({c.title for c in chunks})
            console.print(f"\n[dim]Sources: {', '.join(source_list)}[/dim]")

        # Stream response
        console.print()
        if stream:
            try:
                for token in chat_module.stream(query, chunks, cfg):
                    print(token, end="", flush=True)
                print()  # newline after stream
            except ConnectionError as exc:
                console.print(f"\n[red]✗ Inference error: {exc}[/red]")
        else:
            with console.status("[dim]Thinking...[/dim]", spinner="dots"):
                try:
                    response = chat_module.ask(query, chunks, cfg)
                except ConnectionError as exc:
                    console.print(f"[red]✗ Inference error: {exc}[/red]")
                    raise typer.Exit(1) from exc
            console.print(Markdown(response))


# ------------------------------------------------------------------ #
# vstash chat                                                         #
# ------------------------------------------------------------------ #


@app.command()
def chat(
    top_k: int = typer.Option(0, "--top-k", "-k"),
) -> None:
    """Interactive chat mode. Type 'exit' or Ctrl+C to quit."""
    cfg, store = _get_store()

    with store:
        k = top_k or cfg.chunking.top_k

        console.print(Panel(
            "[bold cyan]vstash[/bold cyan] · Interactive mode\n"
            "[dim]Type your question. 'exit' to quit.[/dim]",
            border_style="cyan",
        ))

        history: list[dict[str, str]] = []

        try:
            while True:
                console.print()
                try:
                    query = console.input("[bold cyan]>[/bold cyan] ").strip()
                except (EOFError, KeyboardInterrupt):
                    break

                if not query:
                    continue
                if query.lower() in ("exit", "quit", "q"):
                    break

                # Search
                q_embedding = embed_query(query, cfg.embeddings.model)
                chunks = store.search(q_embedding, query, top_k=k)

                if not chunks:
                    console.print("[yellow]No relevant context found.[/yellow]")
                    continue

                source_list = list({c.title for c in chunks})
                console.print(f"[dim]Sources: {', '.join(source_list)}[/dim]\n")

                # Stream with history
                full_response = ""
                try:
                    for token in chat_module.stream(query, chunks, cfg, history=history):
                        print(token, end="", flush=True)
                        full_response += token
                    print()
                except ConnectionError as exc:
                    console.print(f"\n[red]✗ Inference error: {exc}[/red]")
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
def list_docs() -> None:
    """List all documents in memory."""
    cfg, store = _get_store()

    with store:
        docs = store.list_documents()

        if not docs:
            console.print(
                "[yellow]Memory is empty. Add documents with [bold]vstash add[/bold].[/yellow]"
            )
            return

        table = Table(show_header=True, header_style="bold cyan", border_style="dim")
        table.add_column("Title", style="bold")
        table.add_column("Type", style="dim")
        table.add_column("Chunks", justify="right")
        table.add_column("Added", style="dim")

        for doc in docs:
            added = doc.added_at[:10]  # just the date
            table.add_row(
                doc.title,
                doc.source_type,
                str(doc.chunk_count),
                added,
            )

        console.print(table)


# ------------------------------------------------------------------ #
# vstash stats                                                        #
# ------------------------------------------------------------------ #


@app.command()
def stats() -> None:
    """Show memory statistics."""
    cfg, store = _get_store()

    with store:
        s = store.stats()

        console.print(Panel(
            f"[bold]Documents:[/bold] {s.documents}\n"
            f"[bold]Chunks:[/bold] {s.chunks}\n"
            f"[bold]Database:[/bold] {s.db_path}\n"
            f"[bold]Size:[/bold] {s.db_size_mb} MB\n"
            f"[bold]Backend:[/bold] {cfg.inference.backend} / {cfg.inference.model}\n"
            f"[bold]Embeddings:[/bold] {cfg.embeddings.model}",
            title="[bold cyan]vstash Memory[/bold cyan]",
            border_style="cyan",
        ))


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
    console.print(Panel(
        f"[bold]Inference backend:[/bold] {cfg.inference.backend}\n"
        f"[bold]Model:[/bold] {cfg.inference.model}\n"
        f"[bold]Embedding model:[/bold] {cfg.embeddings.model}\n"
        f"[bold]Chunk size:[/bold] {cfg.chunking.size} tokens\n"
        f"[bold]Chunk overlap:[/bold] {cfg.chunking.overlap} tokens\n"
        f"[bold]Top-k retrieval:[/bold] {cfg.chunking.top_k}\n"
        f"[bold]Database:[/bold] {cfg.storage.db_path}\n"
        f"[bold]Cerebras key:[/bold] {'set ✓' if cfg.cerebras_api_key else 'not set ✗'}\n"
        f"[bold]OpenAI key:[/bold] {'set ✓' if cfg.openai_api_key else 'not set ✗'}",
        title="[bold cyan]vstash Config[/bold cyan]",
        border_style="cyan",
    ))


# ------------------------------------------------------------------ #
# Entry point                                                         #
# ------------------------------------------------------------------ #


def main() -> None:
    """Main entry point for the vstash CLI."""
    app()


if __name__ == "__main__":
    main()
