"""``vstash retrain`` and ``vstash retrain-multi`` commands (#284).

Eval-gated self-supervised fine-tuning of the embedding model. Imported by
``vstash/cli/__init__.py`` so the ``@app.command()`` decorators register on the
shared Typer ``app``.
"""

from __future__ import annotations

from pathlib import Path

import click
import typer
from rich.table import Table

from .._store_open import open_store_for_config
from ..store import VstashStore
from ._app import (
    _get_store,
    _load_labeled_queries_jsonl,
    _profile_from_ctx,
    app,
    console,
)


@app.command()
def retrain(
    ctx: typer.Context,
    output: str = typer.Option(
        "~/.vstash/models/retrained",
        "--output",
        "-o",
        help="Where to save the fine-tuned model",
    ),
    quick: bool = typer.Option(
        False, "--quick", help="Quick mode: 1 epoch, 1000 queries, higher LR"
    ),
    max_queries: int = typer.Option(
        5000, "--max-queries", help="Maximum pseudo-queries to generate"
    ),
    epochs: int = typer.Option(2, "--epochs", help="Training epochs"),
    lr: float = typer.Option(3e-6, "--lr", help="Learning rate"),
    batch_size: int = typer.Option(64, "--batch-size", help="Training batch size"),
    base_model: str | None = typer.Option(
        None, "--base-model", help="Base model to fine-tune (default: current config model)"
    ),
    min_gain: float = typer.Option(
        0.0,
        "--min-gain",
        help="Required NDCG@10 improvement over baseline (0.0 = no regression). "
        "Pass a negative value to always save.",
    ),
    no_eval: bool = typer.Option(
        False, "--no-eval", help="Skip eval gate entirely and save unconditionally"
    ),
    eval_fraction: float = typer.Option(
        0.15, "--eval-fraction", help="Fraction of corpus reserved for held-out eval"
    ),
    eval_noise_size: int = typer.Option(
        1000,
        "--eval-noise",
        help="Distractor chunks added to the eval index (higher = stricter eval)",
    ),
    synthesize: bool = typer.Option(
        False,
        "--synthesize-queries/--no-synthesize-queries",
        help="Use the configured LLM backend to generate short realistic "
        "queries for each training chunk (InPars-style). Closes the "
        "chunk-prefix vs natural-query distribution gap.",
    ),
    synth_n: int = typer.Option(
        2,
        "--synth-n",
        help="Queries synthesized per chunk when --synthesize-queries is on.",
    ),
    synth_cache: str | None = typer.Option(
        None,
        "--synth-cache",
        help="JSONL cache file for synthesized queries. Reusing the same "
        "path across runs avoids re-calling the LLM on unchanged chunks.",
    ),
    synth_model: str | None = typer.Option(
        None,
        "--synth-model",
        help="Model name override for synthesis (defaults to the configured "
        "inference backend's model).",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Deterministic seed. Controls held-out split, triple sampling, "
        "torch dropout, and DataLoader shuffle, so two runs with the same "
        "inputs and seed produce identical models.",
    ),
    training_queries: str | None = typer.Option(
        None,
        "--training-queries",
        help="Path to a JSONL file of labeled training queries (shape per "
        "line: {query, relevant_paths}). When set, routes through the "
        "labeled-batched miner (v5 recipe) instead of chunk-prefix "
        "pseudo-queries. H-R8 (2026-04-21).",
    ),
    eval_queries_path: str | None = typer.Option(
        None,
        "--eval-queries",
        help="Path to a JSONL file of labeled eval queries (same shape as "
        "--training-queries). When set, the eval gate scores baseline "
        "and fine-tuned models on these real queries instead of the "
        "internal chunk-prefix split, so the gate reflects the corpus "
        "you actually care about (BEIR qrels, search logs, manual "
        "annotations). Mutually exclusive with --no-eval. Ignores "
        "--eval-fraction (the held-out set is the file you provide).",
    ),
    training_pair_source: str = typer.Option(
        "auto",
        "--training-pair-source",
        help="Resolution policy when --training-queries is not set. "
        "'auto' (default, H-R8) reuses --eval-queries as training "
        "queries when present, falling back to chunk-prefix otherwise. "
        "'labeled' errors if no labels. 'prefix' forces chunk-prefix.",
        click_type=click.Choice(["auto", "labeled", "prefix"]),
    ),
    bulk_mine: bool = typer.Option(
        False,
        "--bulk-mine/--no-bulk-mine",
        help="Route chunk-prefix mining through the GPU-batched miner "
        "(retrain_batch.generate_triples_batched). 20-50x faster than "
        "the default per-query path on FiQA-sized corpora, at the cost "
        "of holding the full corpus in GPU memory for a moment. "
        "Ignored when labeled queries are used (labeled path is always batched).",
    ),
    bulk_mine_device: str | None = typer.Option(
        None,
        "--bulk-mine-device",
        help="Device override for --bulk-mine / the labeled-batched miner "
        "('cuda' or 'cpu'). Leave unset to auto-detect.",
    ),
) -> None:
    """Fine-tune the embedding model using your own data.

    Analyzes where vector and keyword search disagree on your corpus,
    generates training pairs from those disagreements, and fine-tunes
    the embedding model to better understand your data. No human labels
    needed.

    By default the fine-tuned candidate is evaluated honestly on a
    held-out slice of the corpus (reindexed with the new model) and
    only saved if NDCG@10 meets or exceeds ``--min-gain`` over the
    baseline. Use ``--no-eval`` to skip the gate.

    Requires: pip install sentence-transformers torch

    After training, use the model with:
        vstash reindex --model <output-path>
    """
    try:
        from ..retrain import retrain as run_retrain
    except ImportError as exc:
        console.print(
            "[red]x[/red] vstash retrain requires sentence-transformers, torch, "
            "and accelerate. Install with: "
            "[bold]pip install 'sentence-transformers>=3' torch 'accelerate>=1.1.0'[/bold]"
        )
        raise typer.Exit(code=1) from exc

    cfg, store = _get_store(profile=_profile_from_ctx(ctx))
    model_name = base_model or cfg.embeddings.model

    if quick:
        max_queries = min(max_queries, 1000)
        epochs = 1
        lr = 5e-6

    stats = store.stats()
    if stats.chunks < 10:
        console.print(
            "[yellow]! Your store has fewer than 10 chunks. "
            "Add more documents before retraining.[/yellow]"
        )
        raise typer.Exit(code=1)

    console.print("[bold cyan]vstash retrain[/bold cyan]")
    console.print(f"  Store:        {stats.documents} docs, {stats.chunks} chunks")
    console.print(f"  Base model:   {model_name}")
    console.print(f"  Max queries:  {max_queries}")
    if synthesize:
        console.print(
            f"  Query source: [cyan]LLM-synthesized[/cyan] "
            f"(n={synth_n}" + (f", cache={synth_cache}" if synth_cache else "") + ")"
        )
    else:
        console.print("  Query source: chunk prefix (legacy)")
    if no_eval:
        console.print("  Eval gate:    [yellow]disabled[/yellow]")
    else:
        console.print(
            f"  Eval gate:    min-gain={min_gain:+.4f}, "
            f"fraction={eval_fraction:.2f}, noise={eval_noise_size}"
        )
    console.print()

    loaded_training_queries: list[dict] | None = None
    if training_queries:
        loaded_training_queries = _load_labeled_queries_jsonl(
            training_queries, flag="--training-queries"
        )
        console.print(
            f"  Training queries: [cyan]{len(loaded_training_queries)} labeled[/cyan] "
            f"from {training_queries}"
        )

    loaded_eval_queries: list[dict] | None = None
    if eval_queries_path:
        if no_eval:
            console.print(
                "[red]x[/red] --eval-queries and --no-eval are mutually exclusive. Drop one."
            )
            raise typer.Exit(code=1)
        loaded_eval_queries = _load_labeled_queries_jsonl(eval_queries_path, flag="--eval-queries")
        # Warn when the eval queries reference paths absent from the
        # store.  The retrain pipeline silently scores those queries as
        # 0 (no relevant chunk to retrieve), which would invalidate
        # NDCG@10 without surfacing a single error.
        #
        # Query only the paths the user actually referenced -- a full
        # ``SELECT path FROM documents`` would pull the entire corpus
        # into memory on stores with millions of docs (MS MARCO scale,
        # production search logs).  Eval sets are typically <10k unique
        # paths so a chunked ``WHERE path IN (...)`` is O(eval_size) on
        # the indexed ``documents.path`` instead of O(corpus_size).
        eval_paths = sorted({p for q in loaded_eval_queries for p in q["relevant_paths"]})
        existing_paths: set[str] = set()
        # 500 keeps the IN-list well below SQLite's default
        # ``SQLITE_LIMIT_VARIABLE_NUMBER`` (999) with headroom.
        chunk_size = 500
        for i in range(0, len(eval_paths), chunk_size):
            chunk = eval_paths[i : i + chunk_size]
            placeholders = ",".join("?" * len(chunk))
            existing_paths.update(
                row[0]
                for row in store._conn.execute(
                    f"SELECT path FROM documents WHERE path IN ({placeholders})", chunk
                ).fetchall()
            )
        missing = sorted(set(eval_paths) - existing_paths)
        if missing:
            preview = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
            console.print(
                f"[yellow]! --eval-queries: {len(missing)} relevant_paths reference "
                f"docs not in this store ({preview}). Those queries will score 0; "
                "ingest the missing docs first or remove them from the eval file.[/yellow]"
            )
        console.print(
            f"  Eval queries:     [cyan]{len(loaded_eval_queries)} labeled[/cyan] "
            f"from {eval_queries_path} (gate uses these instead of internal split)"
        )

    result = run_retrain(
        store,
        base_model=model_name,
        output_path=output,
        max_queries=max_queries,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        eval_fraction=eval_fraction,
        eval_noise_size=eval_noise_size,
        min_gain=min_gain,
        skip_eval=no_eval,
        seed=seed,
        eval_queries=loaded_eval_queries,
        synthesize_queries=synthesize,
        synth_n=synth_n,
        synth_cache=synth_cache,
        synth_model=synth_model,
        training_queries=loaded_training_queries,
        training_pair_source=training_pair_source,  # type: ignore[arg-type]
        bulk_mine=bulk_mine,
        bulk_mine_device=bulk_mine_device,
        cfg=cfg,
    )

    if result.n_pairs == 0:
        console.print(
            "[yellow]! No training pairs generated. "
            "Your corpus may be too small or too homogeneous.[/yellow]"
        )
        raise typer.Exit(code=1)

    console.print(f"  Training pairs: {result.n_pairs}")

    if result.baseline is not None and result.final is not None:
        console.print()
        console.print("[bold]Eval results[/bold]")
        console.print(f"  Queries:          {result.baseline.n_queries}")
        console.print(f"  Baseline NDCG@10: {result.baseline.ndcg_at_10:.4f}")
        console.print(f"  Final NDCG@10:    {result.final.ndcg_at_10:.4f}")
        delta = result.delta_ndcg
        color = "green" if delta >= 0 else "red"
        console.print(f"  Delta NDCG@10:    [{color}]{delta:+.4f}[/{color}]")
        console.print(f"  Baseline NDCG@3:  {result.baseline.ndcg_at_3:.4f}")
        console.print(f"  Final NDCG@3:     {result.final.ndcg_at_3:.4f}")
        console.print(f"  Baseline MRR:     {result.baseline.mrr:.4f}")
        console.print(f"  Final MRR:        {result.final.mrr:.4f}")
        console.print(f"  Baseline Recall@100: {result.baseline.recall_at_100:.4f}")
        console.print(f"  Final Recall@100:    {result.final.recall_at_100:.4f}")

    if result.gated_out:
        console.print()
        if result.final is None:
            # No candidate was trained (too few pairs). Nothing to promote
            # or inspect -- the retrain simply did not produce a model.
            console.print(
                "[red]Training skipped[/red]: not enough training pairs to "
                "fine-tune. Your corpus may be too small or too homogeneous."
            )
        else:
            console.print(
                f"[red]Gated out[/red]: delta NDCG@10 did not meet min-gain "
                f"({result.min_gain:+.4f}). Candidate left at "
                f"[dim]{Path(output).expanduser()}.candidate[/dim] for inspection."
            )
        raise typer.Exit(code=2)

    if result.output_path is None:
        # Defensive: any future RetrainResult path that returns None for
        # output_path without setting gated_out=True would land here.
        console.print("[red]x[/red] retrain did not save a model. See logs for details.")
        raise typer.Exit(code=2)

    console.print()
    console.print(f"[green]Model saved to:[/green] {result.output_path}")
    console.print()
    console.print("To use the retrained model:")
    console.print(f"  [bold]vstash reindex --model {result.output_path}[/bold]")


@app.command(name="retrain-multi")
def retrain_multi_cmd(
    ctx: typer.Context,
    store_spec: list[str] = typer.Option(
        [],
        "--store",
        "-s",
        help="Extra corpus in the form NAME=PATH. Repeat for each dataset. "
        "The current profile's store is also included, aliased by its "
        "config name (or 'primary' if unnamed) unless --exclude-primary is passed.",
    ),
    exclude_primary: bool = typer.Option(
        False,
        "--exclude-primary",
        help="Do not include the current profile's store in the multi-corpus mix. "
        "Use this when --store alone covers all datasets you want to train on.",
    ),
    output: str = typer.Option(
        "~/.vstash/models/retrained-multi",
        "--output",
        "-o",
        help="Where to save the fine-tuned model",
    ),
    sampling_strategy: str = typer.Option(
        "temperature",
        "--sampling-strategy",
        help="Temperature (default) damps the largest corpus toward a more balanced triple budget.",
        click_type=click.Choice(["uniform", "proportional", "temperature"]),
    ),
    sampling_temperature: float = typer.Option(
        0.5,
        "--sampling-temperature",
        help="Exponent for the temperature strategy. 0=uniform, 1=proportional, "
        "0.5=favours smaller corpora without abandoning size signal.",
    ),
    total_triples: int = typer.Option(
        10000,
        "--total-triples",
        help="Target total pseudo-query budget across all datasets. Actual "
        "pair count may be lower when corpora are smaller than their share.",
    ),
    epochs: int = typer.Option(2, "--epochs", help="Training epochs"),
    lr: float = typer.Option(3e-6, "--lr", help="Learning rate"),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        help="Training batch size. Default 32 keeps a 15 GB T4 in range when "
        "three corpora are live during eval; raise on bigger GPUs.",
    ),
    use_amp: bool = typer.Option(
        True,
        "--use-amp/--no-use-amp",
        help="Automatic mixed precision during training. Near-free and halves GPU memory on T4+.",
    ),
    max_seq_length: int | None = typer.Option(
        None,
        "--max-seq-length",
        help="Cap the encoder's max sequence length. Lower values (256, 128) "
        "further reduce memory when most chunks are short.",
    ),
    base_model: str | None = typer.Option(
        None, "--base-model", help="Base model to fine-tune (default: current config model)"
    ),
    min_gain: float = typer.Option(
        0.0,
        "--min-gain",
        help="Required NDCG@10 improvement over baseline. Applied to the "
        "macro-average unless --per-dataset-gate is passed.",
    ),
    per_dataset_gate: bool = typer.Option(
        False,
        "--per-dataset-gate",
        help="Require every dataset individually to clear --min-gain. "
        "Stricter than the default macro-average gate.",
    ),
    no_eval: bool = typer.Option(
        False, "--no-eval", help="Skip eval gate entirely and save unconditionally"
    ),
    eval_fraction: float = typer.Option(
        0.15, "--eval-fraction", help="Fraction of each corpus reserved for held-out eval"
    ),
    eval_noise_size: int = typer.Option(
        1000,
        "--eval-noise",
        help="Distractor chunks added to each eval index (higher = stricter eval)",
    ),
    bulk_mine: bool = typer.Option(
        False,
        "--bulk-mine/--no-bulk-mine",
        help="Route triple generation through the GPU-batched miner "
        "(retrain_batch.generate_triples_batched). 20-50x faster than "
        "the default per-query path on FiQA-sized corpora, at the cost "
        "of holding the full corpus in GPU memory for a moment.",
    ),
    bulk_eval: bool = typer.Option(
        False,
        "--bulk-eval/--no-bulk-eval",
        help="Route baseline + final eval through the GPU-batched "
        "evaluator (retrain_batch.evaluate_model_batched). Biggest win "
        "when --eval-noise is large (e.g. >= 10k). Pair with --bulk-mine "
        "for end-to-end speedup on Colab T4.",
    ),
    bulk_mine_device: str | None = typer.Option(
        None,
        "--bulk-mine-device",
        help="Device override for --bulk-mine / --bulk-eval ('cuda' or "
        "'cpu'). Leave unset to auto-detect.",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Deterministic seed. Derived per-dataset via SHA-256 so each "
        "corpus gets its own stable RNG, and then threaded into train_mnrl "
        "so DataLoader shuffles and torch dropout are reproducible.",
    ),
    training_pair_source: str = typer.Option(
        "auto",
        "--training-pair-source",
        help="Resolution policy for per-dataset training queries. "
        "'auto' (default, H-R1) reuses eval labels as training queries "
        "when present, falling back to chunk-prefix otherwise. "
        "'labeled' errors if any dataset lacks labels. "
        "'prefix' forces chunk-prefix even when eval labels exist.",
        click_type=click.Choice(["auto", "labeled", "prefix"]),
    ),
) -> None:
    """Fine-tune the embedding model over multiple corpora with balanced sampling.

    Wrapper around ``vstash.retrain.retrain_multi``. Give it one or more
    vstash stores (each one a separate corpus) plus the current
    profile's store; retrain-multi computes a per-dataset triple budget,
    generates (query, positive, hard_neg) triples from each corpus,
    shuffles them globally, and fine-tunes a single embedding model.

    Per-dataset NDCG@10 is reported before and after. The macro-average
    gate (or --per-dataset-gate) prevents a regressed model from being
    promoted over your current one.

    Example:
        vstash retrain-multi \\
            --store nfcorpus=/data/vstash-nfcorpus.db \\
            --store fiqa=/data/vstash-fiqa.db \\
            --sampling-strategy temperature \\
            --sampling-temperature 0.5 \\
            --total-triples 30000 \\
            --output ~/.vstash/models/multi-tuned

    Requires: pip install sentence-transformers torch
    """
    try:
        from ..retrain import retrain_multi as run_retrain_multi
    except ImportError as exc:
        console.print(
            "[red]x[/red] vstash retrain-multi requires sentence-transformers, torch, "
            "and accelerate. Install with: "
            "[bold]pip install 'sentence-transformers>=3' torch 'accelerate>=1.1.0'[/bold]"
        )
        raise typer.Exit(code=1) from exc

    from ..embed import get_embedding_dim as _get_dim

    cfg, primary_store = _get_store(profile=_profile_from_ctx(ctx))
    model_name = base_model or cfg.embeddings.model

    # Nudge when --bulk-mine-device is set without either batched path.
    # The device override would otherwise be silently ignored.
    if bulk_mine_device and not (bulk_mine or bulk_eval):
        console.print(
            "[yellow]! --bulk-mine-device was passed without --bulk-mine or "
            "--bulk-eval. The device override is ignored; pass one of the "
            "--bulk-* flags to enable the GPU-batched path.[/yellow]"
        )

    stores: dict[str, VstashStore] = {}
    opened_extra: list[VstashStore] = []
    if not exclude_primary:
        primary_alias = _profile_from_ctx(ctx) or "primary"
        stores[primary_alias] = primary_store

    try:
        dim = _get_dim(cfg.embeddings.model)
        for spec in store_spec:
            if "=" not in spec:
                console.print(
                    f"[red]x[/red] Invalid --store spec [bold]{spec}[/bold]. Expected NAME=PATH."
                )
                raise typer.Exit(code=1)
            alias, _, path = spec.partition("=")
            alias = alias.strip()
            path = path.strip()
            if not alias or not path:
                console.print(
                    f"[red]x[/red] Invalid --store spec [bold]{spec}[/bold]. "
                    "Both NAME and PATH must be non-empty."
                )
                raise typer.Exit(code=1)
            if alias in stores:
                console.print(
                    f"[red]x[/red] Duplicate store alias [bold]{alias}[/bold]. "
                    "Each --store must use a unique name."
                )
                raise typer.Exit(code=1)
            extra_store = open_store_for_config(cfg, db_path=path, embedding_dim=dim)
            opened_extra.append(extra_store)
            stores[alias] = extra_store

        if not stores:
            console.print(
                "[red]x[/red] No stores to train on. "
                "Pass one or more --store NAME=PATH, or remove --exclude-primary."
            )
            raise typer.Exit(code=1)

        # Quick sanity check: each store should have at least a few chunks,
        # otherwise we print a hint rather than failing inside retrain_multi.
        sizes = {name: s.stats().chunks for name, s in stores.items()}
        if all(n < 10 for n in sizes.values()):
            console.print(
                "[yellow]! Every provided store has fewer than 10 chunks. "
                "Add more documents before retraining.[/yellow]"
            )
            raise typer.Exit(code=1)

        console.print("[bold cyan]vstash retrain-multi[/bold cyan]")
        for name, s in stores.items():
            stats = s.stats()
            console.print(f"  {name}: {stats.documents} docs, {stats.chunks} chunks")
        console.print(f"  Base model:       {model_name}")
        console.print(
            f"  Sampling:         {sampling_strategy} (temperature={sampling_temperature})"
        )
        console.print(f"  Total triples:    {total_triples}")
        if no_eval:
            console.print("  Eval gate:        [yellow]disabled[/yellow]")
        else:
            gate_mode = "per-dataset" if per_dataset_gate else "macro-average"
            console.print(
                f"  Eval gate:        {gate_mode}, min-gain={min_gain:+.4f}, "
                f"fraction={eval_fraction:.2f}, noise={eval_noise_size}"
            )
        console.print()

        result = run_retrain_multi(
            stores,
            base_model=model_name,
            output_path=output,
            sampling=sampling_strategy,  # type: ignore[arg-type]
            temperature=sampling_temperature,
            total_triples=total_triples,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            use_amp=use_amp,
            max_seq_length=max_seq_length,
            eval_fraction=eval_fraction,
            eval_noise_size=eval_noise_size,
            min_gain=min_gain,
            per_dataset_gate=per_dataset_gate,
            skip_eval=no_eval,
            seed=seed,
            bulk_mine=bulk_mine,
            bulk_mine_device=bulk_mine_device,
            bulk_eval=bulk_eval,
            training_pair_source=training_pair_source,  # type: ignore[arg-type]
            cfg=cfg,
        )
    finally:
        for s in opened_extra:
            try:
                s.close()
            except Exception:
                pass

    console.print(f"  Total training pairs: {result.total_pairs}")
    for name, n_pairs in result.per_dataset_pairs.items():
        budget_for_name = result.per_dataset_budget.get(name, 0)
        console.print(f"    {name}: {n_pairs} pairs (budget: {budget_for_name})")

    if result.per_dataset_baseline and result.per_dataset_final:
        console.print()
        console.print("[bold]Eval results (per dataset, NDCG@10)[/bold]")
        for name in result.per_dataset_baseline:
            base = result.per_dataset_baseline[name]
            final = result.per_dataset_final.get(name)
            if final is None or base.n_queries == 0:
                continue
            delta = final.ndcg_at_10 - base.ndcg_at_10
            color = "green" if delta >= 0 else "red"
            console.print(
                f"  {name:<20} baseline={base.ndcg_at_10:.4f}  "
                f"final={final.ndcg_at_10:.4f}  "
                f"delta=[{color}]{delta:+.4f}[/{color}]  (n={base.n_queries})"
            )
        macro_color = "green" if result.macro_delta_ndcg >= 0 else "red"
        console.print(
            f"  [bold]macro-avg            baseline={result.macro_baseline_ndcg:.4f}  "
            f"final={result.macro_final_ndcg:.4f}  "
            f"delta=[{macro_color}]{result.macro_delta_ndcg:+.4f}[/{macro_color}][/bold]"
        )

        console.print()
        console.print("[bold]Head quality + candidate health (per dataset)[/bold]")
        head_table = Table(show_header=True, header_style="bold cyan", border_style="dim")
        head_table.add_column("dataset", style="magenta")
        head_table.add_column("NDCG@3 base", justify="right")
        head_table.add_column("NDCG@3 final", justify="right")
        head_table.add_column("delta", justify="right")
        head_table.add_column("Recall@100 base", justify="right")
        head_table.add_column("Recall@100 final", justify="right")
        head_table.add_column("delta", justify="right")
        for name in result.per_dataset_baseline:
            base = result.per_dataset_baseline[name]
            final = result.per_dataset_final.get(name)
            if final is None or base.n_queries == 0:
                continue
            d3 = final.ndcg_at_3 - base.ndcg_at_3
            dr = final.recall_at_100 - base.recall_at_100
            c3 = "green" if d3 >= 0 else "red"
            cr = "green" if dr >= 0 else "red"
            head_table.add_row(
                name,
                f"{base.ndcg_at_3:.4f}",
                f"{final.ndcg_at_3:.4f}",
                f"[{c3}]{d3:+.4f}[/{c3}]",
                f"{base.recall_at_100:.4f}",
                f"{final.recall_at_100:.4f}",
                f"[{cr}]{dr:+.4f}[/{cr}]",
            )
        console.print(head_table)

    if result.gated_out:
        console.print()
        if result.total_pairs < 10:
            console.print(
                "[red]Training skipped[/red]: not enough training pairs across all corpora. "
                "Try a larger --total-triples or add more documents."
            )
        else:
            gate_mode = "per-dataset" if per_dataset_gate else "macro-average"
            console.print(
                f"[red]Gated out[/red] ({gate_mode}): NDCG@10 delta did not meet "
                f"min-gain ({result.min_gain:+.4f}). Candidate left at "
                f"[dim]{Path(output).expanduser()}.candidate[/dim] for inspection."
            )
        raise typer.Exit(code=2)

    if result.output_path is None:
        console.print("[red]x[/red] retrain-multi did not save a model. See logs for details.")
        raise typer.Exit(code=2)

    console.print()
    console.print(f"[green]Model saved to:[/green] {result.output_path}")
    console.print()
    console.print("To use the retrained model:")
    console.print(f"  [bold]vstash reindex --model {result.output_path}[/bold]")
