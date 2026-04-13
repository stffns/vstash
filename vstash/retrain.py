"""
retrain.py -- Self-supervised embedding fine-tuning from hybrid retrieval disagreement.

Generates (query, positive) pairs from vector/FTS signal disagreement
in the user's own corpus, then fine-tunes the embedding model using
MultipleNegativesRankingLoss (MNRL). The resulting model produces
embeddings that better distinguish "semantically close" from "actually
relevant" for the user's specific data.

Requires: pip install sentence-transformers torch
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path

from .embed import embed_query
from .store import VstashStore

logger = logging.getLogger(__name__)

TOP_K = 10


def generate_triples(
    store: VstashStore,
    model_name: str,
    max_queries: int = 5000,
    seed: int = 42,
) -> list[dict]:
    """Generate training triples from RRF signal disagreement.

    For each document chunk, uses it as a pseudo-query against the store.
    Identifies cases where vector-heavy and FTS-heavy search disagree on
    the top results, and builds (query, positive) pairs for MNRL training.

    Args:
        store: VstashStore with ingested documents.
        model_name: Embedding model to use for queries.
        max_queries: Maximum number of pseudo-queries to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of dicts with 'query' and 'positive' keys.
    """
    random.Random(seed)

    # Sample chunks as pseudo-queries
    total = store._conn.execute("SELECT COUNT(*) as n FROM chunks").fetchone()["n"]
    if total == 0:
        return []

    sample_size = min(max_queries, total)
    rows = store._conn.execute(
        "SELECT c.text, d.path FROM chunks c "
        "JOIN documents d ON d.id = c.doc_id "
        "ORDER BY RANDOM() LIMIT ?",
        [sample_size],
    ).fetchall()

    pairs = []
    disagreements = 0

    for row in rows:
        query_text = row["text"][:200]  # use first 200 chars as pseudo-query
        doc_path = row["path"]

        emb = embed_query(query_text, model_name)

        # Vector-heavy search
        try:
            vec_results = store.search(
                query_embedding=emb,
                query_text=query_text,
                top_k=TOP_K,
                vec_weight=0.95,
                fts_weight=0.05,
                adaptive_rrf=False,
            )
        except Exception:
            continue

        # FTS-heavy search
        try:
            fts_results = store.search(
                query_embedding=emb,
                query_text=query_text,
                top_k=TOP_K,
                vec_weight=0.05,
                fts_weight=0.95,
                adaptive_rrf=False,
            )
        except Exception:
            continue

        vec_paths = {r.path for r in vec_results[:5]}
        fts_paths = {r.path for r in fts_results[:5]}

        if vec_paths != fts_paths:
            disagreements += 1

        # The document's own chunk is the positive
        # Find it in results
        positive_text = None
        for r in vec_results + fts_results:
            if r.path == doc_path:
                positive_text = r.text
                break

        if positive_text and positive_text != query_text:
            pairs.append({"query": query_text, "positive": positive_text})

    logger.info(
        "Generated %d pairs from %d queries (%d disagreements, %.0f%%)",
        len(pairs),
        len(rows),
        disagreements,
        disagreements / len(rows) * 100 if rows else 0,
    )
    return pairs


def train_mnrl(
    pairs: list[dict],
    base_model: str = "BAAI/bge-small-en-v1.5",
    output_path: str = "~/.vstash/models/retrained",
    epochs: int = 2,
    lr: float = 3e-6,
    batch_size: int = 64,
) -> str:
    """Fine-tune an embedding model using MNRL on disagreement pairs.

    Args:
        pairs: List of dicts with 'query' and 'positive' keys.
        base_model: HuggingFace model to fine-tune from.
        output_path: Where to save the fine-tuned model.
        epochs: Number of training epochs.
        lr: Learning rate.
        batch_size: Training batch size.

    Returns:
        Path to the saved model.

    Raises:
        ImportError: If sentence-transformers is not installed.
    """
    try:
        from sentence_transformers import InputExample, SentenceTransformer, losses
        from torch.utils.data import DataLoader
    except ImportError:
        raise ImportError(
            "sentence-transformers and torch are required for vstash retrain. "
            "Install with: pip install sentence-transformers torch"
        )

    output = str(Path(output_path).expanduser())
    Path(output).mkdir(parents=True, exist_ok=True)

    logger.info("Loading base model: %s", base_model)
    model = SentenceTransformer(base_model)

    examples = [InputExample(texts=[p["query"], p["positive"]]) for p in pairs]
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = min(50, len(loader) // 5)
    logger.info(
        "Training: %d pairs, %d epochs, batch=%d, lr=%s",
        len(pairs),
        epochs,
        batch_size,
        lr,
    )

    t0 = time.perf_counter()
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        output_path=output,
        show_progress_bar=True,
    )
    elapsed = time.perf_counter() - t0

    model.save(output)

    # Save training metadata
    meta = {
        "base_model": base_model,
        "n_pairs": len(pairs),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "training_time_s": round(elapsed, 1),
    }
    (Path(output) / "training_meta.json").write_text(json.dumps(meta, indent=2))

    logger.info("Model saved to %s (%.0fs)", output, elapsed)
    return output
