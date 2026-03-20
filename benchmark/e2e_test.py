#!/usr/bin/env python3
"""
e2e_test.py — End-to-end retrieval + LLM answer benchmark.

Tests vstash with real queries, full LLM answers, and detailed timing.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vstash.chat import ask
from vstash.config import load_config
from vstash.embed import embed_query, get_embedding_dim, warmup
from vstash.store import VstashStore

REPORT_PATH = Path(__file__).parent / "E2E_TEST_REPORT.md"

QUERIES = [
    # Deep technical — should retrieve NAS paper
    "Explain the difference between DARTS and evolutionary NAS methods, including their computational costs",
    # Cross-document synthesis
    "What are the main arguments for renewable energy, and how do costs compare to fossil fuels?",
    # Literary analysis — Alice in Wonderland
    "Describe Alice's emotional journey and key character interactions in Wonderland",
    # Military strategy — Art of War
    "What does Sun Tzu say about the importance of knowing your enemy and terrain?",
    # API architecture — FastAPI patterns
    "How should a production API implement rate limiting and caching with Redis?",
    # Tool self-knowledge — vstash README/Constitution
    "What is vstash's technology stack and how does it achieve fast local search?",
    # RAG-specific — Wikipedia article
    "How does Retrieval-Augmented Generation reduce hallucinations in language models?",
]


def main() -> None:
    cfg = load_config()
    dim = get_embedding_dim(cfg.embeddings.model)
    db_path = str(Path.home() / ".vstash" / "memory.db")

    # Pre-load ONNX model to eliminate cold start on first query
    warmup(cfg.embeddings.model)

    store = VstashStore(db_path, embedding_dim=dim)

    lines: list[str] = []
    lines.append("# vstash End-to-End Test Report\n")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Backend:** {cfg.inference.backend} / {cfg.inference.model}")
    lines.append(f"**Embedding:** {cfg.embeddings.model}\n")

    stats = store.stats()
    lines.append(f"**Corpus:** {stats.documents} documents, {stats.chunks} chunks, {stats.db_size_mb} MB\n")
    lines.append("---\n")

    total_retrieval = 0.0
    total_llm = 0.0

    with store:
        for i, query in enumerate(QUERIES, 1):
            print(f"\n{'='*70}")
            print(f"  Query {i}/{len(QUERIES)}: {query}")
            print(f"{'='*70}")

            # Phase 1: Embedding + retrieval
            t0 = time.time()
            q_embedding = embed_query(query, cfg.embeddings.model)
            t_embed = time.time() - t0

            t0 = time.time()
            results = store.search(q_embedding, query, top_k=5)
            t_search = time.time() - t0

            retrieval_ms = round((t_embed + t_search) * 1000, 1)
            total_retrieval += t_embed + t_search

            print(f"  Retrieval: {retrieval_ms}ms ({round(t_embed*1000,1)}ms embed + {round(t_search*1000,1)}ms search)")
            print(f"  Results: {len(results)} chunks from: {', '.join(set(r.title for r in results))}")

            # Phase 2: LLM answer
            t0 = time.time()
            answer = ask(query, results, cfg)
            t_llm = time.time() - t0
            total_llm += t_llm

            word_count = len(answer.split())
            print(f"  LLM: {round(t_llm, 2)}s → {word_count} words")
            print(f"  Answer preview: {answer[:200]}...")

            # Write to report
            lines.append(f"## Query {i}: \"{query}\"\n")

            # Timing
            lines.append("### ⏱️ Timing\n")
            lines.append("| Phase | Time |")
            lines.append("|---|---|")
            lines.append(f"| Embed query | {round(t_embed*1000,1)}ms |")
            lines.append(f"| Vector + FTS search | {round(t_search*1000,1)}ms |")
            lines.append(f"| **Total retrieval** | **{retrieval_ms}ms** |")
            lines.append(f"| LLM inference | {round(t_llm, 2)}s |")
            lines.append(f"| **End-to-end** | **{round(t_embed + t_search + t_llm, 2)}s** |\n")

            # Sources
            lines.append("### 📚 Retrieved Sources\n")
            lines.append("| # | Source | Score |")
            lines.append("|---|---|---|")
            for j, r in enumerate(results, 1):
                lines.append(f"| {j} | {r.title} (chunk {r.chunk}) | {r.score} |")

            # Answer
            lines.append(f"\n### 💬 Answer ({word_count} words)\n")
            lines.append(answer)
            lines.append("\n---\n")

    # Summary
    lines.append("## Summary\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Queries tested | {len(QUERIES)} |")
    lines.append(f"| Total retrieval time | {round(total_retrieval*1000)}ms ({round(total_retrieval*1000/len(QUERIES))}ms avg) |")
    lines.append(f"| Total LLM time | {round(total_llm, 1)}s ({round(total_llm/len(QUERIES), 2)}s avg) |")
    lines.append(f"| Avg end-to-end | {round((total_retrieval+total_llm)/len(QUERIES), 2)}s |")
    lines.append(f"| Corpus size | {stats.chunks} chunks |")

    report = "\n".join(lines)
    REPORT_PATH.write_text(report)
    print(f"\n\n📊 Full report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
