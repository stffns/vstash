"""LLM-as-Judge Experiment — Automated Relevance Annotation + RAG Quality

Uses a local LLM (via OpenAI-compatible API, e.g. LM Studio) to:
  1. Pool top-10 results from all 3 search methods (vector, FTS, RRF)
  2. Have the LLM judge relevance of each chunk on a 0-3 scale
  3. Recompute NDCG with complete pooled labels
  4. Evaluate end-to-end RAG answer quality

This addresses the "complete relevance labels" roadmap item and validates
retrieval quality with an independent judge.

Usage:
    python -m experiments.llm_judge [--db PATH] [--base-url http://localhost:1234/v1]
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from openai import OpenAI

from experiments.ablation_rrf import search_fts_only, search_vector_only
from experiments.scoring_grid import (
    EVAL_QUERIES,
    PAPERS,
    URL_TO_NAME,
    dcg_at_k,
    ingest_papers,
)
from vstash.embed import embed_query, get_embedding_dim
from vstash.store import VstashStore

MODEL = "BAAI/bge-small-en-v1.5"


# ------------------------------------------------------------------ #
# LLM Judge                                                            #
# ------------------------------------------------------------------ #

JUDGE_PROMPT = """You are a relevance judge for information retrieval evaluation.

Given a QUERY and a TEXT chunk from a document, rate how relevant the text is to answering the query.

Scale:
  3 = Highly relevant — directly answers or addresses the query
  2 = Relevant — contains useful information related to the query
  1 = Marginally relevant — tangentially related, minor overlap
  0 = Not relevant — no meaningful connection to the query

Reply with ONLY a JSON object: {"score": N, "reason": "one sentence"}"""


def judge_relevance(
    client: OpenAI,
    llm_model: str,
    query: str,
    text: str,
    doc_title: str,
) -> dict:
    """Ask the LLM to judge relevance of a chunk to a query."""
    user_msg = f"QUERY: {query}\n\nDOCUMENT: {doc_title}\n\nTEXT (first 1000 chars):\n{text[:1000]}"

    try:
        resp = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        content = resp.choices[0].message.content or ""
        reasoning_tokens = 0
        if resp.usage and resp.usage.completion_tokens_details:
            reasoning_tokens = (
                getattr(resp.usage.completion_tokens_details, "reasoning_tokens", 0) or 0
            )

        # Parse JSON from content
        content = content.strip()
        # Handle cases where model wraps in markdown
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        result = json.loads(content)
        result["reasoning_tokens"] = reasoning_tokens
        result["total_tokens"] = resp.usage.total_tokens if resp.usage else 0
        return result
    except (json.JSONDecodeError, Exception) as e:
        return {
            "score": -1,
            "reason": f"Parse error: {e}",
            "reasoning_tokens": 0,
            "total_tokens": 0,
        }


# ------------------------------------------------------------------ #
# RAG Quality Evaluation                                               #
# ------------------------------------------------------------------ #

RAG_JUDGE_PROMPT = """You are evaluating the quality of a RAG (Retrieval-Augmented Generation) answer.

Given a QUERY, the SOURCE CHUNKS retrieved, and the GENERATED ANSWER, rate the answer quality.

Scale:
  3 = Excellent — accurate, complete, well-supported by sources
  2 = Good — mostly accurate, covers main points
  1 = Partial — some relevant content but incomplete or partially wrong
  0 = Poor — wrong, hallucinated, or doesn't answer the query

Reply with ONLY a JSON object: {"score": N, "reason": "one sentence"}"""


RAG_QUERIES = [
    "What is MemGPT and how does it manage memory for LLM agents?",
    "How does temporal decay work in memory systems for AI?",
    "What benchmarks exist for evaluating LLM memory capabilities?",
    "How do knowledge graphs help with agent memory?",
    "What approaches exist for memory compression in LLM systems?",
]


def evaluate_rag(
    client: OpenAI,
    llm_model: str,
    store: VstashStore,
    query: str,
    top_k: int = 5,
) -> dict:
    """Run a RAG query and have the LLM judge the answer quality."""
    from vstash.chat import SYSTEM_PROMPT, _build_prompt

    emb = embed_query(query, MODEL)
    chunks = store.search(
        query_embedding=emb,
        query_text=query,
        top_k=top_k,
    )

    # Generate answer using OpenAI client directly
    user_prompt = _build_prompt(query, chunks)
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    answer = resp.choices[0].message.content or ""
    gen_latency = (time.perf_counter() - t0) * 1000

    # Judge the answer
    chunks_text = "\n---\n".join(f"[{c.title}] {c.text[:400]}" for c in chunks[:3])
    user_msg = f"QUERY: {query}\n\nSOURCE CHUNKS:\n{chunks_text}\n\nGENERATED ANSWER:\n{answer}"

    try:
        resp = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": RAG_JUDGE_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        content = (resp.choices[0].message.content or "").strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        judgment = json.loads(content)
    except Exception as e:
        judgment = {"score": -1, "reason": str(e)}

    return {
        "query": query,
        "answer": answer,
        "answer_length": len(answer),
        "generation_latency_ms": gen_latency,
        "n_chunks": len(chunks),
        "judgment": judgment,
    }


# ------------------------------------------------------------------ #
# Pooled NDCG computation                                             #
# ------------------------------------------------------------------ #


def ndcg_pooled(
    result_titles: list[str],
    pooled_labels: dict[str, int],
    k: int = 10,
) -> float:
    """NDCG@k with complete pooled relevance labels."""
    actual_rel = [pooled_labels.get(t, 0) for t in result_titles[:k]]
    ideal_rel = sorted(pooled_labels.values(), reverse=True)[:k]

    actual_dcg = dcg_at_k(actual_rel, k)
    ideal_dcg = dcg_at_k(ideal_rel, k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# ------------------------------------------------------------------ #
# Main experiment                                                      #
# ------------------------------------------------------------------ #


@dataclass
class PooledQueryResult:
    query: str
    n_pooled: int
    n_relevant: int  # score >= 2
    vector_ndcg_5: float
    vector_ndcg_10: float
    fts_ndcg_5: float
    fts_ndcg_10: float
    rrf_ndcg_5: float
    rrf_ndcg_10: float
    labels: dict  # title -> score


@dataclass
class JudgeExperimentResult:
    llm_model: str
    total_judgments: int
    total_tokens: int
    total_reasoning_tokens: int
    avg_judgment_score: float
    pooled_results: list[PooledQueryResult]
    rag_results: list[dict]
    # Aggregates
    avg_vector_ndcg_5: float
    avg_fts_ndcg_5: float
    avg_rrf_ndcg_5: float
    avg_vector_ndcg_10: float
    avg_fts_ndcg_10: float
    avg_rrf_ndcg_10: float
    avg_rag_score: float


def run_judge_experiment(
    store: VstashStore,
    client: OpenAI,
    llm_model: str,
    top_k: int = 10,
) -> JudgeExperimentResult:
    """Run the full LLM judge experiment."""

    total_judgments = 0
    total_tokens = 0
    total_reasoning_tokens = 0
    all_scores = []
    pooled_results = []

    print("\n  Phase 1: Pooled Relevance Judgments")
    print(f"  {'─' * 50}")

    for qi, eq in enumerate(EVAL_QUERIES):
        q = eq["query"]
        print(f"\n  Q{qi + 1}: {q[:60]}...")

        emb = embed_query(q, MODEL)

        # Get top-10 from all 3 methods
        vec_results = search_vector_only(store, emb, top_k)
        fts_results = search_fts_only(store, q, top_k)
        rrf_results = store.search(query_embedding=emb, query_text=q, top_k=top_k, scoring=None)

        # Pool unique chunks (by title + text hash)
        seen = {}  # key -> (title, text)
        for results, method in [
            (vec_results, "vec"),
            (fts_results, "fts"),
            (rrf_results, "rrf"),
        ]:
            for r in results:
                key = r.title
                if key not in seen:
                    seen[key] = (r.title, r.text)

        print(f"    Pooled {len(seen)} unique documents, judging...", flush=True)

        # Judge each pooled chunk
        labels = {}  # title -> score
        for key, (title, text) in seen.items():
            doc_name = URL_TO_NAME.get(title, title[:50])
            judgment = judge_relevance(client, llm_model, q, text, doc_name)
            score = judgment.get("score", -1)
            if score >= 0:
                labels[key] = score
                all_scores.append(score)
                total_tokens += judgment.get("total_tokens", 0)
                total_reasoning_tokens += judgment.get("reasoning_tokens", 0)
            total_judgments += 1
            print(f"      {doc_name[:40]:<40} → {score}/3  ({judgment.get('reason', '')[:40]})")

        # Compute NDCG with pooled labels
        vec_titles = [r.title for r in vec_results]
        fts_titles = [r.title for r in fts_results]
        rrf_titles = [r.title for r in rrf_results]

        pooled_results.append(
            PooledQueryResult(
                query=q,
                n_pooled=len(seen),
                n_relevant=sum(1 for s in labels.values() if s >= 2),
                vector_ndcg_5=ndcg_pooled(vec_titles, labels, k=5),
                vector_ndcg_10=ndcg_pooled(vec_titles, labels, k=10),
                fts_ndcg_5=ndcg_pooled(fts_titles, labels, k=5),
                fts_ndcg_10=ndcg_pooled(fts_titles, labels, k=10),
                rrf_ndcg_5=ndcg_pooled(rrf_titles, labels, k=5),
                rrf_ndcg_10=ndcg_pooled(rrf_titles, labels, k=10),
                labels=labels,
            )
        )

    # Phase 2: RAG quality
    print("\n\n  Phase 2: RAG End-to-End Quality")
    print(f"  {'─' * 50}")

    rag_results = []
    for rq in RAG_QUERIES:
        print(f"\n  Q: {rq[:60]}...")
        result = evaluate_rag(client, llm_model, store, rq)
        rag_results.append(result)
        j = result["judgment"]
        print(
            f"    Score: {j.get('score', '?')}/3  Latency: {result['generation_latency_ms']:.0f}ms"
        )
        print(f"    Answer: {result['answer'][:100]}...")

    # Aggregates
    avg_vec_5 = sum(r.vector_ndcg_5 for r in pooled_results) / len(pooled_results)
    avg_fts_5 = sum(r.fts_ndcg_5 for r in pooled_results) / len(pooled_results)
    avg_rrf_5 = sum(r.rrf_ndcg_5 for r in pooled_results) / len(pooled_results)
    avg_vec_10 = sum(r.vector_ndcg_10 for r in pooled_results) / len(pooled_results)
    avg_fts_10 = sum(r.fts_ndcg_10 for r in pooled_results) / len(pooled_results)
    avg_rrf_10 = sum(r.rrf_ndcg_10 for r in pooled_results) / len(pooled_results)

    rag_scores = [
        r["judgment"].get("score", 0) for r in rag_results if r["judgment"].get("score", -1) >= 0
    ]
    avg_rag = sum(rag_scores) / len(rag_scores) if rag_scores else 0.0

    return JudgeExperimentResult(
        llm_model=llm_model,
        total_judgments=total_judgments,
        total_tokens=total_tokens,
        total_reasoning_tokens=total_reasoning_tokens,
        avg_judgment_score=sum(all_scores) / len(all_scores) if all_scores else 0,
        pooled_results=pooled_results,
        rag_results=rag_results,
        avg_vector_ndcg_5=avg_vec_5,
        avg_fts_ndcg_5=avg_fts_5,
        avg_rrf_ndcg_5=avg_rrf_5,
        avg_vector_ndcg_10=avg_vec_10,
        avg_fts_ndcg_10=avg_fts_10,
        avg_rrf_ndcg_10=avg_rrf_10,
        avg_rag_score=avg_rag,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-Judge Evaluation")
    parser.add_argument("--db", type=str, help="Reuse existing DB")
    parser.add_argument("--base-url", type=str, default="http://localhost:1234/v1")
    parser.add_argument("--llm-model", type=str, default="qwen/qwen3.5-9b")
    parser.add_argument("--output", type=str, default="experiments/results/llm_judge.json")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    client = OpenAI(api_key="not-needed", base_url=args.base_url)

    # Verify LLM is reachable
    try:
        models = client.models.list()
        available = [m.id for m in models.data]
        print(f"  LLM API: {args.base_url}")
        print(f"  Available models: {available}")
        if args.llm_model not in available:
            print(f"  WARNING: {args.llm_model} not in available models")
    except Exception as e:
        print(f"  ERROR: Cannot reach LLM API at {args.base_url}: {e}")
        return

    dim = get_embedding_dim(MODEL)

    if args.db and Path(args.db).exists():
        db_path = args.db
        store = VstashStore(db_path, embedding_dim=dim)
        n_chunks = store._conn.execute("SELECT COUNT(*) as n FROM chunks").fetchone()["n"]
        print(f"  Reusing DB: {db_path} ({n_chunks} chunks)")
    else:
        tmp_dir = tempfile.mkdtemp(prefix="vstash_judge_")
        db_path = str(Path(tmp_dir) / "judge.db")
        store = VstashStore(db_path, embedding_dim=dim)
        print(f"  New DB: {db_path}")
        print("  Ingesting papers...")
        n = ingest_papers(store)
        print(f"  Ingested: {n}/{len(PAPERS)}\n")

    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  LLM-AS-JUDGE EVALUATION ({args.llm_model})")
    print(f"{sep}")

    result = run_judge_experiment(store, client, args.llm_model, top_k=args.top_k)

    # Summary
    print(f"\n{sep}")
    print("  POOLED NDCG RESULTS (LLM-judged relevance labels)")
    print(f"{sep}\n")

    print(f"  {'Mode':<15} {'NDCG@5':>8} {'NDCG@10':>9}")
    print(f"  {'─' * 35}")
    print(
        f"  {'Vector-only':<15} {result.avg_vector_ndcg_5:>8.4f} {result.avg_vector_ndcg_10:>9.4f}"
    )
    print(f"  {'FTS keyword':<15} {result.avg_fts_ndcg_5:>8.4f} {result.avg_fts_ndcg_10:>9.4f}")
    print(f"  {'Hybrid RRF':<15} {result.avg_rrf_ndcg_5:>8.4f} {result.avg_rrf_ndcg_10:>9.4f}")

    print("\n  Per-query breakdown:")
    for pr in result.pooled_results:
        print(f"    Q: {pr.query[:55]}")
        print(f"      Pooled: {pr.n_pooled} docs, {pr.n_relevant} relevant (score≥2)")
        print(f"      Vec={pr.vector_ndcg_5:.3f}  FTS={pr.fts_ndcg_5:.3f}  RRF={pr.rrf_ndcg_5:.3f}")

    print(f"\n{sep}")
    print("  RAG QUALITY")
    print(f"{sep}\n")
    print(f"  Average RAG score: {result.avg_rag_score:.2f}/3.0")
    for rr in result.rag_results:
        j = rr["judgment"]
        print(
            f"    {rr['query'][:55]:<55} {j.get('score', '?')}/3  {rr['generation_latency_ms']:.0f}ms"
        )

    print("\n  LLM Stats:")
    print(f"    Total judgments: {result.total_judgments}")
    print(f"    Total tokens: {result.total_tokens:,}")
    print(f"    Reasoning tokens: {result.total_reasoning_tokens:,}")
    print(f"    Avg relevance score: {result.avg_judgment_score:.2f}/3.0")

    # Save JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "experiment": "llm_judge",
        "llm_model": args.llm_model,
        "db_path": db_path,
        "pooled_results": [asdict(pr) for pr in result.pooled_results],
        "rag_results": result.rag_results,
        "summary": {
            "avg_vector_ndcg_5": result.avg_vector_ndcg_5,
            "avg_fts_ndcg_5": result.avg_fts_ndcg_5,
            "avg_rrf_ndcg_5": result.avg_rrf_ndcg_5,
            "avg_vector_ndcg_10": result.avg_vector_ndcg_10,
            "avg_fts_ndcg_10": result.avg_fts_ndcg_10,
            "avg_rrf_ndcg_10": result.avg_rrf_ndcg_10,
            "avg_rag_score": result.avg_rag_score,
            "total_judgments": result.total_judgments,
            "total_tokens": result.total_tokens,
        },
    }
    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"\n  Results saved to: {output_path}")

    store.close()


if __name__ == "__main__":
    main()
