"""Answer Relevance Benchmark — End-to-End Pipeline Comparison.

Compares vstash (full pipeline: RRF + scoring + MMR + expansion) vs
Chroma (dense-only) on actual answer quality using an LLM judge.

For each query:
1. Retrieve top-K chunks with vstash and Chroma
2. Generate an answer from each context using the same LLM
3. LLM judge scores each answer (0-3 scale)

This measures what the user actually experiences — not just retrieval
ranking, but whether the retrieved context produces a correct answer.

Usage:
    python -m experiments.answer_relevance
    python -m experiments.answer_relevance --queries 50
    python -m experiments.answer_relevance --dataset nfcorpus
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import tempfile
import time

from vstash.embed import embed_query, embed_texts, get_embedding_dim
from vstash.store import VstashStore

# ------------------------------------------------------------------ #
# Configuration                                                        #
# ------------------------------------------------------------------ #

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 5
CACHE_DIR = "experiments/data"

JUDGE_SYSTEM = """You are an expert relevance judge. Given a question and an answer, rate the answer quality on a 0-3 scale:

0 = Completely wrong or irrelevant — the answer does not address the question at all
1 = Partially relevant — touches on the topic but misses the key point or contains errors
2 = Mostly correct — addresses the question with minor gaps or imprecisions
3 = Excellent — directly and correctly answers the question with supporting evidence

Respond with ONLY a single digit (0, 1, 2, or 3). Nothing else."""

ANSWER_SYSTEM = """Answer the question based ONLY on the provided context. Be concise and specific. If the context does not contain enough information, say "insufficient context"."""


# ------------------------------------------------------------------ #
# BEIR data loading                                                    #
# ------------------------------------------------------------------ #


def load_beir(name: str) -> tuple[dict, dict, dict]:
    cache = f"{CACHE_DIR}/beir_{name}"
    corpus: dict[str, dict] = {}
    with open(f"{cache}/corpus.jsonl") as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = doc
    queries: dict[str, str] = {}
    with open(f"{cache}/queries.jsonl") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]
    qrels: dict[str, dict[str, int]] = {}
    with open(f"{cache}/qrels/test.tsv") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            qid, did, score = parts[0], parts[1], int(parts[2])
            qrels.setdefault(qid, {})[did] = score
    return corpus, queries, qrels


# ------------------------------------------------------------------ #
# LLM calls via OpenAI-compatible API                                  #
# ------------------------------------------------------------------ #


def _call_llm(
    system: str,
    user: str,
    *,
    base_url: str = "http://localhost:1234/v1",
    model: str = "qwen/qwen3.5-9b",
    temperature: float = 0.1,
    max_tokens: int = 500,
) -> str:
    """Call a local LLM via OpenAI-compatible API."""
    import urllib.request

    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()

    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"].strip()


def generate_answer(query: str, chunks: list[str]) -> str:
    """Generate an answer from retrieved chunks."""
    context = "\n\n".join(f"--- Context {i + 1} ---\n{c}" for i, c in enumerate(chunks))
    prompt = f"{context}\n\n---\nQuestion: {query}"
    return _call_llm(ANSWER_SYSTEM, prompt)


def judge_answer(query: str, answer: str) -> int:
    """LLM judge scores an answer 0-3."""
    prompt = f"Question: {query}\n\nAnswer: {answer}"
    response = _call_llm(JUDGE_SYSTEM, prompt, max_tokens=5)
    # Extract first digit
    for ch in response:
        if ch.isdigit():
            return min(3, int(ch))
    return 0


# ------------------------------------------------------------------ #
# Main benchmark                                                       #
# ------------------------------------------------------------------ #


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer Relevance Benchmark")
    parser.add_argument("--dataset", default="scifact", help="BEIR dataset name")
    parser.add_argument("--queries", type=int, default=30, help="Number of queries to evaluate")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Embedding model")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Chunks per query")
    args = parser.parse_args()

    corpus, queries, qrels = load_beir(args.dataset)
    test_qids = list(qrels.keys())[: args.queries]
    dim = get_embedding_dim(args.model)

    print(f"\n{'=' * 70}")
    print("  Answer Relevance Benchmark")
    print(f"  Dataset: {args.dataset} ({len(corpus)} docs)")
    print(f"  Queries: {len(test_qids)}")
    print(f"  Top-K: {args.top_k}")
    print(f"{'=' * 70}")

    # Pre-embed corpus
    print("\n  Embedding corpus...", flush=True)
    doc_ids = list(corpus.keys())
    doc_texts = [
        (corpus[d].get("title", "") + "\n" + corpus[d].get("text", "")).strip() for d in doc_ids
    ]
    all_embs: list[list[float]] = []
    for bs in range(0, len(doc_texts), 64):
        all_embs.extend(embed_texts(doc_texts[bs : bs + 64], args.model))
        if (bs + 64) % 2000 < 64:
            print(f"    {min(bs + 64, len(doc_texts))}/{len(doc_texts)}...", flush=True)

    # Setup vstash
    db_path = tempfile.mktemp(suffix=".db")
    store = VstashStore(db_path, embedding_dim=dim)
    vstash_id_map: dict[str, str] = {}
    for doc_id, text, emb in zip(doc_ids, doc_texts, all_embs):
        path = f"/beir/{doc_id}"
        store.add_document(
            path=path,
            title=corpus[doc_id].get("title", ""),
            chunks=[text],
            embeddings=[emb],
            source_type="text",
        )
        vstash_id_map[path] = doc_id
    print(
        f"  vstash ingested ({store._conn.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]} chunks)"
    )

    # Setup Chroma
    try:
        import chromadb

        chroma_dir = tempfile.mkdtemp()
        client = chromadb.PersistentClient(path=chroma_dir)
        col = client.create_collection(name="bench", metadata={"hnsw:space": "cosine"})
        for bs in range(0, len(doc_ids), 4000):
            end = min(bs + 4000, len(doc_ids))
            col.add(ids=doc_ids[bs:end], embeddings=all_embs[bs:end], documents=doc_texts[bs:end])
        has_chroma = True
        print("  Chroma ingested")
    except ImportError:
        has_chroma = False
        print("  Chroma not installed — skipping comparison")

    # Warm up
    embed_query("warmup", args.model)

    # Test LLM connectivity
    print("\n  Testing LLM connection...", flush=True)
    try:
        test = _call_llm("Say OK", "test", max_tokens=5)
        print(f"  LLM OK: {test[:20]}")
    except Exception as e:
        print(f"  LLM FAILED: {e}")
        print("  Start a local LLM server (Ollama/LM Studio) and retry.")
        return

    # Evaluate
    print(f"\n  Evaluating {len(test_qids)} queries...\n", flush=True)

    results: list[dict] = []

    for i, qid in enumerate(test_qids):
        query_text = queries[qid]
        qe = embed_query(query_text, args.model)

        # vstash retrieval (full pipeline)
        v_results = store.search(qe, query_text, top_k=args.top_k)
        v_chunks = [r.text for r in v_results]

        # Generate and judge vstash answer
        v_answer = generate_answer(query_text, v_chunks)
        v_score = judge_answer(query_text, v_answer)

        entry: dict = {
            "qid": qid,
            "query": query_text[:80],
            "vstash_score": v_score,
            "vstash_answer": v_answer[:200],
        }

        # Chroma comparison
        if has_chroma:
            c_results = col.query(query_embeddings=[qe], n_results=args.top_k)
            c_chunks = c_results["documents"][0] if c_results["documents"] else []
            c_answer = generate_answer(query_text, c_chunks)
            c_score = judge_answer(query_text, c_answer)
            entry["chroma_score"] = c_score
            entry["chroma_answer"] = c_answer[:200]

        results.append(entry)

        # Progress
        c_str = f"  chroma={entry.get('chroma_score', '-')}" if has_chroma else ""
        print(
            f'  [{i + 1}/{len(test_qids)}] vstash={v_score}{c_str}  q="{query_text[:50]}..."',
            flush=True,
        )

    # Summary
    v_scores = [r["vstash_score"] for r in results]
    v_mean = statistics.mean(v_scores)

    print(f"\n{'=' * 70}")
    print(f"  RESULTS — Answer Relevance ({args.dataset})")
    print(f"{'=' * 70}")
    print("\n  vstash (full pipeline):")
    print(f"    Mean score:  {v_mean:.2f} / 3.0")
    print(
        f"    Score dist:  0={v_scores.count(0)} 1={v_scores.count(1)} 2={v_scores.count(2)} 3={v_scores.count(3)}"
    )
    print(
        f"    Excellent:   {v_scores.count(3)}/{len(v_scores)} ({v_scores.count(3) / len(v_scores) * 100:.0f}%)"
    )

    if has_chroma:
        c_scores = [r["chroma_score"] for r in results]
        c_mean = statistics.mean(c_scores)
        print("\n  Chroma (dense-only):")
        print(f"    Mean score:  {c_mean:.2f} / 3.0")
        print(
            f"    Score dist:  0={c_scores.count(0)} 1={c_scores.count(1)} 2={c_scores.count(2)} 3={c_scores.count(3)}"
        )
        print(
            f"    Excellent:   {c_scores.count(3)}/{len(c_scores)} ({c_scores.count(3) / len(c_scores) * 100:.0f}%)"
        )

        # Head-to-head
        v_wins = sum(1 for v, c in zip(v_scores, c_scores) if v > c)
        c_wins = sum(1 for v, c in zip(v_scores, c_scores) if c > v)
        ties = sum(1 for v, c in zip(v_scores, c_scores) if v == c)
        print("\n  Head-to-head:")
        print(f"    vstash wins: {v_wins}  chroma wins: {c_wins}  ties: {ties}")
        print(
            f"    Delta: {v_mean - c_mean:+.2f} ({(v_mean - c_mean) / c_mean * 100:+.1f}%)"
            if c_mean > 0
            else ""
        )

    # Save
    os.makedirs("experiments/results", exist_ok=True)
    output = {
        "experiment": "answer_relevance",
        "dataset": args.dataset,
        "queries": len(test_qids),
        "top_k": args.top_k,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "vstash_mean_score": round(v_mean, 3),
        "results": results,
    }
    if has_chroma:
        output["chroma_mean_score"] = round(c_mean, 3)
        output["vstash_wins"] = v_wins
        output["chroma_wins"] = c_wins
        output["ties"] = ties

    output_path = f"experiments/results/answer_relevance_{args.dataset}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {output_path}")

    # Cleanup
    store.close()
    os.unlink(db_path)
    if has_chroma:
        shutil.rmtree(chroma_dir)


if __name__ == "__main__":
    main()
