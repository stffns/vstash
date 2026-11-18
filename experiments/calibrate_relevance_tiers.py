"""Calibrate ``relevance_tier`` thresholds per embedding model.

The built-in tiers (``high <= 0.4513 < medium <= 0.4802 < low``) were set for
``BAAI/bge-small-en-v1.5``. Other models put good answers at very different
cosine distances: with ``paraphrase-multilingual-MiniLM-L12-v2`` a Spanish
memory (KakesuI) saw correct answers at 0.49-0.56, all labelled "low".

What the tiers detect (finding, ``--holdout``)
---------------------------------------------
Best distance does **not** separate in-domain unanswerable questions: holding
out half the queries and removing all their relevant documents leaves their
best distance indistinguishable from answerable ones (BGE on SciFact: median
0.21 vs 0.18; MiniLM 0.37 vs 0.34), and every one of them is "high" under
the BGE constants. Even cross-domain FiQA questions against SciFact mostly
stay under 0.4802. The tiers are an **off-topic** detector: "low" means the
query is about something the store does not cover at all.

Criterion (relative position, transferred from BGE)
---------------------------------------------------
Per model and corpus, two anchors:

* ``m_ans``  median best distance of the corpus's own (answerable) queries;
* ``m_off``  median best distance of off-domain queries against the corpus.

BGE's constants sit at ``p = (t - m_ans) / (m_off - m_ans)`` on SciFact with
FiQA as off-domain. Each model gets ``t = m_ans + p * (m_off - m_ans)`` on
its own scale, so "high" / "medium" sit at the same relative place between
"typical good match" and "typical off-topic match" as they do for BGE. BGE
reproduces its constants exactly; ``p > 1`` means the tiers fire only beyond
the typical off-topic query, which is what the BGE constants do.

Corpora: SciFact (English, FiQA questions as off-domain) and MTEB
``SpanishPassageRetrievalS2P`` (Spanish health passages, subsampled; short
Spanish news texts from ``SpanishNewsClassification`` as off-domain).

Best distance is ``1 - max cosine`` over the corpus, computed in numpy on
L2-normalised vectors (not every model returns unit vectors); that is the
store's exact-KNN ``last_best_distance`` for one-chunk documents.

Run (Jina / EmbeddingGemma need the backends from #522 on the path)::

    python -m experiments.calibrate_relevance_tiers [--models ...] [--holdout]

Embeddings are cached in ``experiments/data/tier_calib_cache/``; results go to
``experiments/results/relevance_tiers_calibration.json``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from vstash.embed import embed_query, embed_texts

BGE = "BAAI/bge-small-en-v1.5"
MODELS = [
    BGE,
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "jinaai/jina-embeddings-v5-text-nano-retrieval",
    "embeddinggemma-300m",
]
HIGH, MEDIUM = 0.4513, 0.4802  # current BGE constants
CACHE = Path("experiments/data/tier_calib_cache")
OUT = Path("experiments/results/relevance_tiers_calibration.json")
S2P = "mteb/SpanishPassageRetrievalS2P"
ES_OFF = "mteb/SpanishNewsClassification"
S2P_CORPUS = 3000
N_OFF = 300
SEED = 13


# ------------------------------------------------------------------ #
# Data                                                                 #
# ------------------------------------------------------------------ #


def _beir(name: str) -> tuple[dict[str, str], dict[str, str], dict[str, set[str]]]:
    from experiments.beir_benchmark import download_beir

    cache = download_beir(name)
    corpus = {}
    with open(f"{cache}/corpus.jsonl") as f:
        for line in f:
            d = json.loads(line)
            corpus[d["_id"]] = (d.get("title", "") + "\n" + d.get("text", "")).strip()
    queries = {}
    with open(f"{cache}/queries.jsonl") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]
    qrels: dict[str, set[str]] = {}
    for split in ("train", "test"):
        path = Path(f"{cache}/qrels/{split}.tsv")
        if not path.exists():
            continue
        with open(path) as f:
            next(f)
            for line in f:
                qid, did, score = line.rstrip("\n").split("\t")[:3]
                if int(score) > 0 and did in corpus and qid in queries:
                    qrels.setdefault(qid, set()).add(did)
    return corpus, {q: queries[q] for q in qrels}, qrels


def _hf_rows(repo: str, path: str) -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    return pq.read_table(hf_hub_download(repo, path, repo_type="dataset")).to_pylist()


def _s2p() -> tuple[dict[str, str], dict[str, str], dict[str, set[str]]]:
    full = {r["_id"]: r["text"] for r in _hf_rows(S2P, "corpus/test-00000-of-00001.parquet")}
    queries = {r["_id"]: r["text"] for r in _hf_rows(S2P, "queries/test-00000-of-00001.parquet")}
    qrels: dict[str, set[str]] = {}
    for r in _hf_rows(S2P, "qrels/test-00000-of-00001.parquet"):
        if float(r["score"]) > 0 and r["corpus-id"] in full:
            qrels.setdefault(r["query-id"], set()).add(r["corpus-id"])
    keep = set().union(*qrels.values())
    rest = sorted(set(full) - keep)
    random.Random(SEED).shuffle(rest)
    keep |= set(rest[: max(0, S2P_CORPUS - len(keep))])
    corpus = {d: full[d] for d in sorted(keep)}
    return corpus, {q: queries[q] for q in qrels}, qrels


def _es_off() -> dict[str, str]:
    rows = _hf_rows(ES_OFF, "data/train-00000-of-00001.parquet")
    texts = [r["text"] for r in rows if r.get("text")]
    random.Random(SEED).shuffle(texts)
    # Question-length snippets: first ~20 words of each news text.
    return {f"news_{i}": " ".join(t.split()[:20]) for i, t in enumerate(texts[:N_OFF])}


# ------------------------------------------------------------------ #
# Embeddings (cached)                                                  #
# ------------------------------------------------------------------ #


def _unit(m: np.ndarray) -> np.ndarray:
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)


def _embed(model: str, tag: str, ids: list[str], texts: list[str], *, query: bool) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{tag}__{'q' if query else 'd'}__{model.replace('/', '__')}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if list(z["ids"]) == ids:
            return z["X"]
    if query:
        X = np.asarray([embed_query(t, model) for t in texts], dtype=np.float32)
    else:
        X = np.vstack(
            [
                np.asarray(embed_texts(texts[i : i + 64], model), dtype=np.float32)
                for i in range(0, len(texts), 64)
            ]
        )
    X = _unit(X)
    np.savez(path, ids=np.array(ids), X=X)
    return X


def _best(Q: np.ndarray, D: np.ndarray) -> np.ndarray:
    return 1.0 - (Q @ D.T).max(axis=1)


def _q(x: np.ndarray, qs: list[float]) -> list[float]:
    return [round(float(v), 4) for v in np.quantile(x, qs)]


# ------------------------------------------------------------------ #
# Anchors + transfer                                                   #
# ------------------------------------------------------------------ #


def _anchors(model: str, tag: str, data, off: dict[str, str]) -> dict:
    corpus, queries, _ = data
    doc_ids = list(corpus)
    D = _embed(model, tag, doc_ids, [corpus[d] for d in doc_ids], query=False)
    q_ids = list(queries)
    ans = _best(_embed(model, tag, q_ids, [queries[q] for q in q_ids], query=True), D)
    o_ids = list(off)
    offd = _best(_embed(model, f"{tag}_off", o_ids, [off[q] for q in o_ids], query=True), D)
    return {"ans": ans, "off": offd}


def _holdout(model: str, tag: str, data) -> dict:
    """In-domain unanswerable hold-out (the finding in the module docstring)."""
    corpus, queries, qrels = data
    doc_ids = list(corpus)
    D = _embed(model, tag, doc_ids, [corpus[d] for d in doc_ids], query=False)
    qids = sorted(qrels)
    random.Random(SEED).shuffle(qids)
    held = qids[: len(qids) // 2]
    drop = set().union(*(qrels[q] for q in held))
    answerable = [q for q in qids[len(qids) // 2 :] if qrels[q] - drop]
    keep = np.asarray([i for i, d in enumerate(doc_ids) if d not in drop])
    q_ids = list(queries)
    Q = _embed(model, tag, q_ids, [queries[q] for q in q_ids], query=True)
    pos = {q: i for i, q in enumerate(q_ids)}
    bd = _best(Q, D[keep])
    a = bd[[pos[q] for q in answerable]]
    u = bd[[pos[q] for q in held]]
    return {
        "n_answerable": len(answerable),
        "n_unanswerable": len(held),
        "answerable_p10_p50_p90": _q(a, [0.1, 0.5, 0.9]),
        "unanswerable_p10_p50_p90": _q(u, [0.1, 0.5, 0.9]),
    }


def _transfer(anc: dict, p_high: float, p_med: float) -> dict:
    m_ans = float(np.median(anc["ans"]))
    m_off = float(np.median(anc["off"]))
    high = m_ans + p_high * (m_off - m_ans)
    med = m_ans + p_med * (m_off - m_ans)
    return {
        "m_ans": round(m_ans, 4),
        "m_off": round(m_off, 4),
        "high_max": round(high, 4),
        "medium_max": round(med, 4),
        "answerable_in_high": round(float((anc["ans"] <= high).mean()), 3),
        "answerable_in_high_or_medium": round(float((anc["ans"] <= med).mean()), 3),
        "off_domain_low": round(float((anc["off"] > med).mean()), 3),
        "answerable_p10_p50_p90": _q(anc["ans"], [0.1, 0.5, 0.9]),
        "off_domain_p10_p50_p90": _q(anc["off"], [0.1, 0.5, 0.9]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--holdout", action="store_true", help="also run the in-domain hold-out")
    args = ap.parse_args()

    scifact = _beir("scifact")
    fiqa_q = _beir("fiqa")[1]
    en_off = dict(random.Random(SEED).sample(sorted(fiqa_q.items()), N_OFF))
    s2p = _s2p()
    es_off = _es_off()

    bge = _anchors(BGE, "scifact", scifact, en_off)
    m_ans, m_off = float(np.median(bge["ans"])), float(np.median(bge["off"]))
    p_high = (HIGH - m_ans) / (m_off - m_ans)
    p_med = (MEDIUM - m_ans) / (m_off - m_ans)
    print(
        f"BGE/SciFact m_ans={m_ans:.4f} m_off={m_off:.4f} -> p_high={p_high:.3f} p_med={p_med:.3f}"
    )

    results: dict = {
        "criterion": (
            "t = m_ans + p * (m_off - m_ans) per model and corpus, with p taken "
            "from BGE's constants on SciFact (FiQA questions as off-domain)"
        ),
        "p_high": round(p_high, 4),
        "p_medium": round(p_med, 4),
        "models": {},
    }
    for model in args.models:
        row: dict = {}
        for tag, data, off in (("scifact", scifact, en_off), ("s2p", s2p, es_off)):
            row[tag] = _transfer(_anchors(model, tag, data, off), p_high, p_med)
            # S2P queries share relevant passages heavily, so a query-level
            # hold-out leaves almost no answerable queries; SciFact only.
            if args.holdout and tag == "scifact":
                row[f"{tag}_holdout"] = _holdout(model, tag, data)
        results["models"][model] = row
        print(model, json.dumps(row, indent=1), flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    if prev.get("criterion") == results["criterion"]:
        merged = prev.get("models", {})
        merged.update(results["models"])
        results["models"] = merged
    OUT.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
