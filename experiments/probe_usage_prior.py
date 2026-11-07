"""Probe: does a "this chunk tends to get used" prior help as an explicit score feature?

Context
-------
The query-adapter probe (PR #514, ``experiments/probe_query_adapter.py``)
found that a query-side ``W`` generalises to nothing new. The one thing it did
learn was a corpus prior, "these documents tend to be the answer", which won
big on SciFact queries whose relevant docs were already train positives, cost
accuracy on queries whose docs were not, and distorted distances. The usage log
(#513) records exactly that prior (``search_results.used``). This probe asks
whether feeding it to the ranker *explicitly* keeps the gain without the harm.

Simulated usage log
-------------------
Each dataset's ``train`` queries play the agent's past searches. Two logs:

* ``oracle``  every train positive counts as one use (upper bound; includes
              docs the search never surfaced, which a real log cannot know).
* ``logged``  realistic: a doc gets a use only if the base hybrid search
              actually *returned* it in the top-10 for that train query and it
              is relevant, kept with p=0.9 (callers forget to report), plus
              one false "use" of a random returned non-relevant doc with
              p=0.2 per query (callers mark the wrong thing).

Ranking with the prior
----------------------
``VstashStore.search`` retrieves the top-``--pool`` (default 50) candidates
through the real hybrid pipeline. The prior then re-scores them and the list
is cut to 10, which approximates applying the feature inside the pipeline
before the ``top_k`` cut:

* ``mult``  ``score * (1 + beta * log1p(uses))`` (same shape as recency boost)
* ``rrf3``  ``score + beta * 1 / (60 + rank_by_uses)`` for docs with uses > 0,
            i.e. a third RRF list ordered by use count.
* ``knn``   query-conditioned: the uses come only from past queries whose
            embedding has cosine >= ``tau`` with the current one (top 20,
            weighted by similarity), then ``score * (1 + beta * log1p(w))``.
            A query with no near past query keeps the base ranking.

``mult`` and ``rrf3`` use one global count per doc, whatever the query.

Distances are untouched by construction, so ``relevance_tier`` calibration
cannot break (the adapter's failure mode).

Protocol
--------
``beta`` is picked on a dev split (10% of train queries) using use counts built
**only from the other 90%**, so a dev query never sees its own positives.
Test then uses counts from the full train log. Reported on ``test``: all
queries, the ``unseen`` subset (no relevant doc is a train positive), and the
``untouched`` subset (no relevant doc received any use in that log, false uses
included). Only ``untouched`` is a strict harm test for the noisy ``logged``
log: a false use can land on a doc relevant to an ``unseen`` / FiQA query and
help it, so an unchanged mean there could hide offsetting losses.

Run::

    python -m experiments.probe_usage_prior

Results: ``experiments/results/usage_prior_probe.json`` + ``.md`` write-up.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.beir_benchmark import download_beir, ndcg_at_k
from vstash.embed import embed_texts
from vstash.store import VstashStore

BASE_MODEL = "BAAI/bge-small-en-v1.5"
DIM = 384
TOP_K = 10
RRF_K = 60
# Shared with experiments/probe_query_adapter.py (same .npz layout).
CACHE_DIR = Path("experiments/data/adapter_cache")
RESULTS_PATH = Path("experiments/results/usage_prior_probe.json")
BETAS = {
    "mult": [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0],
    "rrf3": [0.0, 0.25, 0.5, 1.0, 2.0, 4.0],
}
# Query-conditioned (kNN over past queries) prior grid.
KNN_TAUS = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
KNN_BETAS = [0.1, 0.25, 0.5, 1.0, 2.0]
# Max dev NDCG@10 loss any single dataset may take under the dev-only
# "safe" selection rule (see main).
SAFE_DEV_LOSS = 0.002


# ------------------------------------------------------------------ #
# Data + embeddings                                                    #
# ------------------------------------------------------------------ #


def load_dataset(name: str) -> dict:
    cache = download_beir(name)
    corpus: dict[str, dict] = {}
    with open(f"{cache}/corpus.jsonl") as f:
        for line in f:
            d = json.loads(line)
            corpus[d["_id"]] = d
    queries: dict[str, str] = {}
    with open(f"{cache}/queries.jsonl") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]

    def qrels(split: str) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        path = f"{cache}/qrels/{split}.tsv"
        if not os.path.exists(path):
            return out
        with open(path) as f:
            next(f)
            for line in f:
                qid, did, score = line.rstrip("\n").split("\t")[:3]
                if did in corpus and qid in queries:
                    out.setdefault(qid, {})[did] = int(score)
        return out

    return {"corpus": corpus, "queries": queries, "train": qrels("train"), "test": qrels("test")}


def _doc_text(doc: dict) -> str:
    return (doc.get("title", "") + "\n" + doc.get("text", "")).strip()


def embed_dataset(name: str, ds: dict) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}__{BASE_MODEL.replace('/', '__')}.npz"
    doc_ids = list(ds["corpus"].keys())
    qids = sorted(set(ds["train"]) | set(ds["test"]))
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if list(z["doc_ids"]) == doc_ids and list(z["qids"]) == qids:
            return {"doc_ids": doc_ids, "D": z["D"], "qids": qids, "Q": z["Q"]}
    texts = [_doc_text(ds["corpus"][d]) for d in doc_ids]
    D = np.zeros((len(texts), DIM), np.float32)
    for i in range(0, len(texts), 256):
        D[i : i + 256] = np.asarray(embed_texts(texts[i : i + 256], BASE_MODEL), np.float32)
    qtexts = [ds["queries"][q] for q in qids]
    Q = np.zeros((len(qtexts), DIM), np.float32)
    for i in range(0, len(qtexts), 256):
        Q[i : i + 256] = np.asarray(embed_texts(qtexts[i : i + 256], BASE_MODEL), np.float32)
    np.savez(path, doc_ids=np.array(doc_ids), D=D, qids=np.array(qids), Q=Q)
    return {"doc_ids": doc_ids, "D": D, "qids": qids, "Q": Q}


def build_store(name: str, ds: dict, emb: dict, tmp: str) -> VstashStore:
    store = VstashStore(os.path.join(tmp, f"{name}.db"), embedding_dim=DIM)
    batch: list[dict] = []
    with store.batch_mode(defer_fts=True):
        for i, did in enumerate(emb["doc_ids"]):
            doc = ds["corpus"][did]
            batch.append(
                {
                    "path": f"/beir/{did}",
                    "title": doc.get("title", ""),
                    "chunks": [_doc_text(doc)],
                    "embeddings": [emb["D"][i].tolist()],
                    "source_type": "text",
                }
            )
            if len(batch) >= 1000:
                store.add_documents_batch(batch)
                batch = []
        if batch:
            store.add_documents_batch(batch)
    return store


# ------------------------------------------------------------------ #
# Retrieval, usage log, prior                                          #
# ------------------------------------------------------------------ #


def retrieve(store: VstashStore, emb: dict, ds: dict, qids: list[str], pool: int) -> dict:
    """qid -> [(doc_id, rrf score)] for the top-``pool`` hybrid candidates."""
    qpos = {q: i for i, q in enumerate(emb["qids"])}
    out: dict[str, list[tuple[str, float]]] = {}
    for q in qids:
        res = store.search(emb["Q"][qpos[q]].tolist(), ds["queries"][q], top_k=pool)
        out[q] = [(r.path.removeprefix("/beir/"), float(r.score)) for r in res]
    return out


def usage_log(
    kind: str,
    qids: list[str],
    qrels: dict[str, dict[str, int]],
    returned: dict[str, list[tuple[str, float]]],
    rng: np.random.Generator,
    p_keep: float,
    p_false: float,
) -> dict[str, Counter]:
    """Per past query, the docs it "used" (see module docstring)."""
    log: dict[str, Counter] = {}
    for q in qids:
        rel = {d for d, s in qrels[q].items() if s > 0}
        used: Counter = Counter()
        if kind == "oracle":
            used.update(rel)
        else:
            top = [d for d, _ in returned[q][:TOP_K]]
            for d in top:
                if d in rel and rng.random() < p_keep:
                    used[d] += 1
            wrong = [d for d in top if d not in rel]
            if wrong and rng.random() < p_false:
                used[wrong[int(rng.integers(len(wrong)))]] += 1
        log[q] = used
    return log


def knn_uses(
    query_vecs: np.ndarray,
    past_vecs: np.ndarray,
    past_logs: list[Counter],
    tau: float,
    k: int = 20,
) -> list[Counter]:
    """Query-conditioned prior: for each query, similarity-weighted uses from
    the (at most ``k``) past queries with cosine >= ``tau``. A query with no
    near past query gets an empty Counter, i.e. the base ranking."""
    out: list[Counter] = []
    sims = query_vecs @ past_vecs.T
    for row in sims:
        nn = np.argsort(-row)[:k]
        c: Counter = Counter()
        for j in nn:
            if row[j] < tau:
                break
            for d, n in past_logs[j].items():
                c[d] += float(row[j]) * n
        out.append(c)
    return out


def rerank(cands: list[tuple[str, float]], uses: Counter, mode: str, beta: float) -> list[str]:
    if beta == 0.0 or not uses:
        return [d for d, _ in cands[:TOP_K]]
    if mode in ("mult", "knn"):
        scored = [(s * (1.0 + beta * math.log1p(uses.get(d, 0))), d) for d, s in cands]
    else:
        used = sorted((d for d, _ in cands if uses.get(d, 0) > 0), key=lambda d: -uses[d])
        urank = {d: i for i, d in enumerate(used)}
        scored = [(s + (beta / (RRF_K + urank[d]) if d in urank else 0.0), d) for d, s in cands]
    # Stable on ties: keep the pipeline's original order.
    order = sorted(range(len(scored)), key=lambda i: (-scored[i][0], i))
    return [scored[i][1] for i in order[:TOP_K]]


def ndcgs(
    qids: list[str], qrels: dict, cands: dict, uses: list[Counter], mode: str, beta: float
) -> np.ndarray:
    return np.asarray(
        [ndcg_at_k(rerank(cands[q], u, mode, beta), qrels[q], TOP_K) for q, u in zip(qids, uses)]
    )


def grid() -> list[tuple[str, str, float, float | None]]:
    """(arm, param label, beta, tau) for every configuration tried."""
    out = []
    for mode, betas in BETAS.items():
        out += [(mode, f"beta={b}", b, None) for b in betas]
    out += [("knn", f"tau={t},beta={b}", b, t) for t in KNN_TAUS for b in KNN_BETAS if b > 0] + [
        ("knn", "beta=0.0", 0.0, None)
    ]
    return out


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus", "fiqa"])
    ap.add_argument("--pool", type=int, default=50)
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument("--p-keep", type=float, default=0.9)
    ap.add_argument("--p-false", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(RESULTS_PATH))
    args = ap.parse_args()

    results: dict = {"config": vars(args), "datasets": {}}
    dev_tables: dict = {}
    configs = grid()
    with tempfile.TemporaryDirectory() as tmp:
        for name in args.datasets:
            t0 = time.perf_counter()
            ds = load_dataset(name)
            emb = embed_dataset(name, ds)
            store = build_store(name, ds, emb, tmp)
            rng = np.random.default_rng(args.seed)
            train = sorted(ds["train"])
            rng.shuffle(train)
            n_dev = max(1, int(len(train) * args.dev_frac))
            dev, fit = train[:n_dev], train[n_dev:]
            test = sorted(ds["test"])
            cands = retrieve(store, emb, ds, train + test, args.pool)
            # The simulated log must come from what a normal top-10 search
            # returns. VstashStore.search scales its candidate pool and FTS
            # admission with top_k, so the head of a top_k=50 search is not
            # the same list; use separate top_k=10 searches for the log and
            # keep the 50-pool only for re-ranking.
            shown = retrieve(store, emb, ds, train, TOP_K)
            store.close()
            qpos = {q: i for i, q in enumerate(emb["qids"])}

            def vecs(qs: list[str]) -> np.ndarray:
                return emb["Q"][[qpos[q] for q in qs]]

            train_pos = {d for q in ds["train"].values() for d, s in q.items() if s > 0}
            unseen = np.asarray(
                [not any(d in train_pos for d, s in ds["test"][q].items() if s > 0) for q in test]
            )
            r: dict = {
                "n_test": len(test),
                "n_test_unseen": int(unseen.sum()),
                "n_train": len(train),
                "arms": {},
            }
            no_uses = [Counter()] * len(test)
            base = ndcgs(test, ds["test"], cands, no_uses, "mult", 0.0)
            r["arms"]["base"] = {
                "ndcg_10": round(float(base.mean()), 4),
                "ndcg_10_unseen": round(float(base[unseen].mean()), 4) if unseen.any() else None,
            }
            for log_kind in ("oracle", "logged"):
                # Dev sees only the fit split's log (never its own positives);
                # test sees the whole train log. Independent RNG draws.
                fit_log = usage_log(
                    log_kind,
                    fit,
                    ds["train"],
                    shown,
                    np.random.default_rng(args.seed + 1),
                    args.p_keep,
                    args.p_false,
                )
                all_log = usage_log(
                    log_kind,
                    train,
                    ds["train"],
                    shown,
                    np.random.default_rng(args.seed + 2),
                    args.p_keep,
                    args.p_false,
                )
                # Counter.update: linear in logged entries (sum() re-copied the
                # growing total for every query).
                fit_global: Counter = Counter()
                for c in fit_log.values():
                    fit_global.update(c)
                all_global: Counter = Counter()
                for c in all_log.values():
                    all_global.update(c)
                # "untouched": test queries none of whose relevant docs got
                # ANY use in this log. Unlike "unseen" (no train positive),
                # this also excludes docs hit by the logged log's false uses,
                # so on this subset a prior can only hurt -- a true harm test
                # for the noisy log as well as the oracle.
                untouched = np.asarray(
                    [
                        not any(all_global.get(d, 0) > 0 for d, s in ds["test"][q].items() if s > 0)
                        for q in test
                    ]
                )
                r.setdefault("n_test_untouched", {})[log_kind] = int(untouched.sum())
                knn_cache: dict = {}

                def uses_for(split: str, mode: str, tau: float | None) -> list[Counter]:
                    qs = dev if split == "dev" else test
                    if mode != "knn" or tau is None:
                        g = fit_global if split == "dev" else all_global
                        return [g] * len(qs)
                    key = (split, tau)
                    if key not in knn_cache:
                        past = fit if split == "dev" else train
                        logs = fit_log if split == "dev" else all_log
                        knn_cache[key] = knn_uses(
                            vecs(qs), vecs(past), [logs[q] for q in past], tau
                        )
                    return knn_cache[key]

                for mode in ("mult", "rrf3", "knn"):
                    mode_cfgs = [c for c in configs if c[0] == mode]
                    dev_curve, sweep = {}, {}
                    for _, label, beta, tau in mode_cfgs:
                        dev_curve[label] = float(
                            ndcgs(
                                dev, ds["train"], cands, uses_for("dev", mode, tau), mode, beta
                            ).mean()
                        )
                        v = ndcgs(test, ds["test"], cands, uses_for("test", mode, tau), mode, beta)
                        sweep[label] = {
                            "ndcg_10": round(float(v.mean()), 4),
                            "ndcg_10_unseen": round(float(v[unseen].mean()), 4)
                            if unseen.any()
                            else None,
                            "ndcg_10_untouched": round(float(v[untouched].mean()), 4)
                            if untouched.any()
                            else None,
                        }
                    dev_tables[(name, log_kind, mode)] = dev_curve
                    # Ties go to the weakest intervention (listed first / beta 0).
                    best = max(
                        dev_curve, key=lambda lb: (round(dev_curve[lb], 6), lb == "beta=0.0")
                    )
                    entry = {
                        "param_dev": best,
                        "dev_ndcg": {lb: round(v, 4) for lb, v in dev_curve.items()},
                        "test_at_param_dev": sweep[best],
                        "test_sweep": sweep,
                    }
                    if mode == "knn":
                        tau_best = next((c[3] for c in mode_cfgs if c[1] == best), None)
                        if tau_best is not None:
                            entry["test_queries_with_neighbours"] = round(
                                float(np.mean([bool(u) for u in uses_for("test", mode, tau_best)])),
                                3,
                            )
                    else:
                        entry["docs_with_uses"] = len(all_global)
                    r["arms"][f"{log_kind}/{mode}"] = entry
            results["datasets"][name] = r
            print(f"== {name} ({time.perf_counter() - t0:.0f}s)")
            print(f"  base {r['arms']['base']}")
            for k, v in r["arms"].items():
                if k != "base":
                    print(f"  {k}: {v['param_dev']} test={v['test_at_param_dev']}")

    # One global setting per (log, mode): what a shipped default would be.
    glob: dict = {}
    for log_kind in ("oracle", "logged"):
        for mode in ("mult", "rrf3", "knn"):
            curves = [dev_tables[(n, log_kind, mode)] for n in args.datasets]
            labels = list(curves[0])
            best = max(
                labels,
                key=lambda lb: (
                    round(float(np.mean([c[lb] for c in curves])), 6),
                    lb == "beta=0.0",
                ),
            )
            glob[f"{log_kind}/{mode}"] = {
                "param": best,
                "test": {
                    n: results["datasets"][n]["arms"][f"{log_kind}/{mode}"]["test_sweep"][best]
                    for n in args.datasets
                },
            }
    # Dev-only *safe* selection rule: the setting with the highest mean dev
    # gain subject to no dataset losing more than SAFE_DEV_LOSS on dev vs
    # beta=0. Picked without looking at test, so its test numbers are an
    # honest check of whether "safe on dev" transfers.
    for log_kind in ("oracle", "logged"):
        for mode in ("mult", "rrf3", "knn"):
            curves = [dev_tables[(n, log_kind, mode)] for n in args.datasets]
            ok = [
                lb
                for lb in curves[0]
                if all(c[lb] - c["beta=0.0"] >= -SAFE_DEV_LOSS for c in curves)
            ]
            best = max(
                ok,
                key=lambda lb: (
                    round(float(np.mean([c[lb] - c["beta=0.0"] for c in curves])), 6),
                    lb == "beta=0.0",
                ),
            )
            glob[f"{log_kind}/{mode}"]["safe_rule"] = {
                "param": best,
                "test": {
                    n: results["datasets"][n]["arms"][f"{log_kind}/{mode}"]["test_sweep"][best]
                    for n in args.datasets
                },
            }
    results["global_beta"] = glob
    print(json.dumps(glob, indent=2))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
