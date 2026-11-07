"""Probe: can a query-side adapter replace full fine-tuning (no corpus re-index)?

Hypothesis
----------
Fine-tuning BGE-small (v2/v3) moves the whole vector space, so every stored
chunk has to be re-embedded. If instead the document encoder stays frozen and
only a small adapter ``q' = normalize(W q)`` is learned on the query side, the
index never changes, training runs in seconds on CPU over cached vectors, and
the adapter is a few KB that could live in the SQLite file.

The open question is expressivity: a linear map on the query alone cannot
re-separate documents the base model collapsed together. This probe measures
how much of v3's gain such an adapter recovers, under one evaluation pipeline.

Setup
-----
* Datasets: SciFact, NFCorpus, FiQA (the three corpora v3 was trained on).
  Adapters train on each dataset's ``train`` qrels and are evaluated on
  ``test`` qrels (the same 300 / 323 / 648 test queries v3 reports on).
* Document vectors: base ``BAAI/bge-small-en-v1.5`` via vstash's own embed
  path (FastEmbed), cached once. They are never recomputed for adapters.
* Arms:
    - ``base``           frozen BGE-small, no adapter.
    - ``bias``           ``q + b`` (384 params): a query-independent shift,
                         i.e. a pure corpus prior. Diagnostic.
    - ``lowrank``        ``W = I + A B^T`` (rank r, default 16, ~12k params),
                         one adapter per dataset.
    - ``full``           ``W = I + D`` (d x d, ~147k params), per dataset.
    - ``lowrank_shared`` one rank-r adapter trained on the union of the three
                         train sets (the apples-to-apples comparison with v3,
                         which is one model for all three corpora).
    - ``v3``             ``Stffens/bge-small-rrf-v3`` with the corpus fully
                         re-embedded (the cost the adapter wants to avoid).
* Training: InfoNCE over in-batch documents plus mined hard negatives (top
  base-model hits that are not qrels positives), temperature 0.05, Adam,
  L2 pull toward the identity, early stopping on a 10% dev split of train.
* Metrics on ``test``:
    - ``dense``  NDCG@10 from exact cosine top-10 (isolates the vector branch,
                 which is the only thing an adapter changes).
    - ``hybrid`` NDCG@10 through the real ``VstashStore.search`` pipeline
                 (vector + FTS5 + adaptive RRF + distance cutoff + MMR) with
                 the adapted query vector, plus the ``relevance_tier``
                 distribution, to check whether the cosine thresholds
                 calibrated for BGE (#272) still hold after the adapter.
    - ``*_unseen`` the same NDCG restricted to test queries none of whose
                 relevant docs is a train positive. SciFact shares most of
                 its relevant docs between train and test, so the all-query
                 number rewards memorising a "popular doc" prior.

Findings: ``experiments/results/query_adapter_probe.md``.

Run::

    python -m experiments.probe_query_adapter
    python -m experiments.probe_query_adapter --datasets scifact --skip-v3

Results land in ``experiments/results/query_adapter_probe.json``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

# torch MUST load before onnxruntime (pulled in by vstash.embed / FastEmbed):
# on macOS the reverse order segfaults on the first autograd op (two
# OpenMP runtimes). Verified: same ops, torch-first -> ok, torch-last -> SIGSEGV.
import torch

from experiments.beir_benchmark import download_beir, ndcg_at_k
from vstash.embed import embed_texts, register_encoder_resolver
from vstash.store import VstashStore, relevance_tier

BASE_MODEL = "BAAI/bge-small-en-v1.5"
V3_MODEL = "Stffens/bge-small-rrf-v3"
DIM = 384
TOP_K = 10
CACHE_DIR = Path("experiments/data/adapter_cache")
RESULTS_PATH = Path("experiments/results/query_adapter_probe.json")


# ------------------------------------------------------------------ #
# Data                                                                 #
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

    return {
        "corpus": corpus,
        "queries": queries,
        "train": qrels("train"),
        "test": qrels("test"),
    }


class _HFClsEncoder:
    """CLS-pooled, L2-normalised encoder for safetensors-only checkpoints.

    v3 ships only ``model.safetensors`` (no ONNX), and the vstash venv does
    not carry sentence-transformers. BGE / v3 use CLS pooling (see the repo's
    ``1_Pooling/config.json``), so a plain ``transformers`` forward is
    equivalent. ``--check-encoder`` validates it against FastEmbed on the base
    model before trusting it for v3.
    """

    def __init__(self, model_name: str) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.embedding_dim = self.model.config.hidden_size

    def encode(self, texts: list[str]) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                batch = self.tok(
                    texts[i : i + 64],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                cls = self.model(**batch).last_hidden_state[:, 0]
                out.append(torch.nn.functional.normalize(cls, dim=-1).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.embedding_dim), np.float32)


_hf_encoders: dict[str, _HFClsEncoder] = {}


def _hf_resolver(model_name: str):
    if model_name != V3_MODEL:
        return None
    if model_name not in _hf_encoders:
        _hf_encoders[model_name] = _HFClsEncoder(model_name)
    return _hf_encoders[model_name]


def _doc_text(doc: dict) -> str:
    # Same text shape experiments.beir_benchmark ingests.
    return (doc.get("title", "") + "\n" + doc.get("text", "")).strip()


def embed_dataset(name: str, ds: dict, model: str) -> dict:
    """Embed corpus + every train/test query with ``model``; cached as .npz."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    slug = model.replace("/", "__")
    path = CACHE_DIR / f"{name}__{slug}.npz"
    doc_ids = list(ds["corpus"].keys())
    qids = sorted(set(ds["train"]) | set(ds["test"]))
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if list(z["doc_ids"]) == doc_ids and list(z["qids"]) == qids:
            return {"doc_ids": doc_ids, "D": z["D"], "qids": qids, "Q": z["Q"]}
    t0 = time.perf_counter()
    texts = [_doc_text(ds["corpus"][d]) for d in doc_ids]
    D = np.zeros((len(texts), DIM), np.float32)
    for i in range(0, len(texts), 256):
        D[i : i + 256] = np.asarray(embed_texts(texts[i : i + 256], model), np.float32)
        if (i // 256) % 20 == 0:
            print(f"    [{name}/{model}] docs {i}/{len(texts)}")
    qtexts = [ds["queries"][q] for q in qids]
    Q = np.zeros((len(qtexts), DIM), np.float32)
    for i in range(0, len(qtexts), 256):
        Q[i : i + 256] = np.asarray(embed_texts(qtexts[i : i + 256], model), np.float32)
    np.savez(path, doc_ids=np.array(doc_ids), D=D, qids=np.array(qids), Q=Q)
    print(f"    [{name}/{model}] embedded in {time.perf_counter() - t0:.1f}s")
    return {"doc_ids": doc_ids, "D": D, "qids": qids, "Q": Q}


# ------------------------------------------------------------------ #
# Adapters                                                             #
# ------------------------------------------------------------------ #


def make_adapter(kind: str, rank: int):
    from torch import nn

    class LowRank(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # B starts at zero so the adapter is exactly the identity at init;
            # A gets orthonormal columns (unit scale) so dL/dB = A^T(...) is not
            # throttled by a tiny init -- with A ~ 0.02 the first run never
            # moved off the identity before early stopping fired.
            self.A = nn.Parameter(torch.linalg.qr(torch.randn(DIM, rank))[0])
            self.B = nn.Parameter(torch.zeros(DIM, rank))

        def delta_params(self):
            return [self.B @ self.A.T]

        def forward(self, q):
            return nn.functional.normalize(q + (q @ self.A) @ self.B.T, dim=-1)

    class Full(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.D = nn.Parameter(torch.zeros(DIM, DIM))

        def delta_params(self):
            return [self.D]

        def forward(self, q):
            return nn.functional.normalize(q + q @ self.D, dim=-1)

    class Bias(nn.Module):
        """``q' = normalize(q + b)``: a single learned direction, no query
        dependence. Diagnostic arm: whatever it gains is a corpus prior
        ("prefer the docs that tend to be relevant"), not better semantics."""

        def __init__(self) -> None:
            super().__init__()
            self.b = nn.Parameter(torch.zeros(DIM))

        def delta_params(self):
            return [self.b]

        def forward(self, q):
            return nn.functional.normalize(q + self.b, dim=-1)

    return {"lowrank": LowRank, "full": Full, "bias": Bias}[kind]()


def apply_adapter(adapter, Q: np.ndarray) -> np.ndarray:

    if adapter is None:
        return Q
    with torch.no_grad():
        return adapter(torch.from_numpy(Q)).numpy()


def _topk(scores: np.ndarray, k: int) -> np.ndarray:
    idx = np.argpartition(-scores, kth=min(k, scores.shape[1] - 1), axis=1)[:, :k]
    order = np.take_along_axis(scores, idx, axis=1).argsort(axis=1)[:, ::-1]
    return np.take_along_axis(idx, order, axis=1)


def dense_ndcg_per_query(
    Qa: np.ndarray, D: np.ndarray, doc_ids: list[str], qrels: list[dict]
) -> np.ndarray:
    vals = []
    for s in range(0, len(Qa), 512):
        top = _topk(Qa[s : s + 512] @ D.T, TOP_K)
        for row, qrel in zip(top, qrels[s : s + 512]):
            vals.append(ndcg_at_k([doc_ids[j] for j in row], qrel, TOP_K))
    return np.asarray(vals, dtype=np.float64)


def dense_ndcg(Qa: np.ndarray, D: np.ndarray, doc_ids: list[str], qrels: list[dict]) -> float:
    vals = dense_ndcg_per_query(Qa, D, doc_ids, qrels)
    return float(vals.mean()) if len(vals) else 0.0


def build_pairs(emb: dict, qrels: dict, n_hard: int, pool: int) -> dict:
    """(query vec, positive doc idx, hard-negative doc idxs) per qrels positive."""
    qpos = {q: i for i, q in enumerate(emb["qids"])}
    dpos = {d: i for i, d in enumerate(emb["doc_ids"])}
    qids = [q for q in qrels if q in qpos]
    Q = emb["Q"][[qpos[q] for q in qids]]
    hard: list[list[int]] = []
    for s in range(0, len(Q), 512):
        top = _topk(Q[s : s + 512] @ emb["D"].T, pool)
        for row, qid in zip(top, qids[s : s + 512]):
            pos = {dpos[d] for d, sc in qrels[qid].items() if sc > 0}
            hard.append([j for j in row if j not in pos][:n_hard])
    qv, pv, nv, owner = [], [], [], []
    for i, qid in enumerate(qids):
        for d, sc in qrels[qid].items():
            if sc > 0 and len(hard[i]) == n_hard:
                qv.append(Q[i])
                pv.append(dpos[d])
                nv.append(hard[i])
                owner.append(qid)
    qindex = {q: i for i, q in enumerate(dict.fromkeys(owner))}
    return {
        "Q": np.asarray(qv, np.float32),
        "pos": np.asarray(pv, np.int64),
        "neg": np.asarray(nv, np.int64),
        "qid": np.asarray([qindex[q] for q in owner], np.int64),
    }


def train_adapter(
    kind: str,
    groups: list[dict],
    *,
    rank: int,
    epochs: int,
    lr: float,
    l2: float,
    tau: float,
    batch: int,
    seed: int,
    min_epochs: int = 5,
    patience: int = 3,
) -> tuple[object, dict]:
    """Train one adapter on one or more datasets' pairs.

    ``groups``: one entry per dataset with ``pairs`` (from ``build_pairs``),
    ``D`` (doc vectors), ``dev`` (dev query vecs + qrels for early stopping).
    Batches never mix datasets, so in-batch negatives come from one corpus.
    """

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    adapter = make_adapter(kind, rank)
    # Low-rank / bias adapters have far fewer, coupled parameters; they need a
    # larger step than the full matrix to leave the identity in a few epochs.
    opt = torch.optim.Adam(adapter.parameters(), lr=lr if kind == "full" else lr * 10)
    Dt = [torch.from_numpy(g["D"]) for g in groups]

    def dev_score() -> float:
        return float(
            np.mean(
                [
                    dense_ndcg(
                        apply_adapter(adapter, g["dev"]["Q"]),
                        g["D"],
                        g["doc_ids"],
                        g["dev"]["qrels"],
                    )
                    for g in groups
                ]
            )
        )

    best = dev_score()
    base_dev = best
    best_state = copy.deepcopy(adapter.state_dict())
    best_epoch, bad = 0, 0
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        batches = []
        for gi, g in enumerate(groups):
            perm = rng.permutation(len(g["pairs"]["pos"]))
            batches += [(gi, perm[i : i + batch]) for i in range(0, len(perm), batch)]
        rng.shuffle(batches)
        for gi, idx in batches:
            p = groups[gi]["pairs"]
            q = adapter(torch.from_numpy(p["Q"][idx]))
            pos_ids = torch.from_numpy(p["pos"][idx])
            neg_ids = torch.from_numpy(p["neg"][idx])
            cand = torch.cat([pos_ids, neg_ids.reshape(-1)])
            # Score each query against every in-batch positive + every hard
            # negative; the target is its own positive (column i).
            logits = (q @ Dt[gi][cand].T) / tau
            # Another pair's positive is not a negative when it is the same
            # doc, or another positive of the same query (multi-positive
            # qrels): mask both off the in-batch block.
            qid_ids = torch.from_numpy(p["qid"][idx])
            same = (pos_ids[:, None] == pos_ids[None, :]) | (qid_ids[:, None] == qid_ids[None, :])
            same.fill_diagonal_(False)
            logits[:, : len(idx)] = logits[:, : len(idx)].masked_fill(same, -1e4)
            loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(idx)))
            reg = sum((d**2).sum() for d in adapter.delta_params())
            opt.zero_grad()
            (loss + l2 * reg).backward()
            opt.step()
        score = dev_score()
        if score > best + 1e-4:
            best, best_state, best_epoch, bad = score, copy.deepcopy(adapter.state_dict()), epoch, 0
        elif epoch >= min_epochs:
            bad += 1
            if bad >= patience:
                break
    adapter.load_state_dict(best_state)
    n_params = sum(p.numel() for p in adapter.parameters())
    info = {
        "kind": kind,
        "params": int(n_params),
        "size_kb_fp32": round(n_params * 4 / 1024, 1),
        "dev_ndcg_base": round(base_dev, 4),
        "dev_ndcg_best": round(best, 4),
        "best_epoch": best_epoch,
        "train_seconds": round(time.perf_counter() - t0, 1),
        "n_pairs": int(sum(len(g["pairs"]["pos"]) for g in groups)),
    }
    return adapter, info


# ------------------------------------------------------------------ #
# Hybrid evaluation through VstashStore                               #
# ------------------------------------------------------------------ #


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


def hybrid_eval(store: VstashStore, texts: list[str], Qa: np.ndarray, qrels: list[dict]) -> dict:
    ndcgs, tiers, best = [], Counter(), []
    for text, qv, qrel in zip(texts, Qa, qrels):
        res = store.search(qv.tolist(), text, top_k=TOP_K)
        ndcgs.append(ndcg_at_k([r.path.removeprefix("/beir/") for r in res], qrel, TOP_K))
        bd = store.last_best_distance
        best.append(bd)
        tiers[relevance_tier(bd)] += 1
    n = len(ndcgs)
    return {
        "_per_query": np.asarray(ndcgs, dtype=np.float64),
        "ndcg_10": round(float(np.mean(ndcgs)), 4),
        "best_distance_median": round(statistics.median(best), 4),
        "tier_share": {t: round(tiers[t] / n, 3) for t in ("high", "medium", "low")},
    }


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus", "fiqa"])
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--n-hard", type=int, default=8)
    ap.add_argument("--hard-pool", type=int, default=30)
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-v3", action="store_true", help="Skip the re-embedded v3 arm")
    ap.add_argument("--skip-hybrid", action="store_true", help="Dense metrics only")
    ap.add_argument("--out", default=str(RESULTS_PATH), help="Results JSON path")
    args = ap.parse_args()

    register_encoder_resolver(_hf_resolver)

    if not args.skip_v3:
        probe = ["What causes neural tube defects?", "Interest rates and bond prices."]
        fe = np.asarray(embed_texts(probe, BASE_MODEL), np.float32)
        hf = _HFClsEncoder(BASE_MODEL).encode(probe)
        agree = float(np.min(np.sum(fe * hf, axis=1)))
        print(f"  encoder check: FastEmbed vs HF-CLS cosine on base model = {agree:.5f}")
        if agree < 0.999:
            raise SystemExit("HF CLS encoder does not reproduce FastEmbed; v3 arm untrustworthy")

    rng = np.random.default_rng(args.seed)
    data: dict[str, dict] = {}
    for name in args.datasets:
        print(f"\n== {name}")
        ds = load_dataset(name)
        print(f"  corpus={len(ds['corpus'])} train_q={len(ds['train'])} test_q={len(ds['test'])}")
        emb = embed_dataset(name, ds, BASE_MODEL)
        train_qids = sorted(ds["train"])
        rng.shuffle(train_qids)
        n_dev = max(1, int(len(train_qids) * args.dev_frac))
        dev_qids, fit_qids = train_qids[:n_dev], train_qids[n_dev:]
        qpos = {q: i for i, q in enumerate(emb["qids"])}
        pairs = build_pairs(emb, {q: ds["train"][q] for q in fit_qids}, args.n_hard, args.hard_pool)
        data[name] = {
            "ds": ds,
            "emb": emb,
            "group": {
                "pairs": pairs,
                "D": emb["D"],
                "doc_ids": emb["doc_ids"],
                "dev": {
                    "Q": emb["Q"][[qpos[q] for q in dev_qids]],
                    "qrels": [ds["train"][q] for q in dev_qids],
                },
            },
        }
        print(f"  train pairs={len(pairs['pos'])} dev queries={len(dev_qids)}")

    hp = dict(
        rank=args.rank,
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
        tau=args.tau,
        batch=args.batch,
        seed=args.seed,
    )
    adapters: dict[str, dict] = {name: {} for name in data}
    training: dict = {}
    for name, d in data.items():
        for kind in ("bias", "lowrank", "full"):
            ad, info = train_adapter(kind, [d["group"]], **hp)
            adapters[name][kind] = ad
            training[f"{name}/{kind}"] = info
            print(f"  [{name}] {kind}: {info}")
    if len(data) > 1:
        shared, info = train_adapter("lowrank", [d["group"] for d in data.values()], **hp)
        training["shared/lowrank"] = info
        print(f"  [shared] lowrank: {info}")
        for name in data:
            adapters[name]["lowrank_shared"] = shared

    results: dict = {"config": vars(args), "training": training, "datasets": {}}
    with tempfile.TemporaryDirectory() as tmp:
        for name, d in data.items():
            ds, emb = d["ds"], d["emb"]
            qpos = {q: i for i, q in enumerate(emb["qids"])}
            test_qids = sorted(ds["test"])
            Qtest = emb["Q"][[qpos[q] for q in test_qids]]
            qrels = [ds["test"][q] for q in test_qids]
            texts = [ds["queries"][q] for q in test_qids]
            # "unseen": test queries none of whose relevant docs is a train
            # positive. On SciFact ~2/3 of test queries only cite docs that
            # train already labels relevant, so a learned "popular doc" prior
            # inflates the all-query number; this subset isolates semantics.
            train_pos = {d for q in ds["train"].values() for d, sc in q.items() if sc > 0}
            unseen = np.asarray(
                [not any(d in train_pos for d, sc in qr.items() if sc > 0) for qr in qrels]
            )

            def score(Qa, D, doc_ids, store) -> dict:
                dq = dense_ndcg_per_query(Qa, D, doc_ids, qrels)
                out = {
                    "dense_ndcg_10": round(float(dq.mean()), 4),
                    "dense_ndcg_10_unseen": round(float(dq[unseen].mean()), 4)
                    if unseen.any()
                    else None,
                }
                if store is not None:
                    h = hybrid_eval(store, texts, Qa, qrels)
                    hq = h.pop("_per_query")
                    h["ndcg_10_unseen"] = (
                        round(float(hq[unseen].mean()), 4) if unseen.any() else None
                    )
                    out["hybrid"] = h
                return out

            arms: dict[str, dict] = {}
            store = None if args.skip_hybrid else build_store(name, ds, emb, tmp)
            for arm, ad in [("base", None), *adapters[name].items()]:
                arms[arm] = score(apply_adapter(ad, Qtest), emb["D"], emb["doc_ids"], store)
                print(f"  [{name}] {arm}: {arms[arm]}")
            if store is not None:
                store.close()
            if not args.skip_v3:
                emb3 = embed_dataset(name, ds, V3_MODEL)
                q3 = {q: i for i, q in enumerate(emb3["qids"])}
                Q3 = emb3["Q"][[q3[q] for q in test_qids]]
                s3 = None if args.skip_hybrid else build_store(f"{name}_v3", ds, emb3, tmp)
                arms["v3"] = score(Q3, emb3["D"], emb3["doc_ids"], s3)
                if s3 is not None:
                    s3.close()
                print(f"  [{name}] v3: {arms['v3']}")
            results["datasets"][name] = {
                "n_test": len(test_qids),
                "n_test_unseen": int(unseen.sum()),
                "arms": arms,
            }

    # Share of v3's gain each adapter recovers (per metric, per dataset).
    for name, r in results["datasets"].items():
        arms = r["arms"]
        if "v3" not in arms:
            continue
        rec = {}
        for metric in ("dense", "hybrid"):
            get = (
                (lambda a: a["dense_ndcg_10"])
                if metric == "dense"
                else (lambda a: a.get("hybrid", {}).get("ndcg_10"))
            )
            b, v = get(arms["base"]), get(arms["v3"])
            if b is None or v is None or v == b:
                continue
            rec[metric] = {
                arm: round((get(a) - b) / (v - b), 3)
                for arm, a in arms.items()
                if arm not in ("base", "v3") and get(a) is not None
            }
        r["share_of_v3_gain"] = rec

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    print(
        json.dumps({n: r.get("share_of_v3_gain") for n, r in results["datasets"].items()}, indent=2)
    )


if __name__ == "__main__":
    main()
