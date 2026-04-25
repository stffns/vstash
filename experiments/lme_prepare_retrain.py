"""LongMemEval -> vstash retrain data prep.

Builds the inputs that ``vstash retrain --training-queries
... --eval-queries ...`` consumes, **without** any synthetic query
generation:

    1. Load longmemeval_s.
    2. Deterministic stratified 80/20 split by ``question_type``.
    3. Ingest every haystack session (train + holdout) into ONE
       VstashStore.  Paths are namespaced as
       ``{question_id}:{session_id}`` because ~4300 LongMemEval
       sessions are reused across questions; without the prefix
       paths would collide and we'd lose the per-question gold
       semantics.
    4. Emit two JSONL files matching the shape
       ``{"query": ..., "relevant_paths": [...]}`` -- one for the
       train split (fed to ``--training-queries``), one for the
       holdout (fed to ``--eval-queries``, the gate).

Usage:
    python -m experiments.lme_prepare_retrain \
        --output-db experiments/lme_retrain_corpus.db \
        --output-train experiments/results/lme_train.jsonl \
        --output-eval  experiments/results/lme_eval.jsonl \
        --output-meta  experiments/results/lme_retrain_meta.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

from experiments.longmemeval_retrieval import (
    DATA_PATH,
    _ST_FALLBACK_MODELS,
    _format_session,
    _st_resolver,
)
from vstash.config import VstashConfig
from vstash.embed import (
    embed_texts,
    get_embedding_dim,
    register_encoder_resolver,
    unregister_encoder_resolver,
)
from vstash.ingest import chunk_text
from vstash.store import VstashStore

DEFAULT_SEED = 42
DEFAULT_HOLDOUT_FRACTION = 0.2
COLLECTION = "lme"


def stratified_split(
    data: list[dict],
    holdout_fraction: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Stratified train / holdout split keyed on ``question_type``.

    Deterministic in (seed, dataset). ``ceil`` for the holdout count
    (not ``floor``) so low-count types like single-session-preference
    (n=30) still get a non-trivial holdout slice.
    """
    by_type: dict[str, list[dict]] = defaultdict(list)
    for q in data:
        by_type[q["question_type"]].append(q)
    train: list[dict] = []
    holdout: list[dict] = []
    rng = random.Random(seed)
    for t in sorted(by_type.keys()):
        items = by_type[t][:]
        rng.shuffle(items)
        n_hold = max(1, math.ceil(len(items) * holdout_fraction))
        holdout.extend(items[:n_hold])
        train.extend(items[n_hold:])
    return train, holdout


def _path_for(question_id: str, session_id: str) -> str:
    """Namespace each session's path so cross-question reuse is preserved.

    LongMemEval has ~4300 session_ids that recur across multiple
    questions. Without the question_id prefix they would dedupe in
    the store and we'd silently lose per-question gold semantics.
    """
    return f"{question_id}:{session_id}"


def ingest_corpus(
    store: VstashStore,
    cfg: VstashConfig,
    questions: list[dict],
    *,
    role: str,
    progress_every: int = 25,
) -> int:
    """Ingest every haystack session for the given questions into one store.

    Each session becomes one document. Within a single question, dup
    session_ids are first-wins-deduped (15 questions in longmemeval_s
    ship duplicates inside one haystack and ``batch_mode(defer_fts=True)``
    forbids re-ingesting a path within one batch). Cross-question
    duplicates are already prevented by the path namespace.

    ``role`` is woven into ``source_type`` (``chat:train`` /
    ``chat:holdout``) so downstream tooling can split without
    re-deriving the split.
    """
    n_docs = 0
    t_start = time.time()
    for i, q in enumerate(questions):
        qid = q["question_id"]
        seen: set[str] = set()
        documents: list[dict] = []
        for sid, turns in zip(q["haystack_session_ids"], q["haystack_sessions"]):
            if sid in seen:
                continue
            seen.add(sid)
            text = _format_session(turns)
            if not text.strip():
                continue
            chunks = chunk_text(text, cfg.chunking.size, cfg.chunking.overlap)
            if not chunks:
                continue
            embeddings = embed_texts(chunks, cfg.embeddings.model)
            documents.append(
                {
                    "path": _path_for(qid, sid),
                    "title": _path_for(qid, sid),
                    "chunks": chunks,
                    "embeddings": embeddings,
                    "source_type": f"chat:{role}",
                    "collection": COLLECTION,
                    "tags": f"qid={qid},role={role}",
                }
            )
        if documents:
            with store.batch_mode(defer_fts=True):
                store.add_documents_batch(documents)
            n_docs += len(documents)
        if (i + 1) % progress_every == 0:
            print(
                f"  [{role}] {i + 1}/{len(questions)} questions, "
                f"{n_docs} docs, {time.time() - t_start:.0f}s elapsed",
                flush=True,
            )
    return n_docs


def build_qrels_jsonl(questions: list[dict], output_path: Path) -> int:
    """Write JSONL of {query, relevant_paths, query_id} for the questions.

    ``query_id`` is included for traceability; ``vstash retrain``
    ignores unknown extra fields. Returns the number of records written.
    Skips questions whose gold session_ids would resolve to an empty
    relevant_paths list (LongMemEval never has this, but defensive).
    """
    n = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        for q in questions:
            qid = q["question_id"]
            paths = [_path_for(qid, sid) for sid in q["answer_session_ids"]]
            if not paths:
                continue
            fh.write(
                json.dumps(
                    {
                        "query": q["question"],
                        "relevant_paths": paths,
                        "query_id": qid,
                    }
                )
                + "\n"
            )
            n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-db", type=Path, required=True)
    p.add_argument("--output-train", type=Path, required=True)
    p.add_argument("--output-eval", type=Path, required=True)
    p.add_argument("--output-meta", type=Path, required=True)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--holdout-fraction", type=float, default=DEFAULT_HOLDOUT_FRACTION)
    p.add_argument(
        "--subset-per-type",
        type=int,
        default=None,
        help="Cap questions per question_type before splitting (smoke runs).",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Embedding model. Default: vstash.toml resolution.",
    )
    p.add_argument("--force", action="store_true", help="Overwrite output_db if it exists")
    args = p.parse_args()

    out_db = args.output_db.expanduser()
    if out_db.exists() and not args.force:
        raise SystemExit(f"{out_db} already exists. Pass --force to overwrite.")
    out_db.parent.mkdir(parents=True, exist_ok=True)
    if out_db.exists():
        out_db.unlink()

    cfg = VstashConfig()
    if args.model is not None:
        cfg = cfg.model_copy(
            update={"embeddings": cfg.embeddings.model_copy(update={"model": args.model})}
        )
    resolver_registered = False
    if cfg.embeddings.model in _ST_FALLBACK_MODELS:
        register_encoder_resolver(_st_resolver)
        resolver_registered = True

    try:
        data = json.loads(DATA_PATH.read_text())
        if args.subset_per_type is not None:
            by_type: dict[str, list[dict]] = defaultdict(list)
            for q in data:
                by_type[q["question_type"]].append(q)
            data = []
            for t in sorted(by_type):
                data.extend(by_type[t][: args.subset_per_type])

        train_qs, holdout_qs = stratified_split(
            data, holdout_fraction=args.holdout_fraction, seed=args.seed
        )
        print(
            f"Split: {len(train_qs)} train + {len(holdout_qs)} holdout "
            f"(holdout_fraction={args.holdout_fraction}, seed={args.seed})"
        )

        n_train_qrels = build_qrels_jsonl(train_qs, args.output_train.expanduser())
        n_eval_qrels = build_qrels_jsonl(holdout_qs, args.output_eval.expanduser())
        print(f"Wrote {n_train_qrels} train queries -> {args.output_train}")
        print(f"Wrote {n_eval_qrels} eval  queries -> {args.output_eval}")

        dim = get_embedding_dim(cfg.embeddings.model)
        store = VstashStore(
            str(out_db),
            embedding_dim=dim,
            vector_backend=cfg.storage.vector_backend,
            snapvec_bits=cfg.storage.snapvec_bits,
            observability=cfg.observability,
            limits=cfg.limits,
            cache=cfg.cache,
        )
        try:
            print("Ingesting train haystacks...")
            n_train = ingest_corpus(store, cfg, train_qs, role="train")
            print("Ingesting holdout haystacks...")
            n_hold = ingest_corpus(store, cfg, holdout_qs, role="holdout")
        finally:
            store.close()

        meta = {
            "data_path": str(DATA_PATH),
            "seed": args.seed,
            "holdout_fraction": args.holdout_fraction,
            "subset_per_type": args.subset_per_type,
            "n_train_questions": len(train_qs),
            "n_holdout_questions": len(holdout_qs),
            "n_train_docs": n_train,
            "n_holdout_docs": n_hold,
            "n_train_qrels": n_train_qrels,
            "n_eval_qrels": n_eval_qrels,
            "embedding_model": cfg.embeddings.model,
            "by_type": {
                t: {
                    "train": sum(1 for q in train_qs if q["question_type"] == t),
                    "holdout": sum(1 for q in holdout_qs if q["question_type"] == t),
                }
                for t in sorted({q["question_type"] for q in data})
            },
        }
        args.output_meta.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.output_meta.expanduser().write_text(json.dumps(meta, indent=2))

        print()
        print(f"Corpus DB:    {out_db}  ({n_train + n_hold} docs)")
        print(f"Train JSONL:  {args.output_train}  ({n_train_qrels} queries)")
        print(f"Eval JSONL:   {args.output_eval}  ({n_eval_qrels} queries)")
        print(f"Meta JSON:    {args.output_meta}")
        print()
        print("Run retrain with the new --eval-queries flag:")
        print()
        print(
            "    vstash retrain \\\n"
            f"        --training-queries {args.output_train} \\\n"
            f"        --eval-queries     {args.output_eval} \\\n"
            f"        --base-model       {cfg.embeddings.model} \\\n"
            "        --max-queries 5000 \\\n"
            "        --epochs 2 --lr 3e-6 --batch-size 64 \\\n"
            "        --output ~/.vstash/models/bge-small-rrf-lme-v1"
        )
        print()
        print(
            "(vstash retrain operates on the active profile's store. "
            "Either ingest the corpus into your active profile, or "
            f"export VSTASH_DB_PATH={out_db} before invoking retrain.)"
        )
    finally:
        if resolver_registered:
            unregister_encoder_resolver(_st_resolver)


if __name__ == "__main__":
    main()
