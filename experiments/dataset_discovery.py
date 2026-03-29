"""Dataset Discovery Engine — vstash as a Kaggle/HuggingFace dataset finder

Fetches ~500 dataset descriptions from HuggingFace Hub, ingests them into
vstash, and evaluates retrieval quality using task_categories as ground truth.

Also serves as a practical demo: "describe what you need, get the right dataset."

Ground truth: HuggingFace datasets have `task_categories` labels (e.g.,
text-classification, image-segmentation). Queries target specific task types.
A retrieved dataset is "relevant" if its task_categories overlap the target.

Metrics:
  - Precision@k, NDCG@k, MRR (same as arxiv_retrieval_bench)
  - Discovery rate: fraction of queries with at least one relevant result in top-5
  - Category coverage: how many unique task_categories appear in results

Interactive mode:
    python -m experiments.dataset_discovery --interactive
    > time series forecasting for retail sales
    1. Walmart Sales Dataset (time-series-forecasting) — 0.87
    2. Store Sales Dataset (tabular-regression) — 0.81
    ...

Usage:
    python -m experiments.dataset_discovery
    python -m experiments.dataset_discovery --datasets 500
    python -m experiments.dataset_discovery --interactive
    python -m experiments.dataset_discovery --skip-download

Output: experiments/results/dataset_discovery.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from vstash.config import ScoringConfig
from vstash.embed import embed_query, embed_texts
from vstash.store import VstashStore

# ------------------------------------------------------------------ #
# Task category groups and queries                                     #
# ------------------------------------------------------------------ #

# Group related HuggingFace task_categories for evaluation.
# Each group has benchmark queries that should retrieve datasets in those categories.
TASK_GROUPS: list[dict] = [
    {
        "id": "text-generation",
        "name": "Text Generation & Language Modeling",
        "categories": ["text-generation", "text2text-generation", "fill-mask"],
        "queries": [
            "large language model training dataset with diverse text",
            "dataset for text generation and creative writing",
            "pretraining corpus for language modeling",
        ],
    },
    {
        "id": "classification",
        "name": "Text Classification & Sentiment",
        "categories": ["text-classification", "sentiment-analysis"],
        "queries": [
            "sentiment analysis dataset with positive and negative labels",
            "text classification benchmark for topic categorization",
            "dataset for spam detection or intent classification",
        ],
    },
    {
        "id": "qa",
        "name": "Question Answering",
        "categories": ["question-answering", "visual-question-answering"],
        "queries": [
            "question answering dataset with context paragraphs",
            "reading comprehension benchmark with extractive answers",
            "open-domain QA dataset for evaluating knowledge retrieval",
        ],
    },
    {
        "id": "translation",
        "name": "Translation & Multilingual",
        "categories": ["translation"],
        "queries": [
            "parallel corpus for machine translation between languages",
            "multilingual dataset for cross-lingual transfer learning",
            "translation pairs for low-resource language pairs",
        ],
    },
    {
        "id": "vision",
        "name": "Computer Vision",
        "categories": [
            "image-classification", "object-detection", "image-segmentation",
            "image-to-text", "image-to-image", "depth-estimation",
        ],
        "queries": [
            "image classification dataset with labeled categories",
            "object detection dataset with bounding box annotations",
            "image captioning dataset with text descriptions",
        ],
    },
    {
        "id": "ner",
        "name": "Named Entity Recognition",
        "categories": ["token-classification"],
        "queries": [
            "named entity recognition dataset with person and location labels",
            "token classification dataset for biomedical entities",
            "sequence labeling dataset for information extraction",
        ],
    },
    {
        "id": "similarity",
        "name": "Similarity & Retrieval",
        "categories": ["sentence-similarity", "text-retrieval", "feature-extraction"],
        "queries": [
            "sentence similarity dataset for semantic textual similarity",
            "information retrieval dataset with queries and relevant documents",
            "dataset for training embedding models and retrieval",
        ],
    },
    {
        "id": "robotics",
        "name": "Robotics & Reinforcement Learning",
        "categories": ["robotics", "reinforcement-learning"],
        "queries": [
            "robotics dataset with manipulation trajectories and actions",
            "reinforcement learning environment dataset with rewards",
            "robot navigation and control dataset",
        ],
    },
    {
        "id": "tabular",
        "name": "Tabular & Structured Data",
        "categories": ["tabular-classification", "tabular-regression"],
        "queries": [
            "tabular dataset for regression or classification tasks",
            "structured data with numerical features for prediction",
            "CSV dataset for machine learning benchmarking",
        ],
    },
    {
        "id": "audio",
        "name": "Audio & Speech",
        "categories": [
            "automatic-speech-recognition", "audio-classification",
            "text-to-speech", "voice-activity-detection",
        ],
        "queries": [
            "speech recognition dataset with audio and transcriptions",
            "audio classification dataset for sound event detection",
            "text to speech dataset with speaker recordings",
        ],
    },
]

# Cross-group queries
CROSS_QUERIES = [
    {
        "query": "multimodal dataset combining images and text descriptions",
        "relevant_groups": ["vision", "qa"],
    },
    {
        "query": "dataset for training a chatbot or conversational AI",
        "relevant_groups": ["text-generation", "qa"],
    },
    {
        "query": "benchmark dataset for evaluating NLP models across multiple tasks",
        "relevant_groups": ["classification", "qa", "ner"],
    },
    {
        "query": "dataset for autonomous driving with sensor data",
        "relevant_groups": ["vision", "robotics"],
    },
    {
        "query": "code generation dataset with programming problems and solutions",
        "relevant_groups": ["text-generation"],
    },
]

# All task categories flattened for matching
ALL_CATEGORIES: dict[str, str] = {}  # category → group_id
for _g in TASK_GROUPS:
    for _cat in _g["categories"]:
        ALL_CATEGORIES[_cat] = _g["id"]


# ------------------------------------------------------------------ #
# HuggingFace Hub fetcher                                              #
# ------------------------------------------------------------------ #

HF_API = "https://huggingface.co/api/datasets"
CACHE_DIR = Path("experiments/data/hf_cache")


@dataclass
class DatasetInfo:
    dataset_id: str
    description: str
    task_categories: list[str]
    tags: list[str]
    downloads: int
    group: str  # assigned group from TASK_GROUPS


def clean_description(raw: str) -> str:
    """Clean HTML/markdown noise from HuggingFace descriptions."""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", raw)
    # Remove markdown images
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_datasets(target_count: int = 500) -> list[DatasetInfo]:
    """Fetch datasets from HuggingFace Hub API, querying by task category."""
    datasets: list[DatasetInfo] = []
    seen_ids: set[str] = set()
    target_per_group = max(target_count // len(TASK_GROUPS), 20)

    # Collect all unique categories to query
    category_to_group: list[tuple[str, str]] = []
    for g in TASK_GROUPS:
        for cat in g["categories"]:
            category_to_group.append((cat, g["id"]))

    print(f"  Fetching datasets from HuggingFace Hub by task category...")

    for cat, group_id in category_to_group:
        # Fetch datasets for this specific task category
        per_cat = min(target_per_group, 100)
        url = (
            f"{HF_API}?limit={per_cat}&sort=downloads&direction=-1"
            f"&filter=task_categories:{cat}&full=true"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "vstash-experiment/0.8"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"    ⚠ Failed for {cat}: {e}")
            continue

        added = 0
        for d in data:
            did = d.get("id", "")
            if did in seen_ids or not did:
                continue
            seen_ids.add(did)

            # Extract description
            desc = clean_description(d.get("description", ""))
            card = d.get("cardData", {})
            pretty_name = card.get("pretty_name", "")
            if pretty_name and len(desc) < 50:
                desc = f"{pretty_name}. {desc}".strip()
            if len(desc) < 20:
                desc = did.replace("/", " - ").replace("-", " ").replace("_", " ")

            # Extract task categories from tags
            tags = d.get("tags", [])
            task_cats = [
                t.replace("task_categories:", "")
                for t in tags
                if t.startswith("task_categories:")
            ]
            if not task_cats:
                task_cats = [cat]

            datasets.append(
                DatasetInfo(
                    dataset_id=did,
                    description=desc[:2000],
                    task_categories=task_cats,
                    tags=[t for t in tags if not t.startswith("task_categories:")],
                    downloads=d.get("downloads", 0),
                    group=group_id,
                )
            )
            added += 1

        print(f"    {cat:<45} → {added} datasets")
        time.sleep(0.3)

    print(f"  Total: {len(datasets)} datasets")
    return datasets


def download_corpus(target_count: int = 500) -> list[DatasetInfo]:
    """Download with caching."""
    cache_file = CACHE_DIR / f"datasets_{target_count}.json"

    if cache_file.exists():
        print(f"  Loading cached corpus from {cache_file}")
        data = json.loads(cache_file.read_text())
        return [DatasetInfo(**d) for d in data]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    datasets = fetch_datasets(target_count=target_count)

    cache_data = [asdict(d) for d in datasets]
    cache_file.write_text(json.dumps(cache_data, indent=2))
    print(f"  Cached {len(datasets)} datasets to {cache_file}")

    return datasets


# ------------------------------------------------------------------ #
# Ingestion                                                            #
# ------------------------------------------------------------------ #

MODEL = "BAAI/bge-small-en-v1.5"
DIMS = 384


def ingest_datasets(
    datasets: list[DatasetInfo], store: VstashStore
) -> dict[str, DatasetInfo]:
    """Ingest datasets into vstash. Returns path → DatasetInfo mapping."""
    path_map: dict[str, DatasetInfo] = {}

    print(f"\n  Ingesting {len(datasets)} datasets...")
    t0 = time.time()

    # Build texts: dataset_id + description + tags as document
    texts = []
    for ds in datasets:
        tag_str = ", ".join(ds.task_categories)
        text = f"# {ds.dataset_id}\n\nTasks: {tag_str}\n\n{ds.description}"
        texts.append(text)

    # Batch embed
    batch_size = 64
    all_embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        all_embeddings.extend(embed_texts(batch, model_name=MODEL, backend="onnx"))
        if start + batch_size < len(texts):
            print(f"    Embedded {start + len(batch)}/{len(texts)}...")

    # Ingest
    for i, (ds, emb) in enumerate(zip(datasets, all_embeddings)):
        path = f"hf://{ds.dataset_id}"

        store.add_document(
            path=path,
            title=ds.dataset_id,
            chunks=[texts[i]],
            embeddings=[emb],
            source_type="huggingface",
            collection=ds.group,
            project="dataset-discovery",
        )
        path_map[path] = ds

        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(datasets)} ingested...")

    elapsed = time.time() - t0
    print(f"  Ingested {len(path_map)} datasets in {elapsed:.1f}s")
    return path_map


# ------------------------------------------------------------------ #
# Evaluation metrics                                                   #
# ------------------------------------------------------------------ #


def dcg_at_k(relevances: list[float], k: int) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))


def ndcg_at_k(relevances: list[float], ideal: list[float], k: int) -> float:
    idcg = dcg_at_k(sorted(ideal, reverse=True), k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(relevances, k) / idcg


def precision_at_k(relevances: list[float], k: int) -> float:
    top_k = relevances[:k]
    if not top_k:
        return 0.0
    return sum(1 for r in top_k if r > 0) / len(top_k)


def mrr(relevances: list[float]) -> float:
    for i, r in enumerate(relevances):
        if r > 0:
            return 1.0 / (i + 1)
    return 0.0


# ------------------------------------------------------------------ #
# Benchmark                                                            #
# ------------------------------------------------------------------ #


@dataclass
class QueryResult:
    query: str
    target_groups: list[str]
    precision_5: float
    ndcg_5: float
    mrr_score: float
    hit_at_5: bool  # at least one relevant in top-5
    top_results: list[dict]  # for display


@dataclass
class BenchmarkResult:
    corpus_size: int
    num_queries: int
    mean_precision_5: float
    mean_ndcg_5: float
    mean_mrr: float
    discovery_rate: float  # fraction with hit@5
    per_query: list[QueryResult] = field(default_factory=list)


def run_evaluation(
    store: VstashStore,
    path_map: dict[str, DatasetInfo],
    top_k: int = 10,
) -> BenchmarkResult:
    """Run all benchmark queries."""
    results: list[QueryResult] = []

    # Build queries
    all_queries: list[dict] = []
    for group in TASK_GROUPS:
        for q in group["queries"]:
            all_queries.append({"query": q, "target_groups": [group["id"]]})
    for cq in CROSS_QUERIES:
        all_queries.append(
            {"query": cq["query"], "target_groups": cq["relevant_groups"]}
        )

    scoring = ScoringConfig(enabled=False)

    for qinfo in all_queries:
        query_text = qinfo["query"]
        target_groups = qinfo["target_groups"]

        query_emb = embed_query(query_text, model_name=MODEL, backend="onnx")
        chunks = store.search(
            query_embedding=query_emb,
            query_text=query_text,
            top_k=top_k,
            scoring=scoring,
        )

        relevances: list[float] = []
        top_results: list[dict] = []

        for chunk in chunks:
            ds = path_map.get(chunk.path)
            if ds:
                rel = 1.0 if ds.group in target_groups else 0.0
                relevances.append(rel)
                top_results.append({
                    "dataset": ds.dataset_id,
                    "group": ds.group,
                    "tasks": ds.task_categories,
                    "relevant": rel > 0,
                    "score": round(chunk.score, 3),
                })
            else:
                relevances.append(0.0)

        total_relevant = sum(
            1 for ds in path_map.values() if ds.group in target_groups
        )
        ideal = sorted(
            relevances + [1.0] * max(0, total_relevant - len(relevances)),
            reverse=True,
        )

        results.append(
            QueryResult(
                query=query_text,
                target_groups=target_groups,
                precision_5=precision_at_k(relevances, 5),
                ndcg_5=ndcg_at_k(relevances, ideal, 5),
                mrr_score=mrr(relevances),
                hit_at_5=any(r > 0 for r in relevances[:5]),
                top_results=top_results[:5],
            )
        )

    num_q = len(results)
    return BenchmarkResult(
        corpus_size=len(path_map),
        num_queries=num_q,
        mean_precision_5=sum(q.precision_5 for q in results) / num_q,
        mean_ndcg_5=sum(q.ndcg_5 for q in results) / num_q,
        mean_mrr=sum(q.mrr_score for q in results) / num_q,
        discovery_rate=sum(1 for q in results if q.hit_at_5) / num_q,
        per_query=results,
    )


# ------------------------------------------------------------------ #
# Interactive mode                                                     #
# ------------------------------------------------------------------ #


def interactive_mode(store: VstashStore, path_map: dict[str, DatasetInfo]) -> None:
    """Interactive dataset discovery REPL."""
    print("\n  Dataset Discovery Engine")
    print("  Describe what you need. Type 'quit' to exit.\n")

    scoring = ScoringConfig(enabled=True, track_access=True)

    while True:
        try:
            query = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            break

        query_emb = embed_query(query, model_name=MODEL, backend="onnx")
        chunks = store.search(
            query_embedding=query_emb,
            query_text=query,
            top_k=5,
            scoring=scoring,
        )

        print()
        for i, chunk in enumerate(chunks):
            ds = path_map.get(chunk.path)
            if ds:
                tasks = ", ".join(ds.task_categories[:3])
                downloads = f"{ds.downloads:,}" if ds.downloads else "?"
                print(f"  {i+1}. {ds.dataset_id}")
                print(f"     Tasks: {tasks} | Downloads: {downloads} | Score: {chunk.score:.3f}")
                # Show first 150 chars of description
                desc_preview = ds.description[:150].replace("\n", " ")
                print(f"     {desc_preview}...")
                print()

        print()


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset Discovery Engine")
    parser.add_argument(
        "--datasets", type=int, default=500,
        help="Number of datasets to fetch (default: 500)",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Use cached corpus",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Interactive discovery mode",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  Dataset Discovery Engine — vstash")
    print("=" * 70)

    # Step 1: Download corpus
    if args.skip_download:
        cache_file = CACHE_DIR / f"datasets_{args.datasets}.json"
        if not cache_file.exists():
            print(f"  Cache not found, downloading...")
            datasets = download_corpus(target_count=args.datasets)
        else:
            data = json.loads(cache_file.read_text())
            datasets = [DatasetInfo(**d) for d in data]
            print(f"  Loaded {len(datasets)} datasets from cache")
    else:
        datasets = download_corpus(target_count=args.datasets)

    print(f"\n  Corpus: {len(datasets)} datasets")

    # Group distribution
    group_counts: dict[str, int] = {}
    for ds in datasets:
        group_counts[ds.group] = group_counts.get(ds.group, 0) + 1
    for g in TASK_GROUPS:
        count = group_counts.get(g["id"], 0)
        print(f"    {g['id']:<20} {g['name']:<35} {count:>4}")

    # Step 2: Ingest
    db_path = Path(tempfile.mkdtemp()) / "dataset_discovery.db"
    store = VstashStore(str(db_path), embedding_dim=DIMS)
    path_map = ingest_datasets(datasets, store)

    if args.interactive:
        interactive_mode(store, path_map)
        store.close()
        return

    # Step 3: Evaluate
    print("\n  Running benchmark...")
    t0 = time.time()
    result = run_evaluation(store, path_map)
    elapsed = time.time() - t0

    # Step 4: Results
    print(f"\n{'='*70}")
    print("  RESULTS")
    print(f"{'='*70}")
    print(f"  Corpus:          {result.corpus_size} datasets")
    print(f"  Queries:         {result.num_queries}")
    print(f"  Precision@5:     {result.mean_precision_5:.3f}")
    print(f"  NDCG@5:          {result.mean_ndcg_5:.3f}")
    print(f"  MRR:             {result.mean_mrr:.3f}")
    print(f"  Discovery rate:  {result.discovery_rate:.1%}")
    print(f"  Time:            {elapsed:.1f}s")

    # Show sample queries
    print(f"\n  Sample results:")
    for qr in result.per_query[:5]:
        print(f"\n    Q: \"{qr.query}\"")
        print(f"    Targets: {qr.target_groups} | P@5={qr.precision_5:.2f} | MRR={qr.mrr_score:.2f}")
        for tr in qr.top_results[:3]:
            marker = "✓" if tr["relevant"] else "✗"
            print(f"      {marker} {tr['dataset']} ({tr['group']}) — {tr['score']:.3f}")

    # Step 5: Save
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "dataset_discovery.json"

    output_data = {
        "experiment": "dataset_discovery",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "corpus_size": result.corpus_size,
        "num_queries": result.num_queries,
        "mean_precision_5": result.mean_precision_5,
        "mean_ndcg_5": result.mean_ndcg_5,
        "mean_mrr": result.mean_mrr,
        "discovery_rate": result.discovery_rate,
        "per_query": [asdict(q) for q in result.per_query],
    }
    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"\n  Results saved to {output_path}")

    store.close()


if __name__ == "__main__":
    main()
