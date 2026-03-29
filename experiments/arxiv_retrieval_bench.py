"""ArXiv Retrieval Benchmark — Kaggle-scale evaluation of vstash

Downloads ~1,000 ML paper abstracts from HuggingFace (CShorten/ML-ArXiv-Papers),
assigns topic labels via keyword classification, and evaluates retrieval quality.

Ground truth: each paper is classified into one of 10 topic clusters based on
abstract keywords. Queries target specific topics. A retrieved paper is "relevant"
if it belongs to the target topic. This gives us automatic relevance labels at
scale — no manual annotation needed.

Metrics:
  - Precision@k: fraction of top-k results that are relevant
  - NDCG@k: normalized discounted cumulative gain (graded by topic match)
  - MRR: mean reciprocal rank of first relevant result
  - Recall@k: fraction of relevant docs found in top-k

Ablations:
  - Embedding models: BGE-small (384) vs multilingual-MiniLM (384) vs BGE-base (768)
  - Search modes: vector-only vs FTS-only vs hybrid RRF
  - Scoring: off vs on
  - MMR dedup: off (lambda=1.0) vs on (lambda=0.5)

Usage:
    python -m experiments.arxiv_retrieval_bench
    python -m experiments.arxiv_retrieval_bench --papers 1000 --top-k 10
    python -m experiments.arxiv_retrieval_bench --skip-download  # reuse cached corpus
    python -m experiments.arxiv_retrieval_bench --models bge-small  # single model

Output: experiments/results/arxiv_bench.json
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
# Topic clusters and queries                                           #
# ------------------------------------------------------------------ #

# 10 ML topic clusters with keyword patterns for classification
# and natural-language benchmark queries.
TOPICS: list[dict] = [
    {
        "id": "nlp",
        "name": "Natural Language Processing",
        "keywords": [
            "language model", "text classification", "sentiment", "translation",
            "named entity", "question answering", "summarization", "parsing",
            "word embedding", "tokeniz", "bert", "gpt", "transformer",
            "seq2seq", "attention mechanism", "corpus", "nlp", "dialogue",
            "text generation", "language understanding",
        ],
        "queries": [
            "language models and text generation techniques",
            "sentiment analysis and opinion mining from text",
            "neural machine translation and multilingual models",
        ],
    },
    {
        "id": "cv",
        "name": "Computer Vision",
        "keywords": [
            "image classification", "object detection", "segmentation",
            "convolutional neural", "cnn", "visual", "image recognition",
            "face recognition", "image generation", "gan", "pixel",
            "resnet", "yolo", "image feature", "scene understanding",
            "optical flow", "video understanding", "pose estimation",
        ],
        "queries": [
            "image classification and object detection with deep learning",
            "image segmentation and feature extraction methods",
            "generative adversarial networks for image synthesis",
        ],
    },
    {
        "id": "rl",
        "name": "Reinforcement Learning",
        "keywords": [
            "reinforcement learning", "reward", "policy gradient", "q-learning",
            "markov decision", "exploration", "multi-armed bandit", "actor-critic",
            "temporal difference", "sarsa", "value function", "bellman",
            "model-based rl", "model-free", "on-policy", "off-policy",
        ],
        "queries": [
            "reinforcement learning policy optimization and reward shaping",
            "exploration strategies in multi-armed bandit problems",
            "model-based reinforcement learning for decision making",
        ],
    },
    {
        "id": "optimization",
        "name": "Optimization and Training",
        "keywords": [
            "gradient descent", "stochastic optimization", "adam optimizer",
            "learning rate", "convergence", "loss function", "backpropagation",
            "batch normalization", "dropout", "regularization", "weight decay",
            "hyperparameter", "momentum", "sgd", "optimization landscape",
        ],
        "queries": [
            "stochastic gradient descent and optimization convergence",
            "regularization techniques to prevent overfitting",
            "hyperparameter tuning and learning rate schedules",
        ],
    },
    {
        "id": "graphs",
        "name": "Graph Neural Networks",
        "keywords": [
            "graph neural", "graph convolution", "node classification",
            "link prediction", "graph embedding", "knowledge graph",
            "message passing", "graph attention", "spectral graph",
            "community detection", "graph generation", "molecular graph",
        ],
        "queries": [
            "graph neural networks for node classification",
            "knowledge graph embedding and link prediction",
            "message passing neural networks on molecular graphs",
        ],
    },
    {
        "id": "generative",
        "name": "Generative Models",
        "keywords": [
            "variational autoencoder", "vae", "generative adversarial",
            "diffusion model", "flow-based", "normalizing flow",
            "latent space", "decoder", "image synthesis", "score matching",
            "denoising", "sampling", "likelihood", "generative model",
        ],
        "queries": [
            "variational autoencoders and latent space representations",
            "diffusion models for high quality image generation",
            "generative modeling and density estimation techniques",
        ],
    },
    {
        "id": "federated",
        "name": "Federated and Distributed Learning",
        "keywords": [
            "federated learning", "distributed training", "communication efficiency",
            "privacy preserving", "differential privacy", "data heterogeneity",
            "model aggregation", "gradient compression", "decentralized",
            "parallel training", "data parallel", "model parallel",
        ],
        "queries": [
            "federated learning with privacy preserving aggregation",
            "distributed training and communication efficiency",
            "differential privacy in machine learning systems",
        ],
    },
    {
        "id": "transfer",
        "name": "Transfer Learning and Few-Shot",
        "keywords": [
            "transfer learning", "domain adaptation", "few-shot", "zero-shot",
            "meta-learning", "pre-training", "fine-tuning", "self-supervised",
            "contrastive learning", "representation learning", "prompt",
            "in-context learning", "foundation model",
        ],
        "queries": [
            "transfer learning and domain adaptation strategies",
            "few-shot and zero-shot learning with meta-learning",
            "self-supervised pre-training and contrastive learning",
        ],
    },
    {
        "id": "fairness",
        "name": "Fairness and Interpretability",
        "keywords": [
            "fairness", "bias", "interpretab", "explainab", "saliency",
            "feature importance", "attention visualization", "lime", "shap",
            "counterfactual", "causal", "algorithmic fairness", "transparency",
            "trustworth", "accountability",
        ],
        "queries": [
            "algorithmic fairness and bias mitigation in ML models",
            "interpretability and explainability of neural networks",
            "causal inference and counterfactual explanations",
        ],
    },
    {
        "id": "timeseries",
        "name": "Time Series and Forecasting",
        "keywords": [
            "time series", "forecasting", "temporal", "recurrent neural",
            "lstm", "gru", "autoregressive", "anomaly detection",
            "sequence modeling", "signal processing", "wavelet",
            "seasonal", "trend", "prediction horizon",
        ],
        "queries": [
            "time series forecasting with recurrent neural networks",
            "anomaly detection in temporal data sequences",
            "autoregressive models for sequence prediction",
        ],
    },
]

# Cross-topic queries (relevant to multiple topics)
CROSS_QUERIES = [
    {
        "query": "using transformers for image and text understanding",
        "relevant_topics": ["nlp", "cv"],
    },
    {
        "query": "generative models with reinforcement learning fine-tuning",
        "relevant_topics": ["generative", "rl"],
    },
    {
        "query": "fairness in federated learning across distributed clients",
        "relevant_topics": ["fairness", "federated"],
    },
    {
        "query": "graph neural networks for knowledge representation in NLP",
        "relevant_topics": ["graphs", "nlp"],
    },
    {
        "query": "few-shot time series forecasting with meta-learning",
        "relevant_topics": ["transfer", "timeseries"],
    },
]


# ------------------------------------------------------------------ #
# Data source: HuggingFace datasets API                                #
# ------------------------------------------------------------------ #

HF_API = "https://datasets-server.huggingface.co/rows"
HF_DATASET = "CShorten/ML-ArXiv-Papers"
CACHE_DIR = Path("experiments/data/arxiv_cache")


@dataclass
class Paper:
    title: str
    abstract: str
    topic: str  # assigned via keyword classification


def classify_topic(title: str, abstract: str) -> str | None:
    """Classify a paper into a topic based on keyword matching."""
    text = (title + " " + abstract).lower()
    scores: dict[str, int] = {}
    for topic in TOPICS:
        count = 0
        for kw in topic["keywords"]:
            count += len(re.findall(re.escape(kw), text))
        if count > 0:
            scores[topic["id"]] = count

    if not scores:
        return None
    # Return topic with highest keyword match count
    return max(scores, key=scores.get)  # type: ignore[arg-type]


def fetch_from_huggingface(total_papers: int = 1000) -> list[Paper]:
    """Fetch papers from HuggingFace datasets API."""
    papers: list[Paper] = []
    topic_counts: dict[str, int] = {t["id"]: 0 for t in TOPICS}
    target_per_topic = total_papers // len(TOPICS)

    offset = 0
    batch_size = 100
    max_offset = 10000  # don't scan the entire dataset

    print(f"  Fetching papers from HuggingFace ({HF_DATASET})...")

    while len(papers) < total_papers and offset < max_offset:
        url = f"{HF_API}?dataset={HF_DATASET}&config=default&split=train&offset={offset}&length={batch_size}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "vstash-experiment/0.8"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"    ⚠ Failed at offset {offset}: {e}")
            break

        rows = data.get("rows", [])
        if not rows:
            break

        for r in rows:
            row = r.get("row", {})
            title = row.get("title", "").strip()
            abstract = row.get("abstract", "").strip()

            if not title or not abstract or len(abstract) < 100:
                continue

            topic = classify_topic(title, abstract)
            if topic is None:
                continue

            # Balance: skip if this topic is already full
            if topic_counts[topic] >= target_per_topic * 2:
                continue

            papers.append(Paper(title=title, abstract=abstract, topic=topic))
            topic_counts[topic] += 1

            if len(papers) >= total_papers:
                break

        offset += batch_size
        print(f"    Scanned {offset} rows, collected {len(papers)} papers...")
        time.sleep(0.5)  # gentle rate limit

    return papers


def download_corpus(total_papers: int = 1000) -> list[Paper]:
    """Download papers with caching."""
    cache_file = CACHE_DIR / f"corpus_{total_papers}.json"

    if cache_file.exists():
        print(f"  Loading cached corpus from {cache_file}")
        data = json.loads(cache_file.read_text())
        return [Paper(**p) for p in data]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    papers = fetch_from_huggingface(total_papers=total_papers)

    # Cache
    cache_data = [{"title": p.title, "abstract": p.abstract, "topic": p.topic} for p in papers]
    cache_file.write_text(json.dumps(cache_data, indent=2))
    print(f"  Cached {len(papers)} papers to {cache_file}")

    return papers


# ------------------------------------------------------------------ #
# Ingestion                                                            #
# ------------------------------------------------------------------ #


def ingest_papers(
    papers: list[Paper], store: VstashStore, model: str, backend: BackendType = "onnx"
) -> dict[str, str]:
    """Ingest papers into vstash store. Returns path → topic mapping."""
    path_to_topic: dict[str, str] = {}

    print(f"\n  Ingesting {len(papers)} papers...")
    t0 = time.time()

    # Batch embed all texts (force ONNX for cross-model consistency)
    texts = [f"# {p.title}\n\n{p.abstract}" for p in papers]
    batch_size = 64
    all_embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        all_embeddings.extend(embed_texts(batch, model_name=model, backend=backend))
        if start + batch_size < len(texts):
            print(f"    Embedded {start + len(batch)}/{len(texts)}...")

    # Ingest via public API
    for i, (paper, emb) in enumerate(zip(papers, all_embeddings)):
        path = f"arxiv://{paper.topic}/{i}"
        text = texts[i]

        store.add_document(
            path=path,
            title=paper.title,
            chunks=[text],
            embeddings=[emb],
            source_type="arxiv",
            collection=paper.topic,
            project="arxiv-bench",
        )
        path_to_topic[path] = paper.topic

        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(papers)} ingested...")

    elapsed = time.time() - t0
    print(f"  Ingested {len(path_to_topic)} papers in {elapsed:.1f}s")
    return path_to_topic


# ------------------------------------------------------------------ #
# Evaluation metrics                                                   #
# ------------------------------------------------------------------ #


def dcg_at_k(relevances: list[float], k: int) -> float:
    """Discounted cumulative gain at k."""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))


def ndcg_at_k(relevances: list[float], ideal: list[float], k: int) -> float:
    """Normalized DCG at k."""
    idcg = dcg_at_k(sorted(ideal, reverse=True), k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(relevances, k) / idcg


def precision_at_k(relevances: list[float], k: int) -> float:
    """Fraction of top-k results that are relevant (rel > 0)."""
    top_k = relevances[:k]
    if not top_k:
        return 0.0
    return sum(1 for r in top_k if r > 0) / len(top_k)


def recall_at_k(relevances: list[float], total_relevant: int, k: int) -> float:
    """Fraction of relevant docs found in top-k."""
    if total_relevant == 0:
        return 0.0
    found = sum(1 for r in relevances[:k] if r > 0)
    return found / total_relevant


def mrr(relevances: list[float]) -> float:
    """Mean reciprocal rank of first relevant result."""
    for i, r in enumerate(relevances):
        if r > 0:
            return 1.0 / (i + 1)
    return 0.0


# ------------------------------------------------------------------ #
# Benchmark runner                                                     #
# ------------------------------------------------------------------ #


@dataclass
class QueryResult:
    query: str
    target_topics: list[str]
    precision_5: float
    precision_10: float
    ndcg_5: float
    ndcg_10: float
    mrr_score: float
    recall_10: float
    results_returned: int


@dataclass
class BenchmarkResult:
    model: str
    model_dims: int
    search_mode: str  # "hybrid", "vector", "fts"
    scoring: str  # "off", "on"
    mmr_lambda: float
    corpus_size: int
    num_queries: int
    mean_precision_5: float
    mean_precision_10: float
    mean_ndcg_5: float
    mean_ndcg_10: float
    mean_mrr: float
    mean_recall_10: float
    elapsed_s: float
    per_query: list[QueryResult] = field(default_factory=list)


def evaluate_queries(
    store: VstashStore,
    model: str,
    path_to_topic: dict[str, str],
    scoring_config: ScoringConfig | None,
    mmr_lambda: float,
    search_mode: str,
    top_k: int = 10,
) -> list[QueryResult]:
    """Run all queries and compute metrics."""
    results: list[QueryResult] = []

    # Build all queries: per-topic + cross-topic
    all_queries: list[dict] = []
    for topic_info in TOPICS:
        for q in topic_info["queries"]:
            all_queries.append(
                {"query": q, "target_topics": [topic_info["id"]]}
            )
    for cq in CROSS_QUERIES:
        all_queries.append(
            {"query": cq["query"], "target_topics": cq["relevant_topics"]}
        )

    # Override scoring config mmr_lambda
    if scoring_config:
        scoring_config = ScoringConfig(
            enabled=scoring_config.enabled,
            alpha=scoring_config.alpha,
            beta=scoring_config.beta,
            decay_lambda=scoring_config.decay_lambda,
            over_fetch=scoring_config.over_fetch,
            track_access=False,  # don't pollute access counts during eval
            mmr_lambda=mmr_lambda,
        )

    for qinfo in all_queries:
        query_text = qinfo["query"]
        target_topics = qinfo["target_topics"]

        # Embed query
        query_emb = embed_query(query_text, model_name=model, backend="onnx")

        # Search based on mode
        if search_mode == "vector":
            # Vector-only: pass empty query_text to skip FTS
            chunks = store.search(
                query_embedding=query_emb,
                query_text="",
                top_k=top_k,
                scoring=scoring_config,
            )
        elif search_mode == "fts":
            # FTS-only: use first few keywords as query
            chunks = store.search(
                query_embedding=query_emb,
                query_text=query_text,
                top_k=top_k,
                scoring=scoring_config,
            )
        else:
            # Hybrid (default)
            chunks = store.search(
                query_embedding=query_emb,
                query_text=query_text,
                top_k=top_k,
                scoring=scoring_config,
            )

        # Compute relevance for each result
        relevances: list[float] = []
        for chunk in chunks:
            paper_topic = path_to_topic.get(chunk.path, "")

            # Binary relevance: 1.0 if topic matches any target
            rel = 1.0 if paper_topic in target_topics else 0.0
            relevances.append(rel)

        # Count total relevant in corpus for recall
        total_relevant = sum(
            1 for t in path_to_topic.values() if t in target_topics
        )

        # Ideal ranking for NDCG
        ideal_rels = sorted(
            relevances + [1.0] * max(0, total_relevant - len(relevances)),
            reverse=True,
        )

        results.append(
            QueryResult(
                query=query_text,
                target_topics=target_topics,
                precision_5=precision_at_k(relevances, 5),
                precision_10=precision_at_k(relevances, 10),
                ndcg_5=ndcg_at_k(relevances, ideal_rels, 5),
                ndcg_10=ndcg_at_k(relevances, ideal_rels, 10),
                mrr_score=mrr(relevances),
                recall_10=recall_at_k(relevances, total_relevant, 10),
                results_returned=len(chunks),
            )
        )

    return results


# ------------------------------------------------------------------ #
# Model configurations                                                 #
# ------------------------------------------------------------------ #

MODEL_CONFIGS = {
    "bge-small": {
        "name": "BAAI/bge-small-en-v1.5",
        "dims": 384,
        "label": "BGE-small-EN (384d)",
    },
    "multilingual": {
        "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "dims": 384,
        "label": "Multilingual-MiniLM (384d)",
    },
    "bge-base": {
        "name": "BAAI/bge-base-en-v1.5",
        "dims": 768,
        "label": "BGE-base-EN (768d)",
    },
}


# ------------------------------------------------------------------ #
# Main experiment                                                      #
# ------------------------------------------------------------------ #


def run_benchmark(
    papers: list[Paper],
    model_key: str,
    top_k: int = 10,
) -> list[BenchmarkResult]:
    """Run all ablations for a single model."""
    model_cfg = MODEL_CONFIGS[model_key]
    model_name = model_cfg["name"]
    dims = model_cfg["dims"]

    print(f"\n{'='*70}")
    print(f"  Model: {model_cfg['label']}")
    print(f"{'='*70}")

    # Create temp DB and ingest
    db_path = Path(tempfile.mkdtemp()) / f"arxiv_bench_{model_key}.db"
    store = VstashStore(str(db_path), embedding_dim=dims)
    path_to_topic = ingest_papers(papers, store, model_name)

    corpus_size = len(path_to_topic)
    all_results: list[BenchmarkResult] = []

    # Ablation matrix
    configs = [
        {"search_mode": "hybrid", "scoring": "off", "mmr": 0.5, "label": "Hybrid RRF"},
        {"search_mode": "hybrid", "scoring": "off", "mmr": 1.0, "label": "Hybrid RRF (no MMR)"},
        {"search_mode": "vector", "scoring": "off", "mmr": 0.5, "label": "Vector-only"},
        {"search_mode": "fts", "scoring": "off", "mmr": 0.5, "label": "FTS-only"},
        {"search_mode": "hybrid", "scoring": "on", "mmr": 0.5, "label": "Hybrid + Scoring"},
    ]

    for cfg in configs:
        scoring_config = None
        if cfg["scoring"] == "on":
            scoring_config = ScoringConfig(
                enabled=True,
                alpha=0.8,
                beta=0.2,
                decay_lambda=0.05,
                over_fetch=50,
                track_access=False,
                mmr_lambda=cfg["mmr"],
            )
        elif cfg["mmr"] != 0.5:
            # Need scoring config just for mmr_lambda
            scoring_config = ScoringConfig(
                enabled=False,
                mmr_lambda=cfg["mmr"],
            )

        print(f"\n  Config: {cfg['label']}")
        t0 = time.time()

        query_results = evaluate_queries(
            store=store,
            model=model_name,
            path_to_topic=path_to_topic,
            scoring_config=scoring_config,
            mmr_lambda=cfg["mmr"],
            search_mode=cfg["search_mode"],
            top_k=top_k,
        )

        elapsed = time.time() - t0
        num_q = len(query_results)

        bench = BenchmarkResult(
            model=model_cfg["label"],
            model_dims=dims,
            search_mode=cfg["search_mode"],
            scoring=cfg["scoring"],
            mmr_lambda=cfg["mmr"],
            corpus_size=corpus_size,
            num_queries=num_q,
            mean_precision_5=sum(q.precision_5 for q in query_results) / num_q,
            mean_precision_10=sum(q.precision_10 for q in query_results) / num_q,
            mean_ndcg_5=sum(q.ndcg_5 for q in query_results) / num_q,
            mean_ndcg_10=sum(q.ndcg_10 for q in query_results) / num_q,
            mean_mrr=sum(q.mrr_score for q in query_results) / num_q,
            mean_recall_10=sum(q.recall_10 for q in query_results) / num_q,
            elapsed_s=round(elapsed, 2),
            per_query=query_results,
        )
        all_results.append(bench)

        print(
            f"    P@5={bench.mean_precision_5:.3f}  P@10={bench.mean_precision_10:.3f}  "
            f"NDCG@5={bench.mean_ndcg_5:.3f}  NDCG@10={bench.mean_ndcg_10:.3f}  "
            f"MRR={bench.mean_mrr:.3f}  R@10={bench.mean_recall_10:.3f}  "
            f"({elapsed:.1f}s)"
        )

    store.close()
    return all_results


def print_summary(all_results: list[BenchmarkResult]) -> None:
    """Print a comparison table."""
    print(f"\n{'='*90}")
    print("  RESULTS SUMMARY")
    print(f"{'='*90}")
    print(
        f"  {'Model':<28} {'Config':<22} {'P@5':>6} {'P@10':>6} "
        f"{'NDCG@5':>7} {'NDCG@10':>8} {'MRR':>6}"
    )
    print(f"  {'─'*85}")

    for r in all_results:
        label = f"{r.search_mode}"
        if r.scoring == "on":
            label += "+score"
        if r.mmr_lambda == 1.0:
            label += " (no MMR)"
        print(
            f"  {r.model:<28} {label:<22} "
            f"{r.mean_precision_5:>6.3f} {r.mean_precision_10:>6.3f} "
            f"{r.mean_ndcg_5:>7.3f} {r.mean_ndcg_10:>8.3f} {r.mean_mrr:>6.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="ArXiv Retrieval Benchmark")
    parser.add_argument(
        "--papers",
        type=int,
        default=1000,
        help="Total papers to fetch (default: 1000)",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-k for evaluation")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download, use cached corpus",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(MODEL_CONFIGS.keys()),
        choices=list(MODEL_CONFIGS.keys()),
        help="Which models to benchmark",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  ArXiv Retrieval Benchmark for vstash")
    print("=" * 70)

    # Step 1: Download corpus
    if args.skip_download:
        cache_file = CACHE_DIR / f"corpus_{args.papers}.json"
        if not cache_file.exists():
            print(f"  Cache not found at {cache_file}, downloading...")
            papers = download_corpus(total_papers=args.papers)
        else:
            data = json.loads(cache_file.read_text())
            papers = [Paper(**p) for p in data]
            print(f"  Loaded {len(papers)} papers from cache")
    else:
        papers = download_corpus(total_papers=args.papers)

    print(f"\n  Corpus: {len(papers)} papers across {len(TOPICS)} topics")

    # Topic distribution
    topic_counts: dict[str, int] = {}
    for p in papers:
        topic_counts[p.topic] = topic_counts.get(p.topic, 0) + 1
    for topic_info in TOPICS:
        tid = topic_info["id"]
        count = topic_counts.get(tid, 0)
        print(f"    {tid:<15} {topic_info['name']:<35} {count:>4} papers")

    # Step 2: Run benchmarks
    all_results: list[BenchmarkResult] = []

    for model_key in args.models:
        results = run_benchmark(papers, model_key, top_k=args.top_k)
        all_results.extend(results)

    # Step 3: Summary
    print_summary(all_results)

    # Step 4: Save results
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "arxiv_bench.json"

    output_data = {
        "experiment": "arxiv_retrieval_bench",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "corpus_size": len(papers),
        "topics": len(TOPICS),
        "queries": all_results[0].num_queries if all_results else 0,
        "top_k": args.top_k,
        "results": [asdict(r) for r in all_results],
    }

    # Strip per_query from saved results to keep file manageable
    for r in output_data["results"]:
        r.pop("per_query", None)

    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
