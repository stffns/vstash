"""Adaptive Cold Start Experiment — γ (maturity-gated scoring) vs fixed β.

Compares three scoring strategies on a 120-article Wikipedia corpus:
  1. baseline:  RRF only (no scoring)
  2. fixed:     Fixed β=0.5 from round 1 (current behavior pre-v0.7)
  3. adaptive:  β scaled by γ = outlier ratio (new behavior)

The corpus consists of 120 real Wikipedia articles across 12 topic clusters
(10 articles per cluster).  Articles are downloaded via the Wikipedia API on
first run and cached locally to ``experiments/data/wikipedia_corpus.json``
so subsequent runs require no network access.

Each article is chunked through vstash's real chunking pipeline (chunk_text)
with default parameters (1024-token chunks, 128-token overlap).
Target: ~3,000-5,000 total chunks from 120 articles.

Usage:
    python -m experiments.adaptive_cold_start [--rounds 30] [--output results/adaptive_cold_start.json]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import tempfile
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from vstash.config import ScoringConfig
from vstash.embed import embed_query, get_embedding_dim
from vstash.ingest import chunk_text
from vstash.store import VstashStore

MODEL = "BAAI/bge-small-en-v1.5"

# Default chunking parameters (vstash defaults)
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128


# ------------------------------------------------------------------ #
# Wikipedia article definitions: 12 clusters × 10 articles             #
# ------------------------------------------------------------------ #

WIKI_ARTICLES: dict[str, list[str]] = {
    "transformer_architecture": [
        "Transformer (deep learning)",
        "Attention (machine learning)",
        "BERT (language model)",
        "GPT-3",
        "Vision transformer",
        "Mixture of experts",
        "Transfer learning",
        "Seq2seq",
        "Neural machine translation",
        "Large language model",
    ],
    "reinforcement_learning": [
        "Reinforcement learning",
        "Q-learning",
        "Proximal policy optimization",
        "Deep reinforcement learning",
        "AlphaGo",
        "Multi-agent reinforcement learning",
        "Model-free (reinforcement learning)",
        "Temporal difference learning",
        "Markov decision process",
        "Monte Carlo tree search",
    ],
    "natural_language_processing": [
        "Natural language processing",
        "Named-entity recognition",
        "Sentiment analysis",
        "Machine translation",
        "Text mining",
        "Question answering",
        "Word embedding",
        "Recurrent neural network",
        "Part-of-speech tagging",
        "Semantic analysis (linguistics)",
    ],
    "computer_vision": [
        "Computer vision",
        "Object detection",
        "Image segmentation",
        "Convolutional neural network",
        "You Only Look Once",
        "Generative adversarial network",
        "Edge detection",
        "Optical character recognition",
        "Feature (computer vision)",
        "Facial recognition system",
    ],
    "database_systems": [
        "Database",
        "Database index",
        "Query optimization",
        "SQL",
        "PostgreSQL",
        "B-tree",
        "ACID",
        "Database normalization",
        "NoSQL",
        "Transaction processing",
    ],
    "distributed_systems": [
        "Distributed computing",
        "Paxos (computer science)",
        "Raft (algorithm)",
        "MapReduce",
        "Consensus (computer science)",
        "CAP theorem",
        "Distributed hash table",
        "Apache Kafka",
        "Load balancing (computing)",
        "Replication (computing)",
    ],
    "cryptography": [
        "Cryptography",
        "RSA cryptosystem",
        "Advanced Encryption Standard",
        "Diffie–Hellman key exchange",
        "Public-key cryptography",
        "Hash function",
        "Digital signature",
        "Transport Layer Security",
        "Elliptic-curve cryptography",
        "Block cipher",
    ],
    "operating_systems": [
        "Operating system",
        "Process (computing)",
        "Virtual memory",
        "File system",
        "Linux kernel",
        "Scheduling (computing)",
        "Interrupt",
        "Memory management",
        "Device driver",
        "System call",
    ],
    "graph_algorithms": [
        "Graph theory",
        "Dijkstra's algorithm",
        "PageRank",
        "Graph neural network",
        "Minimum spanning tree",
        "Breadth-first search",
        "Depth-first search",
        "Bellman–Ford algorithm",
        "Network flow problem",
        "Shortest path problem",
    ],
    "information_retrieval": [
        "Information retrieval",
        "Tf–idf",
        "Okapi BM25",
        "Search engine",
        "Inverted index",
        "Precision and recall",
        "Vector space model",
        "Latent semantic analysis",
        "Relevance feedback",
        "Google Search",
    ],
    "optimization_theory": [
        "Mathematical optimization",
        "Gradient descent",
        "Convex optimization",
        "Stochastic gradient descent",
        "Linear programming",
        "Genetic algorithm",
        "Simulated annealing",
        "Lagrange multiplier",
        "Newton's method",
        "Particle swarm optimization",
    ],
    "compiler_design": [
        "Compiler",
        "LLVM",
        "Abstract syntax tree",
        "Just-in-time compilation",
        "Parsing",
        "Lexical analysis",
        "Code generation (compiler)",
        "Register allocation",
        "Garbage collection (computer science)",
        "Intermediate representation",
    ],
}

# Topic names in the order they appear (matches EVAL_QUERIES indexing)
TOPIC_NAMES = list(WIKI_ARTICLES.keys())

# 10 eval queries with expected top results
EVAL_QUERIES = [
    {
        "query": "optimization algorithms for training deep neural networks",
        "relevant_clusters": {"transformer_architecture": 3, "reinforcement_learning": 2, "computer_vision": 1},
    },
    {
        "query": "learning representations from sequential data",
        "relevant_clusters": {"natural_language_processing": 3, "transformer_architecture": 2, "reinforcement_learning": 1},
    },
    {
        "query": "scalable indexing structures for fast similarity search",
        "relevant_clusters": {"database_systems": 3, "information_retrieval": 2, "distributed_systems": 1},
    },
    {
        "query": "security protocols for network communication",
        "relevant_clusters": {"cryptography": 3, "distributed_systems": 2, "operating_systems": 1},
    },
    {
        "query": "resource allocation and scheduling in computing systems",
        "relevant_clusters": {"operating_systems": 3, "distributed_systems": 2, "database_systems": 1},
    },
    {
        "query": "graph neural networks for structured prediction",
        "relevant_clusters": {"graph_algorithms": 3, "transformer_architecture": 2, "natural_language_processing": 1},
    },
    {
        "query": "feature extraction and pattern recognition in images",
        "relevant_clusters": {"computer_vision": 3, "natural_language_processing": 1, "transformer_architecture": 1},
    },
    {
        "query": "ranking models and relevance feedback in search",
        "relevant_clusters": {"information_retrieval": 3, "natural_language_processing": 2, "database_systems": 1},
    },
    {
        "query": "reward signals and policy optimization under uncertainty",
        "relevant_clusters": {"reinforcement_learning": 3, "graph_algorithms": 1, "operating_systems": 1},
    },
    {
        "query": "hash functions and authenticated data structures",
        "relevant_clusters": {"cryptography": 3, "database_systems": 2, "graph_algorithms": 1},
    },
]

# Zipf-weighted query distribution: some topics queried much more.
QUERY_WEIGHTS = [8, 6, 5, 4, 3, 3, 2, 2, 1, 1]


# ------------------------------------------------------------------ #
# Wikipedia corpus download and caching                                #
# ------------------------------------------------------------------ #

CACHE_PATH = Path(__file__).parent / "data" / "wikipedia_corpus.json"


def _fetch_wikipedia_article(title: str) -> str | None:
    """Fetch full article text from Wikipedia API.

    Includes a small delay to avoid rate limiting when fetching many articles.
    """
    api_title = title.replace(" ", "_")
    url = (
        "https://en.wikipedia.org/w/api.php?action=query&prop=extracts"
        f"&explaintext=true&titles={urllib.parse.quote(api_title)}&format=json"
    )
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "vstash-experiment/1.0"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                pages = data["query"]["pages"]
                for page_id, page in pages.items():
                    if page_id != "-1" and "extract" in page:
                        return page["extract"]
            break  # Got a response but no valid extract — don't retry
        except Exception:
            if attempt < 2:
                time.sleep(2)  # Wait before retry on network errors
    return None


def load_or_download_corpus() -> dict[str, dict[str, str]]:
    """Load cached corpus or download from Wikipedia.

    Returns:
        Dict mapping cluster_name -> {article_title: article_text}.
    """
    if CACHE_PATH.exists():
        print(f"  Loading cached corpus from {CACHE_PATH}")
        with open(CACHE_PATH) as f:
            return json.load(f)

    print("  Downloading articles from Wikipedia (first run only)...")
    corpus: dict[str, dict[str, str]] = {}
    total = sum(len(titles) for titles in WIKI_ARTICLES.values())
    downloaded = 0

    for cluster, titles in WIKI_ARTICLES.items():
        corpus[cluster] = {}
        for title in titles:
            downloaded += 1
            print(f"    [{downloaded}/{total}] Fetching: {title}")
            text = _fetch_wikipedia_article(title)
            if text and len(text) > 500:
                corpus[cluster][title] = text
            else:
                print(f"    WARNING: Could not fetch '{title}', skipping")
            time.sleep(0.5)  # Rate limit: avoid Wikipedia throttling

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(corpus, f)
    print(f"  Corpus cached to {CACHE_PATH}")
    return corpus


# ------------------------------------------------------------------ #
# Corpus ingestion                                                     #
# ------------------------------------------------------------------ #


def build_corpus(store: VstashStore) -> int:
    """Download (or load cached) Wikipedia articles and ingest them.

    Each article is chunked through vstash's real chunking pipeline
    (chunk_text) with default parameters before embedding and ingestion.

    Returns:
        Total number of chunks ingested.
    """
    corpus = load_or_download_corpus()

    total_chunks = 0
    total_docs = 0
    chunk_counts: list[int] = []

    for cluster_name, articles in corpus.items():
        for doc_index, (title, text) in enumerate(articles.items()):
            # Use vstash's real chunking pipeline with default parameters
            chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

            if not chunks:
                # Fallback: if chunking produces nothing, use the whole text
                chunks = [text]

            # Embed each chunk
            embeddings = [embed_query(c, MODEL) for c in chunks]

            # Use a sanitized title for the path
            safe_title = title.replace("/", "_").replace(" ", "_")
            path = f"/corpus/{cluster_name}/{safe_title}.md"

            store.add_document(
                path=path,
                title=title,
                chunks=chunks,
                embeddings=embeddings,
                source_type="markdown",
            )
            total_chunks += len(chunks)
            total_docs += 1
            chunk_counts.append(len(chunks))

    # Print corpus stats
    avg_chunks = sum(chunk_counts) / len(chunk_counts) if chunk_counts else 0
    print(f"  Corpus stats: {total_docs} docs, {total_chunks} total chunks, "
          f"{avg_chunks:.1f} avg chunks/doc")

    return total_chunks


# ------------------------------------------------------------------ #
# NDCG computation                                                     #
# ------------------------------------------------------------------ #


def ndcg_at_k(
    result_clusters: list[str],
    relevant_clusters: dict[str, int],
    k: int = 5,
    docs_per_cluster: int = 10,
) -> float:
    """Compute NDCG@k using graded relevance from cluster relevance map.

    Args:
        result_clusters: Cluster names for each result position.
        relevant_clusters: Dict mapping cluster name -> relevance grade (0-3).
        k: Cutoff position.
        docs_per_cluster: Number of documents per cluster (for ideal DCG).
    """
    dcg = 0.0
    for i, cluster in enumerate(result_clusters[:k]):
        rel = float(relevant_clusters.get(cluster, 0))
        dcg += rel / math.log2(i + 2)

    # Ideal: each cluster can contribute up to docs_per_cluster results,
    # all at the same relevance grade.  Sort all available grades descending.
    ideal_rels: list[float] = []
    for cluster, grade in relevant_clusters.items():
        ideal_rels.extend([float(grade)] * docs_per_cluster)
    ideal_rels.sort(reverse=True)
    ideal_rels = ideal_rels[:k]
    while len(ideal_rels) < k:
        ideal_rels.append(0.0)
    ideal_dcg = sum(ideal_rels[i] / math.log2(i + 2) for i in range(k))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def get_cluster_for_path(path: str) -> str:
    """Extract cluster name from document path."""
    # /corpus/transformer_architecture/Transformer_(...).md -> transformer_architecture
    parts = path.split("/")
    if len(parts) >= 3:
        return parts[2]
    return "unknown"


# ------------------------------------------------------------------ #
# Usage simulation                                                     #
# ------------------------------------------------------------------ #


def simulate_usage_round(
    store: VstashStore,
    query_embeddings: dict[str, list[float]],
    round_num: int,
    top_k: int = 10,
) -> int:
    """Simulate one round of non-uniform user queries.  Returns access count."""
    rng = random.Random(42 + round_num)

    # Build weighted query list
    weighted_queries = []
    for eq, weight in zip(EVAL_QUERIES, QUERY_WEIGHTS):
        weighted_queries.extend([eq] * weight)

    n_queries = min(5 + round_num * 2, len(weighted_queries))
    selected = rng.sample(weighted_queries, k=min(n_queries, len(weighted_queries)))

    total_tracked = 0
    for eq in selected:
        q = eq["query"]
        results = store.search(
            query_embedding=query_embeddings[q],
            query_text=q,
            top_k=top_k,
            scoring=None,  # Usage queries build history, don't use scoring
        )

        chunk_ids = []
        for r in results[:top_k]:
            row = store._conn.execute(
                "SELECT id FROM chunks WHERE text = ? LIMIT 1", [r.text]
            ).fetchone()
            if row:
                chunk_ids.append(row["id"])
        if chunk_ids:
            store.track_access(chunk_ids)
            total_tracked += len(chunk_ids)

    return total_tracked


# ------------------------------------------------------------------ #
# Data classes                                                         #
# ------------------------------------------------------------------ #


@dataclass
class RoundResult:
    round_num: int
    baseline_ndcg: float
    fixed_ndcg: float
    adaptive_ndcg: float
    gamma: float
    effective_beta: float
    total_accesses: int
    max_access: int
    mean_access: float


@dataclass
class ExperimentResult:
    rounds: list[RoundResult]
    total_rounds: int
    corpus_docs: int
    corpus_chunks: int
    n_queries: int
    fixed_crossover: int | None
    adaptive_crossover: int | None
    fixed_degradation_rounds: int
    adaptive_degradation_rounds: int


# ------------------------------------------------------------------ #
# Experiment                                                           #
# ------------------------------------------------------------------ #


def run_experiment(
    store: VstashStore,
    n_rounds: int = 30,
    top_k: int = 5,
    alpha: float = 0.5,
    beta: float = 0.5,
    decay_lambda: float = 0.10,
) -> ExperimentResult:
    """Run the adaptive vs fixed scoring experiment."""

    fixed_scoring = ScoringConfig(
        enabled=True, alpha=alpha, beta=beta,
        decay_lambda=decay_lambda, over_fetch=50, track_access=False,
    )
    # Baseline uses same over_fetch pool but gamma=0 (pure RRF + dedup)
    baseline_scoring = ScoringConfig(
        enabled=True, alpha=alpha, beta=beta,
        decay_lambda=decay_lambda, over_fetch=50, track_access=False,
    )

    # Pre-compute query embeddings
    query_embeddings = {}
    for eq in EVAL_QUERIES:
        query_embeddings[eq["query"]] = embed_query(eq["query"], MODEL)

    # Baseline: RRF only with same candidate pool (gamma forced to 0)
    baseline_ndcgs = []
    for eq in EVAL_QUERIES:
        q = eq["query"]
        results = store.search(
            query_embedding=query_embeddings[q], query_text=q,
            top_k=top_k, scoring=baseline_scoring,
            _gamma_override=0.0,  # force pure RRF
        )
        clusters = [get_cluster_for_path(r.path) for r in results]
        baseline_ndcgs.append(ndcg_at_k(clusters, eq["relevant_clusters"], k=top_k))
    avg_baseline = sum(baseline_ndcgs) / len(baseline_ndcgs)

    # Reset access history
    store._conn.execute("UPDATE chunks SET access_count = 0, last_accessed_at = NULL")
    store._conn.commit()

    rounds: list[RoundResult] = []
    cumulative_accesses = 0

    for round_num in range(n_rounds):
        # Phase A: simulate usage
        usage = simulate_usage_round(store, query_embeddings, round_num, top_k)
        cumulative_accesses += usage

        # Read gamma from the store (adaptive)
        gamma = store.scoring_maturity()
        effective_beta = beta * gamma

        # Access stats
        row = store._conn.execute(
            "SELECT AVG(access_count) as mean, MAX(access_count) as mx "
            "FROM chunks WHERE access_count > 0"
        ).fetchone()
        mean_acc = float(row["mean"]) if row and row["mean"] else 0.0
        max_acc = int(row["mx"]) if row and row["mx"] else 0

        # Phase B: evaluate with FIXED scoring (gamma=1.0 always — old behavior)
        fixed_ndcgs = []
        for eq in EVAL_QUERIES:
            q = eq["query"]
            results = store.search(
                query_embedding=query_embeddings[q], query_text=q,
                top_k=top_k, scoring=fixed_scoring,
                _gamma_override=1.0,  # force full beta — simulates pre-v0.7
            )
            clusters = [get_cluster_for_path(r.path) for r in results]
            fixed_ndcgs.append(ndcg_at_k(clusters, eq["relevant_clusters"], k=top_k))
        avg_fixed = sum(fixed_ndcgs) / len(fixed_ndcgs)

        # Phase C: evaluate with ADAPTIVE scoring (gamma computed from data)
        adaptive_ndcgs = []
        for eq in EVAL_QUERIES:
            q = eq["query"]
            results = store.search(
                query_embedding=query_embeddings[q], query_text=q,
                top_k=top_k, scoring=fixed_scoring,
                # _gamma_override not set -> uses real scoring_maturity()
            )
            clusters = [get_cluster_for_path(r.path) for r in results]
            adaptive_ndcgs.append(ndcg_at_k(clusters, eq["relevant_clusters"], k=top_k))
        avg_adaptive = sum(adaptive_ndcgs) / len(adaptive_ndcgs)

        rounds.append(RoundResult(
            round_num=round_num + 1,
            baseline_ndcg=avg_baseline,
            fixed_ndcg=avg_fixed,
            adaptive_ndcg=avg_adaptive,
            gamma=gamma,
            effective_beta=effective_beta,
            total_accesses=cumulative_accesses,
            max_access=max_acc,
            mean_access=round(mean_acc, 2),
        ))

    # Phase D: inject a strong outlier to simulate a "power user" pattern
    # (one topic queried 50x more than others — like a researcher's focus area)
    print("\n  --- Injecting power-user pattern (50x boost on transformers) ---\n")
    transformer_chunks = store._conn.execute(
        "SELECT c.id FROM chunks c JOIN documents d ON d.id = c.doc_id "
        "WHERE d.path LIKE '%transformer%'"
    ).fetchall()
    for row in transformer_chunks:
        store._conn.execute(
            "UPDATE chunks SET access_count = access_count + 50 WHERE id = ?",
            [row["id"]],
        )
    store._conn.commit()

    # Re-evaluate after power-user boost
    gamma_post = store.scoring_maturity()
    row = store._conn.execute(
        "SELECT AVG(access_count) as mean, MAX(access_count) as mx "
        "FROM chunks WHERE access_count > 0"
    ).fetchone()
    mean_post = float(row["mean"]) if row["mean"] else 0
    max_post = int(row["mx"]) if row["mx"] else 0

    fixed_post = []
    adaptive_post = []
    for eq in EVAL_QUERIES:
        q = eq["query"]
        # Fixed
        r_fixed = store.search(
            query_embedding=query_embeddings[q], query_text=q,
            top_k=top_k, scoring=fixed_scoring, _gamma_override=1.0,
        )
        clusters = [get_cluster_for_path(r.path) for r in r_fixed]
        fixed_post.append(ndcg_at_k(clusters, eq["relevant_clusters"], k=top_k))
        # Adaptive
        r_adapt = store.search(
            query_embedding=query_embeddings[q], query_text=q,
            top_k=top_k, scoring=fixed_scoring,
        )
        clusters = [get_cluster_for_path(r.path) for r in r_adapt]
        adaptive_post.append(ndcg_at_k(clusters, eq["relevant_clusters"], k=top_k))

    avg_fixed_post = sum(fixed_post) / len(fixed_post)
    avg_adaptive_post = sum(adaptive_post) / len(adaptive_post)
    eff_beta_post = beta * gamma_post

    print(f"  After power-user injection:")
    print(f"    gamma = {gamma_post:.3f}, effective beta = {eff_beta_post:.3f}")
    print(f"    max/mean = {max_post}/{mean_post:.1f} (ratio = {max_post/mean_post:.1f}x)")
    print(f"    Baseline:  {avg_baseline:.4f}")
    print(f"    Fixed beta:   {avg_fixed_post:.4f} ({(avg_fixed_post-avg_baseline)/avg_baseline*100:+.1f}%)")
    print(f"    Adaptive:  {avg_adaptive_post:.4f} ({(avg_adaptive_post-avg_baseline)/avg_baseline*100:+.1f}%)")

    # Analyze crossover and degradation
    def find_crossover(values: list[float], baseline: float) -> int | None:
        consec = 0
        for i, v in enumerate(values):
            if v > baseline:
                consec += 1
                if consec >= 3:
                    return i - 1  # first of 3 consecutive
            else:
                consec = 0
        return None

    fixed_crossover = find_crossover([r.fixed_ndcg for r in rounds], avg_baseline)
    adaptive_crossover = find_crossover([r.adaptive_ndcg for r in rounds], avg_baseline)

    fixed_degrade = sum(1 for r in rounds if r.fixed_ndcg < avg_baseline)
    adaptive_degrade = sum(1 for r in rounds if r.adaptive_ndcg < avg_baseline)

    return ExperimentResult(
        rounds=rounds,
        total_rounds=n_rounds,
        corpus_docs=len(TOPIC_NAMES) * 10,
        corpus_chunks=store._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
        n_queries=len(EVAL_QUERIES),
        fixed_crossover=fixed_crossover,
        adaptive_crossover=adaptive_crossover,
        fixed_degradation_rounds=fixed_degrade,
        adaptive_degradation_rounds=adaptive_degrade,
    )


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive Cold Start Experiment")
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--output", type=str, default="experiments/results/adaptive_cold_start.json")
    args = parser.parse_args()

    dim = get_embedding_dim(MODEL)
    tmp_dir = tempfile.mkdtemp(prefix="vstash_adaptive_")
    db_path = str(Path(tmp_dir) / "adaptive.db")
    store = VstashStore(db_path, embedding_dim=dim)

    print(f"DB: {db_path}")
    print("Building corpus from 120 Wikipedia articles (12 clusters x 10 articles)...")
    print("Articles chunked via vstash pipeline (chunk_size=1024, overlap=128)...")
    t0 = time.time()
    n_chunks = build_corpus(store)
    t1 = time.time()
    print(f"Ingested: {n_chunks} chunks in {t1 - t0:.1f}s\n")

    sep = "=" * 90
    print(f"{sep}")
    print("  ADAPTIVE COLD START: fixed beta vs adaptive beta (gamma-gated)")
    print(f"{sep}\n")

    t0 = time.time()
    result = run_experiment(store, n_rounds=args.rounds)
    elapsed = time.time() - t0
    print(f"  Experiment completed in {elapsed:.1f}s\n")

    baseline = result.rounds[0].baseline_ndcg
    print(f"  Baseline NDCG@5 (RRF only): {baseline:.4f}")
    print(f"  Corpus: {result.corpus_docs} docs, {result.corpus_chunks} chunks")
    print(f"  Queries: {result.n_queries}\n")

    print(f"  {'Rnd':>4} {'Baseline':>9} {'Fixed':>8} {'d%':>7} {'Adaptive':>9} {'d%':>7} "
          f"{'gamma':>5} {'eff_b':>6} {'max/mean':>10} {'Accesses':>9}")
    print(f"  {'-' * 85}")

    for r in result.rounds:
        fixed_d = ((r.fixed_ndcg - baseline) / baseline * 100) if baseline > 0 else 0
        adapt_d = ((r.adaptive_ndcg - baseline) / baseline * 100) if baseline > 0 else 0
        ratio_str = f"{r.max_access}/{r.mean_access:.1f}" if r.mean_access > 0 else "--"
        print(
            f"  {r.round_num:>4} {baseline:>9.4f} {r.fixed_ndcg:>8.4f} {fixed_d:>+6.1f}% "
            f"{r.adaptive_ndcg:>9.4f} {adapt_d:>+6.1f}% {r.gamma:>5.2f} {r.effective_beta:>6.3f} "
            f"{ratio_str:>10} {r.total_accesses:>9}"
        )

    print(f"\n{sep}")
    print(f"  Fixed beta:    crossover={result.fixed_crossover or 'never'}, "
          f"degradation={result.fixed_degradation_rounds}/{result.total_rounds} rounds")
    print(f"  Adaptive gamma: crossover={result.adaptive_crossover or 'never'}, "
          f"degradation={result.adaptive_degradation_rounds}/{result.total_rounds} rounds")

    if result.adaptive_degradation_rounds < result.fixed_degradation_rounds:
        improvement = result.fixed_degradation_rounds - result.adaptive_degradation_rounds
        print(f"\n  * Adaptive scoring eliminates {improvement} rounds of degradation")
    print(f"{sep}\n")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(result), indent=2))
    print(f"  Results saved to: {output_path}")

    store.close()


if __name__ == "__main__":
    main()
