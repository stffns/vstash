"""Adaptive Cold Start Experiment — γ (maturity-gated scoring) vs fixed β.

Compares three scoring strategies on a 120-document synthetic corpus:
  1. baseline:  RRF only (no scoring)
  2. fixed:     Fixed β=0.5 from round 1 (current behavior pre-v0.7)
  3. adaptive:  β scaled by γ = outlier ratio (new behavior)

The corpus is generated locally from 12 topic clusters × 10 docs each,
with realistic chunk counts (3-8 per doc).  No network access required.

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
from dataclasses import asdict, dataclass
from pathlib import Path

from vstash.config import ScoringConfig
from vstash.embed import embed_query, get_embedding_dim
from vstash.store import VstashStore

MODEL = "BAAI/bge-small-en-v1.5"

# ------------------------------------------------------------------ #
# Synthetic corpus: 12 clusters × 10 documents × 3-8 chunks          #
# ------------------------------------------------------------------ #

TOPICS = [
    {
        "name": "transformer_architecture",
        "docs": [
            "Self-attention mechanism computes pairwise token similarities using query, key, and value projections.",
            "Multi-head attention allows the model to attend to information from different representation subspaces.",
            "Positional encoding injects sequence order information since transformers have no inherent position awareness.",
            "Layer normalization and residual connections stabilize training in deep transformer networks.",
            "The feed-forward network in each transformer layer applies two linear transformations with a ReLU activation.",
            "Masked self-attention in the decoder prevents attending to future tokens during autoregressive generation.",
            "Cross-attention in encoder-decoder models allows the decoder to attend to all encoder hidden states.",
            "Relative position embeddings like RoPE encode distance between tokens rather than absolute positions.",
            "Sparse attention patterns like Longformer reduce quadratic complexity for processing long documents.",
            "Flash attention optimizes memory access patterns to speed up attention computation on GPUs.",
        ],
    },
    {
        "name": "reinforcement_learning",
        "docs": [
            "Q-learning estimates the optimal action-value function through iterative Bellman equation updates.",
            "Policy gradient methods directly optimize the policy by estimating the gradient of expected reward.",
            "Actor-critic architectures combine value function estimation with policy optimization for stable learning.",
            "Proximal Policy Optimization clips the policy ratio to prevent destructively large updates.",
            "Reward shaping provides intermediate signals to guide agents through sparse reward environments.",
            "Multi-agent reinforcement learning studies how multiple agents learn to cooperate or compete.",
            "Model-based reinforcement learning builds environment models to plan ahead without trial and error.",
            "Hierarchical reinforcement learning decomposes tasks into subtasks with options or goal-conditioned policies.",
            "Inverse reinforcement learning infers reward functions from expert demonstrations of desired behavior.",
            "Offline reinforcement learning trains policies from fixed datasets without additional environment interaction.",
        ],
    },
    {
        "name": "natural_language_processing",
        "docs": [
            "Named entity recognition identifies and classifies proper nouns into categories like person and location.",
            "Dependency parsing reveals grammatical relationships between words as directed tree structures.",
            "Sentiment analysis classifies text into positive, negative, or neutral emotional tone categories.",
            "Machine translation converts text from one language to another using encoder-decoder neural architectures.",
            "Text summarization condenses long documents into shorter versions preserving key information.",
            "Question answering systems extract or generate answers from context passages given a query.",
            "Coreference resolution links pronouns and other referring expressions to their antecedent entities.",
            "Word sense disambiguation determines which meaning of a polysemous word is intended in context.",
            "Tokenization splits text into subword units that balance vocabulary size and sequence length.",
            "Semantic role labeling identifies who did what to whom in a sentence's predicate-argument structure.",
        ],
    },
    {
        "name": "computer_vision",
        "docs": [
            "Convolutional neural networks apply learned filters to extract hierarchical visual features from images.",
            "Object detection localizes and classifies multiple objects within an image using bounding box regression.",
            "Image segmentation assigns a class label to every pixel for fine-grained scene understanding.",
            "Data augmentation artificially expands training sets through random crops, flips, and color jittering.",
            "Transfer learning fine-tunes pretrained visual models on downstream tasks with limited labeled data.",
            "Generative adversarial networks synthesize realistic images through minimax training of generator and discriminator.",
            "Vision transformers partition images into patches and process them as token sequences using self-attention.",
            "Feature pyramid networks aggregate multi-scale features for detecting objects at different sizes.",
            "Optical flow estimation computes pixel-level motion between consecutive video frames.",
            "Depth estimation predicts per-pixel distances from monocular images using geometric constraints.",
        ],
    },
    {
        "name": "database_systems",
        "docs": [
            "B-tree indexes organize data in balanced tree structures for efficient range queries and point lookups.",
            "Write-ahead logging ensures durability by persisting changes to a log before modifying data pages.",
            "Query optimization transforms SQL queries into efficient execution plans using cost-based analysis.",
            "MVCC provides snapshot isolation by maintaining multiple versions of each row for concurrent readers.",
            "Columnar storage formats store data by column rather than row for analytical query performance.",
            "Distributed consensus protocols like Raft ensure data consistency across replicated database nodes.",
            "Hash joins build an in-memory hash table on the smaller relation for efficient equi-join processing.",
            "Connection pooling reuses database connections to reduce overhead from repeated handshake operations.",
            "Materialized views precompute and cache expensive query results for fast subsequent access.",
            "Sharding partitions data across multiple nodes using a shard key for horizontal scalability.",
        ],
    },
    {
        "name": "distributed_systems",
        "docs": [
            "CAP theorem states that distributed systems cannot simultaneously guarantee consistency, availability, and partition tolerance.",
            "Eventual consistency allows replicas to temporarily diverge but guarantees convergence over time.",
            "Leader election algorithms select a single coordinator node to manage writes in a cluster.",
            "Consistent hashing distributes keys across nodes while minimizing key redistribution when nodes change.",
            "Two-phase commit coordinates atomic transactions across multiple distributed participants.",
            "Vector clocks track causal ordering of events across processes without synchronized physical clocks.",
            "Gossip protocols disseminate information through random peer-to-peer message exchanges.",
            "Circuit breakers prevent cascading failures by stopping requests to unhealthy downstream services.",
            "Service mesh infrastructure provides observability, traffic management, and security between microservices.",
            "CRDTs enable conflict-free replication by encoding merge semantics into the data structure itself.",
        ],
    },
    {
        "name": "cryptography",
        "docs": [
            "AES symmetric encryption processes 128-bit blocks through multiple rounds of substitution and permutation.",
            "RSA public key cryptography relies on the computational difficulty of factoring large semiprime numbers.",
            "Hash functions like SHA-256 produce fixed-size digests that are computationally infeasible to reverse.",
            "Digital signatures verify message authenticity and integrity using asymmetric key pairs.",
            "Key exchange protocols like Diffie-Hellman establish shared secrets over insecure communication channels.",
            "Zero-knowledge proofs demonstrate possession of information without revealing the information itself.",
            "Homomorphic encryption allows computation on ciphertext that produces encrypted results matching plaintext operations.",
            "Certificate authorities issue X.509 certificates that bind public keys to verified domain identities.",
            "Post-quantum cryptography develops algorithms resistant to attacks from quantum computers.",
            "Secure multi-party computation enables joint function evaluation without revealing individual private inputs.",
        ],
    },
    {
        "name": "operating_systems",
        "docs": [
            "Process scheduling algorithms allocate CPU time among competing processes to maximize throughput.",
            "Virtual memory maps process address spaces to physical memory pages with transparent swapping.",
            "File systems organize persistent data into hierarchical directory structures with metadata inodes.",
            "Interrupt handlers respond to hardware signals by suspending current execution and running service routines.",
            "Mutexes and semaphores synchronize concurrent threads to prevent race conditions on shared resources.",
            "Page replacement policies like LRU decide which memory pages to evict when physical memory is full.",
            "System calls provide the interface between user-space applications and kernel-level operations.",
            "Device drivers translate generic I/O requests into hardware-specific commands for peripheral devices.",
            "Container namespaces isolate process trees, network stacks, and file systems for lightweight virtualization.",
            "Memory-mapped I/O exposes device registers as memory addresses for efficient hardware communication.",
        ],
    },
    {
        "name": "graph_algorithms",
        "docs": [
            "Breadth-first search explores vertices layer by layer to find shortest paths in unweighted graphs.",
            "Dijkstra's algorithm computes single-source shortest paths in graphs with non-negative edge weights.",
            "Minimum spanning tree algorithms like Kruskal's connect all vertices with minimum total edge weight.",
            "Topological sorting orders directed acyclic graph vertices so that edges point forward in the sequence.",
            "Strongly connected components partition a directed graph into maximal subgraphs with mutual reachability.",
            "Maximum flow algorithms like Ford-Fulkerson find the greatest throughput from source to sink nodes.",
            "Graph coloring assigns colors to vertices such that no two adjacent vertices share the same color.",
            "PageRank computes node importance scores based on the link structure of directed graphs.",
            "Community detection algorithms identify densely connected groups within large social or biological networks.",
            "A-star search combines heuristic distance estimation with path cost for efficient informed graph traversal.",
        ],
    },
    {
        "name": "machine_learning_fundamentals",
        "docs": [
            "Gradient descent iteratively adjusts model parameters in the direction that minimizes the loss function.",
            "Regularization techniques like L2 penalty and dropout prevent overfitting by constraining model complexity.",
            "Cross-validation partitions data into folds to estimate generalization performance without a separate test set.",
            "Feature engineering transforms raw data into informative representations that improve model predictive power.",
            "Ensemble methods like random forests and boosting combine multiple weak learners for stronger predictions.",
            "Bias-variance tradeoff balances underfitting from overly simple models against overfitting from complex ones.",
            "Batch normalization standardizes layer inputs to accelerate training and reduce sensitivity to initialization.",
            "Learning rate scheduling adjusts the step size during training to balance convergence speed and stability.",
            "Loss functions like cross-entropy and mean squared error measure the discrepancy between predictions and targets.",
            "Hyperparameter tuning searches for optimal configuration settings that maximize model validation performance.",
        ],
    },
    {
        "name": "software_engineering",
        "docs": [
            "Test-driven development writes failing tests before implementation to ensure correct behavior specification.",
            "Continuous integration automatically builds and tests code changes to catch regressions early.",
            "Code review identifies bugs, enforces standards, and transfers knowledge through peer examination of changes.",
            "Dependency injection decouples components by providing dependencies externally rather than creating them internally.",
            "Design patterns like observer and strategy provide reusable solutions to common software design problems.",
            "Refactoring improves internal code structure without changing external behavior to reduce technical debt.",
            "Version control systems track file changes over time enabling collaboration and historical recovery.",
            "API versioning strategies maintain backward compatibility while evolving service interfaces over time.",
            "Load testing simulates concurrent users to identify performance bottlenecks before production deployment.",
            "Monitoring and alerting systems detect anomalies in application metrics to enable rapid incident response.",
        ],
    },
    {
        "name": "information_retrieval",
        "docs": [
            "TF-IDF weights terms by frequency within a document normalized by inverse document frequency across the corpus.",
            "BM25 ranking function extends TF-IDF with document length normalization and saturation parameters.",
            "Vector space models represent documents and queries as high-dimensional vectors for cosine similarity matching.",
            "Inverted indexes map terms to posting lists of document identifiers for efficient keyword retrieval.",
            "Learning to rank trains models to combine multiple relevance signals into a single ranking function.",
            "Pseudo-relevance feedback expands queries with terms from top-ranked documents to improve recall.",
            "Reciprocal rank fusion combines multiple ranked lists by summing inverse rank contributions per candidate.",
            "Semantic search uses dense vector embeddings to find conceptually similar documents beyond keyword matching.",
            "Evaluation metrics like NDCG and MAP measure ranking quality with graded and binary relevance judgments.",
            "Hybrid retrieval combines sparse keyword matching with dense semantic search for complementary coverage.",
        ],
    },
]

# 10 eval queries with expected top results (by document text prefix)
EVAL_QUERIES = [
    {
        "query": "how does self-attention work in transformers",
        "expected_cluster": "transformer_architecture",
    },
    {
        "query": "policy gradient reinforcement learning algorithms",
        "expected_cluster": "reinforcement_learning",
    },
    {
        "query": "named entity recognition and text classification",
        "expected_cluster": "natural_language_processing",
    },
    {
        "query": "object detection bounding box neural network",
        "expected_cluster": "computer_vision",
    },
    {
        "query": "database indexing and query optimization",
        "expected_cluster": "database_systems",
    },
    {
        "query": "consensus protocols for distributed replication",
        "expected_cluster": "distributed_systems",
    },
    {
        "query": "public key encryption and digital signatures",
        "expected_cluster": "cryptography",
    },
    {
        "query": "process scheduling and virtual memory management",
        "expected_cluster": "operating_systems",
    },
    {
        "query": "shortest path algorithms in weighted graphs",
        "expected_cluster": "graph_algorithms",
    },
    {
        "query": "BM25 ranking and inverted index retrieval",
        "expected_cluster": "information_retrieval",
    },
]

# Zipf-weighted query distribution: some topics queried much more.
QUERY_WEIGHTS = [8, 6, 5, 4, 3, 3, 2, 2, 1, 1]


# ------------------------------------------------------------------ #
# Corpus generation                                                    #
# ------------------------------------------------------------------ #


def build_corpus(store: VstashStore) -> int:
    """Generate and ingest 120 synthetic documents (12 clusters × 10)."""
    total_chunks = 0
    rng = random.Random(42)

    for topic in TOPICS:
        for i, doc_text in enumerate(topic["docs"]):
            # Split each doc into 3-8 chunks to simulate realistic chunking
            sentences = doc_text.replace(". ", ".\n").split("\n")
            n_extra = rng.randint(2, 6)
            chunks = list(sentences)
            # Generate additional chunks by paraphrasing / expanding
            for j in range(n_extra):
                base = sentences[j % len(sentences)]
                chunks.append(f"Additionally, {base.lower()} This is fundamental to {topic['name'].replace('_', ' ')}.")

            embeddings = [embed_query(c, MODEL) for c in chunks]
            path = f"/corpus/{topic['name']}/doc_{i:02d}.md"
            title = f"{topic['name']}_{i:02d}"

            store.add_document(
                path=path,
                title=title,
                chunks=chunks,
                embeddings=embeddings,
                source_type="markdown",
            )
            total_chunks += len(chunks)

    return total_chunks


# ------------------------------------------------------------------ #
# NDCG computation                                                     #
# ------------------------------------------------------------------ #


def ndcg_at_k(result_clusters: list[str], expected_cluster: str, k: int = 5) -> float:
    """Compute NDCG@k using binary relevance (1 if cluster matches, 0 otherwise)."""
    dcg = 0.0
    for i, cluster in enumerate(result_clusters[:k]):
        rel = 1.0 if cluster == expected_cluster else 0.0
        dcg += rel / math.log2(i + 2)

    # Ideal: all top-k are relevant
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, 10)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def get_cluster_for_path(path: str) -> str:
    """Extract cluster name from document path."""
    # /corpus/transformer_architecture/doc_00.md → transformer_architecture
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
    # Baseline uses same over_fetch pool but γ=0 (pure RRF + dedup)
    baseline_scoring = ScoringConfig(
        enabled=True, alpha=alpha, beta=beta,
        decay_lambda=decay_lambda, over_fetch=50, track_access=False,
    )

    # Pre-compute query embeddings
    query_embeddings = {}
    for eq in EVAL_QUERIES:
        query_embeddings[eq["query"]] = embed_query(eq["query"], MODEL)

    # Baseline: RRF only with same candidate pool (γ forced to 0)
    baseline_ndcgs = []
    for eq in EVAL_QUERIES:
        q = eq["query"]
        results = store.search(
            query_embedding=query_embeddings[q], query_text=q,
            top_k=top_k, scoring=baseline_scoring,
            _gamma_override=0.0,  # force pure RRF
        )
        clusters = [get_cluster_for_path(r.path) for r in results]
        baseline_ndcgs.append(ndcg_at_k(clusters, eq["expected_cluster"], k=top_k))
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

        # Read γ from the store (adaptive)
        gamma = store.scoring_maturity()
        effective_beta = beta * gamma

        # Access stats
        row = store._conn.execute(
            "SELECT AVG(access_count) as mean, MAX(access_count) as mx "
            "FROM chunks WHERE access_count > 0"
        ).fetchone()
        mean_acc = float(row["mean"]) if row and row["mean"] else 0.0
        max_acc = int(row["mx"]) if row and row["mx"] else 0

        # Phase B: evaluate with FIXED scoring (γ=1.0 always — old behavior)
        fixed_ndcgs = []
        for eq in EVAL_QUERIES:
            q = eq["query"]
            results = store.search(
                query_embedding=query_embeddings[q], query_text=q,
                top_k=top_k, scoring=fixed_scoring,
                _gamma_override=1.0,  # force full β — simulates pre-v0.7
            )
            clusters = [get_cluster_for_path(r.path) for r in results]
            fixed_ndcgs.append(ndcg_at_k(clusters, eq["expected_cluster"], k=top_k))
        avg_fixed = sum(fixed_ndcgs) / len(fixed_ndcgs)

        # Phase C: evaluate with ADAPTIVE scoring (γ computed from data)
        adaptive_ndcgs = []
        for eq in EVAL_QUERIES:
            q = eq["query"]
            results = store.search(
                query_embedding=query_embeddings[q], query_text=q,
                top_k=top_k, scoring=fixed_scoring,
                # _gamma_override not set → uses real scoring_maturity()
            )
            clusters = [get_cluster_for_path(r.path) for r in results]
            adaptive_ndcgs.append(ndcg_at_k(clusters, eq["expected_cluster"], k=top_k))
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
    print("\n  --- Injecting power-user pattern (50× boost on transformers) ---\n")
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
        fixed_post.append(ndcg_at_k(clusters, eq["expected_cluster"], k=top_k))
        # Adaptive
        r_adapt = store.search(
            query_embedding=query_embeddings[q], query_text=q,
            top_k=top_k, scoring=fixed_scoring,
        )
        clusters = [get_cluster_for_path(r.path) for r in r_adapt]
        adaptive_post.append(ndcg_at_k(clusters, eq["expected_cluster"], k=top_k))

    avg_fixed_post = sum(fixed_post) / len(fixed_post)
    avg_adaptive_post = sum(adaptive_post) / len(adaptive_post)
    eff_beta_post = beta * gamma_post

    print(f"  After power-user injection:")
    print(f"    γ = {gamma_post:.3f}, effective β = {eff_beta_post:.3f}")
    print(f"    max/mean = {max_post}/{mean_post:.1f} (ratio = {max_post/mean_post:.1f}×)")
    print(f"    Baseline:  {avg_baseline:.4f}")
    print(f"    Fixed β:   {avg_fixed_post:.4f} ({(avg_fixed_post-avg_baseline)/avg_baseline*100:+.1f}%)")
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
        corpus_docs=len(TOPICS) * 10,
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
    print("Generating 120-document corpus (12 clusters × 10 docs)...")
    t0 = time.time()
    n_chunks = build_corpus(store)
    t1 = time.time()
    print(f"Ingested: 120 docs, {n_chunks} chunks in {t1 - t0:.1f}s\n")

    sep = "=" * 90
    print(f"{sep}")
    print("  ADAPTIVE COLD START: fixed β vs adaptive β (γ-gated)")
    print(f"{sep}\n")

    t0 = time.time()
    result = run_experiment(store, n_rounds=args.rounds)
    elapsed = time.time() - t0
    print(f"  Experiment completed in {elapsed:.1f}s\n")

    baseline = result.rounds[0].baseline_ndcg
    print(f"  Baseline NDCG@5 (RRF only): {baseline:.4f}")
    print(f"  Corpus: {result.corpus_docs} docs, {result.corpus_chunks} chunks")
    print(f"  Queries: {result.n_queries}\n")

    print(f"  {'Rnd':>4} {'Baseline':>9} {'Fixed':>8} {'Δ%':>7} {'Adaptive':>9} {'Δ%':>7} "
          f"{'γ':>5} {'eff_β':>6} {'max/mean':>10} {'Accesses':>9}")
    print(f"  {'─' * 85}")

    for r in result.rounds:
        fixed_d = ((r.fixed_ndcg - baseline) / baseline * 100) if baseline > 0 else 0
        adapt_d = ((r.adaptive_ndcg - baseline) / baseline * 100) if baseline > 0 else 0
        ratio_str = f"{r.max_access}/{r.mean_access:.1f}" if r.mean_access > 0 else "—"
        print(
            f"  {r.round_num:>4} {baseline:>9.4f} {r.fixed_ndcg:>8.4f} {fixed_d:>+6.1f}% "
            f"{r.adaptive_ndcg:>9.4f} {adapt_d:>+6.1f}% {r.gamma:>5.2f} {r.effective_beta:>6.3f} "
            f"{ratio_str:>10} {r.total_accesses:>9}"
        )

    print(f"\n{sep}")
    print(f"  Fixed β:    crossover={result.fixed_crossover or 'never'}, "
          f"degradation={result.fixed_degradation_rounds}/{result.total_rounds} rounds")
    print(f"  Adaptive γ: crossover={result.adaptive_crossover or 'never'}, "
          f"degradation={result.adaptive_degradation_rounds}/{result.total_rounds} rounds")

    if result.adaptive_degradation_rounds < result.fixed_degradation_rounds:
        improvement = result.fixed_degradation_rounds - result.adaptive_degradation_rounds
        print(f"\n  ✓ Adaptive scoring eliminates {improvement} rounds of degradation")
    print(f"{sep}\n")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(result), indent=2))
    print(f"  Results saved to: {output_path}")

    store.close()


if __name__ == "__main__":
    main()
