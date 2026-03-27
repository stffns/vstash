"""Frequency + Decay Scoring Experiment

Ingests 25 arxiv papers into a temporary vstash DB, then compares:
  Phase 1: Baseline (RRF puro) vs Scoring enabled (sin historial diferencial)
  Phase 2: Simula uso diferencial y mide el efecto del frequency+decay
  Phase 3: Varía λ y muestra cómo afecta el ranking

Usage:
    python -m experiments.scoring_experiment
"""

from __future__ import annotations

import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from vstash.config import ScoringConfig, VstashConfig
from vstash.embed import embed_query, get_embedding_dim
from vstash.store import VstashStore

# ------------------------------------------------------------------ #
# Corpus: 25 papers                                                    #
# ------------------------------------------------------------------ #

PAPERS: list[dict[str, str]] = [
    # Foundational (2023)
    {"title": "MemGPT: Towards LLMs as Operating Systems", "url": "https://arxiv.org/pdf/2310.08560", "year": "2023"},
    {"title": "Generative Agents: Interactive Simulacra of Human Behavior", "url": "https://arxiv.org/pdf/2304.03442", "year": "2023"},
    {"title": "RET-LLM: Towards a General Read-Write Memory for LLMs", "url": "https://arxiv.org/pdf/2305.14322", "year": "2023"},
    {"title": "SCM: Self-Controlled Memory Framework", "url": "https://arxiv.org/pdf/2304.13343", "year": "2023"},
    # Benchmarks & Evaluation (2024–2025)
    {"title": "LoCoMo: Evaluating Very Long-Term Conversational Memory", "url": "https://arxiv.org/pdf/2402.17753", "year": "2024"},
    {"title": "SORT: Sequence Order Recall Tasks for Episodic Memory", "url": "https://arxiv.org/pdf/2410.08133", "year": "2024"},
    {"title": "Episodic Memories Generation and Evaluation Benchmark", "url": "https://arxiv.org/pdf/2501.13121", "year": "2025"},
    # Memory Architectures (2024–2025)
    {"title": "MemoryBank: Enhancing LLMs with Long-Term Memory", "url": "https://arxiv.org/pdf/2305.10250", "year": "2024"},
    {"title": "Dynamic Tree Memory Representation for LLMs", "url": "https://arxiv.org/pdf/2410.14052", "year": "2024"},
    {"title": "A-MEM: Agentic Memory for LLM Agents", "url": "https://arxiv.org/pdf/2502.12110", "year": "2025"},
    {"title": "Mem0: Production-Ready AI Agents with Scalable Long-Term Memory", "url": "https://arxiv.org/pdf/2504.19413", "year": "2025"},
    {"title": "Memoria: Scalable Agentic Memory for Personalized Conversational AI", "url": "https://arxiv.org/pdf/2512.12686", "year": "2025"},
    {"title": "Memori: Persistent Memory Layer for Context-Aware LLM Agents", "url": "https://arxiv.org/pdf/2603.19935", "year": "2025"},
    {"title": "Nemori: Self-Organizing Agent Memory Inspired by Cognitive Science", "url": "https://arxiv.org/pdf/2508.03341", "year": "2025"},
    {"title": "SeCom: Memory Construction for Personalized Conversational Agents", "url": "https://arxiv.org/pdf/2502.05589", "year": "2025"},
    {"title": "R³Mem: Memory Retention and Retrieval via Reversible Compression", "url": "https://arxiv.org/pdf/2502.15957", "year": "2025"},
    # Temporal & Decay (2025–2026)
    {"title": "Zep: Temporal Knowledge Graph Architecture for Agent Memory", "url": "https://arxiv.org/pdf/2501.13956", "year": "2025"},
    {"title": "MaRS: Forgetful but Faithful — Cognitive Memory Architecture", "url": "https://arxiv.org/pdf/2512.12856", "year": "2025"},
    {"title": "PAM: Predictive Associative Memory via Temporal Co-occurrence", "url": "https://arxiv.org/pdf/2602.11322", "year": "2026"},
    # Advanced Systems (2025–2026)
    {"title": "E-mem: Multi-agent Episodic Context Reconstruction", "url": "https://arxiv.org/pdf/2601.21714", "year": "2026"},
    {"title": "Beyond Fact Retrieval: Episodic Memory for RAG with GSW", "url": "https://arxiv.org/pdf/2511.07587", "year": "2026"},
    {"title": "MAGMA: Multi-Graph based Agentic Memory Architecture", "url": "https://arxiv.org/pdf/2601.03236", "year": "2026"},
    {"title": "EverMemOS: Self-Organizing Memory OS for Long-Horizon Reasoning", "url": "https://arxiv.org/pdf/2601.02163", "year": "2026"},
    {"title": "AI PERSONA: Towards Life-long Personalization of LLMs", "url": "https://arxiv.org/pdf/2412.13103", "year": "2024"},
    # WMR uses frontiersin — skip for now (not arxiv PDF)
]

# Queries designed to test different aspects of the scoring
QUERIES = [
    "long-term memory for conversational AI agents",
    "temporal decay and forgetting in memory systems",
    "memory retrieval benchmarks and evaluation",
    "episodic memory architecture for LLMs",
    "personalization through persistent memory",
]

MODEL = "BAAI/bge-small-en-v1.5"


def banner(text: str) -> None:
    sep = "=" * 70
    print(f"\n{sep}\n  {text}\n{sep}")


def print_ranking(results: list, label: str, top_n: int = 10) -> None:
    print(f"\n  [{label}] Top {min(top_n, len(results))} results:")
    for i, r in enumerate(results[:top_n]):
        print(f"    {i+1:>2}. [{r.score:.6f}] {r.title[:65]}")


def run_queries(
    store: VstashStore,
    queries: list[str],
    scoring: ScoringConfig | None,
    label: str,
    top_k: int = 10,
) -> dict[str, list]:
    """Run all queries and return {query: [results]}."""
    all_results = {}
    for q in queries:
        emb = embed_query(q, MODEL)
        results = store.search(
            query_embedding=emb,
            query_text=q,
            top_k=top_k,
            scoring=scoring,
        )
        all_results[q] = results
        print_ranking(results, label, top_n=top_k)
    return all_results


def compare_rankings(
    baseline: dict[str, list],
    scored: dict[str, list],
) -> None:
    """Compare two sets of rankings and show differences."""
    print("\n  Ranking changes (baseline → scored):")
    for q in baseline:
        b_titles = [r.title for r in baseline[q]]
        s_titles = [r.title for r in scored[q]]

        changes = []
        for i, title in enumerate(s_titles):
            if title in b_titles:
                old_pos = b_titles.index(title)
                if old_pos != i:
                    direction = "↑" if i < old_pos else "↓"
                    changes.append(f"    {direction} {title[:50]}: #{old_pos+1} → #{i+1}")
            else:
                changes.append(f"    + {title[:50]}: NEW in top-k")

        if changes:
            print(f"\n  Query: \"{q[:60]}\"")
            for c in changes:
                print(c)
        else:
            print(f"\n  Query: \"{q[:60]}\" — no changes")


# ------------------------------------------------------------------ #
# Main experiment                                                      #
# ------------------------------------------------------------------ #

def main() -> None:
    tmp_dir = tempfile.mkdtemp(prefix="vstash_experiment_")
    db_path = str(Path(tmp_dir) / "experiment.db")
    dim = get_embedding_dim(MODEL)

    store = VstashStore(db_path, embedding_dim=dim)

    # ============================================================== #
    # Phase 0: Ingest papers                                          #
    # ============================================================== #
    banner("Phase 0: Ingesting 25 papers from arXiv")

    from vstash.ingest import ingest

    cfg = VstashConfig()
    ingested = 0
    failed = []

    for paper in PAPERS:
        print(f"  Ingesting: {paper['title'][:60]}...", end=" ", flush=True)
        t0 = time.time()
        try:
            r = ingest(
                source=paper["url"],
                cfg=cfg,
                store=store,
                collection="experiment",
            )
            elapsed = time.time() - t0
            if r.status == "ok":
                print(f"✓ {r.chunks} chunks, {elapsed:.1f}s")
                ingested += 1

                # Set created_at based on publication year to simulate temporal spread
                year = int(paper["year"])
                simulated_date = datetime(year, 6, 15, tzinfo=timezone.utc).isoformat()
                store._conn.execute(
                    "UPDATE chunks SET created_at = ?, last_accessed_at = ? "
                    "WHERE doc_id = (SELECT id FROM documents WHERE path = ?)",
                    [simulated_date, simulated_date, paper["url"]],
                )
                store._conn.commit()
            else:
                print(f"✗ {r.status}: {r.error}")
                failed.append(paper["title"])
        except Exception as e:
            elapsed = time.time() - t0
            print(f"✗ ERROR ({elapsed:.1f}s): {e}")
            failed.append(paper["title"])

    print(f"\n  Ingested: {ingested}/{len(PAPERS)}")
    if failed:
        print(f"  Failed: {failed}")

    stats_row = store._conn.execute("SELECT COUNT(*) as n FROM chunks").fetchone()
    print(f"  Total chunks in DB: {stats_row['n']}")

    # ============================================================== #
    # Phase 1: Baseline vs Scoring (no differential access)           #
    # ============================================================== #
    banner("Phase 1: Baseline (RRF) vs Scoring (frequency+decay)")
    print("  All papers have access_count=1 — scoring should only reflect temporal decay.")

    cfg_scoring = VstashConfig.model_validate({
        "scoring": {
            "enabled": True,
            "alpha": 0.7,
            "beta": 0.3,
            "decay_lambda": 0.07,
            "over_fetch": 50,
            "track_access": False,  # Don't mutate during baseline comparison
        }
    })

    print("\n--- Baseline (RRF puro) ---")
    baseline_results = run_queries(store, QUERIES, None, "BASELINE")

    print("\n--- Scoring (freq+decay, uniform access) ---")
    scored_results = run_queries(store, QUERIES, cfg_scoring.scoring, "SCORED")

    compare_rankings(baseline_results, scored_results)

    # ============================================================== #
    # Phase 2: Simulate differential access patterns                   #
    # ============================================================== #
    banner("Phase 2: Differential access simulation")

    # Find chunk IDs for key papers (vstash stores URL as path when ingesting from URL)
    def _find_chunks(url: str):
        return store._conn.execute(
            "SELECT c.id FROM chunks c JOIN documents d ON d.id = c.doc_id "
            "WHERE d.path = ?", [url]
        ).fetchall()

    locomo_chunks = _find_chunks("https://arxiv.org/pdf/2402.17753")
    memgpt_chunks = _find_chunks("https://arxiv.org/pdf/2310.08560")
    pam_chunks = _find_chunks("https://arxiv.org/pdf/2602.11322")

    # LoCoMo: simulate 10 recent accesses (heavily used benchmark)
    if locomo_chunks:
        ids = [r["id"] for r in locomo_chunks]
        print(f"  Boosting LoCoMo ({len(ids)} chunks) with 10 recent accesses...")
        for _ in range(10):
            store.track_access(ids)

    # MemGPT: simulate 1 access 90 days ago (foundational but stale)
    if memgpt_chunks:
        ids = [r["id"] for r in memgpt_chunks]
        old_date = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        print(f"  Setting MemGPT ({len(ids)} chunks) to 2 accesses, 90 days ago...")
        store._conn.execute(
            f"UPDATE chunks SET access_count = 2, last_accessed_at = ? "
            f"WHERE id IN ({','.join('?' * len(ids))})",
            [old_date, *ids],
        )
        store._conn.commit()

    # PAM: simulate 5 recent accesses (hot new paper)
    if pam_chunks:
        ids = [r["id"] for r in pam_chunks]
        print(f"  Boosting PAM ({len(ids)} chunks) with 5 recent accesses...")
        for _ in range(5):
            store.track_access(ids)

    print("\n--- Scoring after differential access ---")
    cfg_scoring_track = VstashConfig.model_validate({
        "scoring": {
            "enabled": True,
            "alpha": 0.7,
            "beta": 0.3,
            "decay_lambda": 0.07,
            "over_fetch": 50,
            "track_access": False,  # Don't mutate further during measurement
        }
    })
    post_access_results = run_queries(store, QUERIES, cfg_scoring_track.scoring, "POST-ACCESS")

    print("\n  Comparing: Baseline → Post-differential-access")
    compare_rankings(baseline_results, post_access_results)

    # ============================================================== #
    # Phase 3: Lambda sensitivity                                      #
    # ============================================================== #
    banner("Phase 3: Lambda sensitivity analysis")

    test_query = "long-term memory for conversational AI agents"
    test_emb = embed_query(test_query, MODEL)
    lambdas = [0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.5]

    print(f"  Query: \"{test_query}\"")
    print(f"  Testing λ values: {lambdas}\n")

    for lam in lambdas:
        cfg_lam = VstashConfig.model_validate({
            "scoring": {
                "enabled": True,
                "alpha": 0.7,
                "beta": 0.3,
                "decay_lambda": lam,
                "over_fetch": 50,
                "track_access": False,
            }
        })
        results = store.search(
            query_embedding=test_emb,
            query_text=test_query,
            top_k=5,
            scoring=cfg_lam.scoring,
        )
        print(f"  λ={lam:<5} → ", end="")
        top3 = ", ".join(f"{r.title[:30]}({r.score:.4f})" for r in results[:3])
        print(top3)

    # ============================================================== #
    # Summary                                                          #
    # ============================================================== #
    banner("Experiment complete")
    print(f"  DB: {db_path}")
    print(f"  Papers ingested: {ingested}/{len(PAPERS)}")
    print(f"  Queries tested: {len(QUERIES)}")
    print(f"  Lambda values tested: {len(lambdas)}")
    print()

    store.close()


if __name__ == "__main__":
    main()
