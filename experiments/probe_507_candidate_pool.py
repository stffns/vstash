"""Probe for PR #507: does the candidate pool cause known-item misses?

PR #507 claims that with ``top_k=5`` the candidate pool caps at 50, so a
distinctive one-chunk document falls outside the pool when thousands of
near-duplicate documents dominate the embedding space, and proposes
raising the pool floor to 100 (>2k chunks) / 200 (>5k chunks).

This probe reconstructs the reported corpus shape (2404 near-duplicate
"audit" documents + 1332 "default" documents + one target document in its
own collection) and measures where the target actually ends up.

The global vector rank is computed in numpy, so it is independent of the
candidate-pool code under review: it answers "how many chunks are strictly
closer to the query than the target", which is the only thing a larger pool
could ever fix.

Two dimensions are varied:

* **Noise relatedness** -- audit records unrelated to the query vs audit
  records that restate the target's own content (what a merken-style audit
  log actually holds).
* **Target strength** -- from a target that shares the query's vocabulary
  down to a paraphrase with no lexical overlap at all.

Run:
    python -m experiments.probe_507_candidate_pool
"""

from __future__ import annotations

import os
import tempfile
import time

import numpy as np

from vstash._store_open import open_store_for_config
from vstash.config import VstashConfig
from vstash.embed import embed_query, embed_texts

MODEL = "BAAI/bge-small-en-v1.5"
QUERY = "Preferencias de escritura de Jay sin emojis"

# Collection sizes as reported in the PR description.
N_AUDIT = 2404
N_DEFAULT = 1332

# Candidate pool sizes: what develop computes at top_k=5 today, and what
# PR #507 would force for the same query on a >5k-chunk corpus.
POOL_TODAY = 50
POOL_PR = 200

TARGETS: dict[str, str] = {
    # Shares the query's vocabulary almost verbatim.
    "T1_lexical": (
        "Preferencias de escritura de Jay: sin emojis, tono directo, sin florituras. "
        "Prefiere respuestas concisas y en espanol, con el codigo siempre en ingles."
    ),
    # Same fact, different words.
    "T2_paraphrase": (
        "Jay quiere que la comunicacion escrita sea sobria: nada de iconos decorativos, "
        "frases cortas y directas, sin adornos ni entusiasmo impostado."
    ),
    # Semantic only: no query term survives stemming.
    "T3_semantic": (
        "Nada de caritas ni simbolos graficos en los mensajes. El tono debe ser seco, "
        "profesional y sin rodeos. Menos es mas en cada respuesta."
    ),
    # Oblique phrasing of the same rule.
    "T4_oblique": (
        "Regla de estilo personal: la sobriedad manda. Evitar decoracion visual "
        "en el texto y cualquier floritura iconografica. Ir al grano siempre."
    ),
}


def audit_restating_target(i: int) -> str:
    """Near-duplicate audit record that restates the remembered fact."""
    return (
        f"[audit {4000 + i}] remember: Preferencias de escritura de Jay -- "
        f"sin emojis, tono directo. decision=keep confidence=0.{70 + i % 30} "
        f"source=sesion-{i % 97}"
    )


def audit_unrelated(i: int) -> str:
    """Audit record with no bearing on the query."""
    return (
        f"[audit {4000 + i}] decision=keep confidence=0.{70 + i % 30} "
        f"source=sesion-{i % 97} politica=consolidacion aplicada. "
        f"Evento registrado en el ciclo {i % 13} del pipeline de memoria."
    )


def default_doc(i: int) -> str:
    topics = (
        "Notas del proyecto vstash sobre busqueda hibrida y fusion RRF",
        "Reunion semanal: estado del pipeline de ingesta y chunking",
        "Configuracion de despliegue y variables de entorno del servidor",
        "Resumen de la investigacion sobre embeddings multilingues",
        "Registro de incidencias del watcher de ficheros y su debounce",
    )
    return f"{topics[i % len(topics)]}. Entrada numero {i} con detalle adicional."


def _global_vector_rank(query_emb, corpus: np.ndarray, target_row: int) -> tuple[float, int]:
    """Exact cosine distance and 1-based global rank of ``target_row``.

    Pool-independent by construction: it ranks the whole corpus.
    """
    q = np.asarray(query_emb, dtype=np.float32)
    q = q / np.linalg.norm(q)
    mat = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
    dists = 1.0 - mat @ q
    d = float(dists[target_row])
    return d, int((dists < d).sum()) + 1


def main() -> None:
    t0 = time.time()
    print(f"model={MODEL}")
    print(f"query={QUERY!r}\n")

    query_emb = embed_query(QUERY, MODEL)
    target_texts = list(TARGETS.values())
    target_embs = embed_texts(target_texts, MODEL)

    for noise_label, noise_fn in (
        ("audit noise UNRELATED to the query", audit_unrelated),
        ("audit noise RESTATING the target", audit_restating_target),
    ):
        noise_texts = [noise_fn(i) for i in range(N_AUDIT)]
        noise_texts += [default_doc(i) for i in range(N_DEFAULT)]
        noise_embs = embed_texts(noise_texts, MODEL)
        corpus = np.asarray(noise_embs + target_embs, dtype=np.float32)

        print(f"=== {noise_label} ({len(corpus)} chunks) ===")
        print(
            f"{'variant':16s} {'dist':>7s} {'vec_rank':>9s} "
            f"{'pool' + str(POOL_TODAY):>7s} {'pool' + str(POOL_PR):>8s} "
            f"{'top5':>6s}  dropped_at"
        )

        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "probe507.db")
            store = open_store_for_config(VstashConfig(), db_path=db_path)
            try:
                docs: list[dict] = []
                for i in range(N_AUDIT):
                    docs.append(
                        {
                            "path": f"audit_{i}.md",
                            "title": f"Audit {i}",
                            "chunks": [noise_texts[i]],
                            "embeddings": [noise_embs[i]],
                            "source_type": "text",
                            "collection": "merken_audit",
                        }
                    )
                for j in range(N_DEFAULT):
                    i = N_AUDIT + j
                    docs.append(
                        {
                            "path": f"note_{j}.md",
                            "title": f"Nota {j}",
                            "chunks": [noise_texts[i]],
                            "embeddings": [noise_embs[i]],
                            "source_type": "text",
                            "collection": "default",
                        }
                    )
                for n, name in enumerate(TARGETS):
                    docs.append(
                        {
                            "path": f"{name}.md",
                            "title": name,
                            "chunks": [target_texts[n]],
                            "embeddings": [target_embs[n]],
                            "source_type": "text",
                            "collection": "preferences",
                        }
                    )

                with store.batch_mode(defer_fts=True):
                    store.add_documents_batch(docs)

                results = store.search(query_emb, QUERY, top_k=5)
                top5 = [r.path for r in results]

                for n, name in enumerate(TARGETS):
                    dist, rank = _global_vector_rank(query_emb, corpus, len(noise_texts) + n)
                    hit = next((i for i, p in enumerate(top5) if p == f"{name}.md"), None)
                    miss = store.miss_analysis(
                        query_emb, QUERY, expected_path=f"{name}.md", top_k=5
                    )
                    print(
                        f"{name:16s} {dist:7.4f} {rank:9d} "
                        f"{rank <= POOL_TODAY!s:>7s} {rank <= POOL_PR!s:>8s} "
                        f"{('#' + str(hit + 1)) if hit is not None else 'MISS':>6s}  "
                        f"{miss.dropped_at or '-'}"
                    )
                print(f"  actual top-5: {top5}\n")
            finally:
                store.close()

    print(f"total {time.time() - t0:.1f}s")
    print(
        "\nReading: a larger candidate pool can only rescue a target whose global\n"
        "vector rank is inside the new pool. Any target ranked in the thousands is\n"
        "beaten by that many chunks outright — no pool size recovers it, and the\n"
        "miss is a precision/corpus problem, not a pool-size problem."
    )


if __name__ == "__main__":
    main()
