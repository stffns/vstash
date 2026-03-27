# Frequency + Decay Memory Scoring
## Implementation Plan — vstash

> Memoria humana sin sus desventajas biológicas.
> Miles de años de evolución, replicados en SQLite con tres columnas y una fórmula.

---

## La Fórmula

```
final_score = α · rrf_score + β · log(1 + access_count · e^(−λ · days_ago))
```

| Parámetro | Valor sugerido | Efecto |
|-----------|---------------|--------|
| α | 0.7 | Peso de similitud (RRF score) |
| β | 0.3 | Peso del historial de uso |
| λ | 0.05–0.1 | Velocidad de decay (0.05 = semanas, 0.1 = días) |

> **Nota:** la fórmula se aplica como **post-RRF re-ranker** — después de que
> el pipeline existente (vector similarity + FTS5 keyword + RRF fusion) produce
> su ranking. Esto mantiene la lógica de búsqueda intacta y agrega el scoring
> de memoria como capa final.

### Configuración en vstash.toml

```toml
[scoring]
enabled = true        # habilitado por defecto — NDCG +8.35% vs RRF puro
alpha = 0.5           # peso de similitud semántica (RRF normalizado)
beta = 0.5            # peso del historial de uso (frecuencia × decay)
decay_lambda = 0.05   # velocidad de decay (0.05 = semanas, 0.1 = días)
over_fetch = 50       # candidatos a recuperar antes del re-rank (ver Paso 4)
track_access = true   # si false, scoring usa datos existentes sin registrar nuevos accesos
```

---

## Paso 1 — Schema Migration

```sql
ALTER TABLE chunks ADD COLUMN access_count INTEGER DEFAULT 1;
ALTER TABLE chunks ADD COLUMN last_accessed_at TIMESTAMP;
ALTER TABLE chunks ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;

-- Backfill: copiar ingested_at del documento padre como created_at inicial
-- para chunks existentes (en vez de usar la fecha de migración)
UPDATE chunks SET created_at = (
    SELECT d.ingested_at FROM documents d WHERE d.id = chunks.doc_id
) WHERE created_at IS NULL OR created_at = CURRENT_TIMESTAMP;
```

### Cold Start: `access_count = 1`

Los chunks nuevos inician con `access_count = 1` (no 0). El acto de ingestar un
documento es el primer "acceso" — el usuario eligió deliberadamente agregar ese
contenido a su memoria.

**Por qué importa:** con `access_count = 0`, la parte de frecuencia se anula
completamente (`log(1 + 0) = 0`) y el chunk depende 100% de similitud semántica.
Chunks nuevos podrían ser enterrados permanentemente por chunks antiguos con
historial de acceso, sin oportunidad de competir.

Con `access_count = 1`, el chunk nuevo recibe un pequeño impulso de "novedad"
que decae rápidamente si no se vuelve a usar — una ventana justa para demostrar
relevancia antes de ser olvidado.

El backfill preserva la temporalidad real — un chunk ingestado hace 3 meses no
debería verse como "recién creado" después de la migración.

---

## Paso 2 — Scoring Function

### Problema de escala: RRF vs log

RRF produce valores fraccionales muy pequeños. Con `k=60` (estándar):
- Rank #1: `1/(60+1) ≈ 0.016`
- Rank #50: `1/(60+50) ≈ 0.009`

Mientras tanto, `log(1 + freq_score)` produce valores mucho mayores:
- 1 acceso reciente: `log(2) ≈ 0.69`
- 10 accesos recientes: `log(11) ≈ 2.40`

Sin normalización, con α=0.7 y β=0.3:
- Chunk A (rank #1, nuevo): `0.7 × 0.016 + 0.3 × log(2) ≈ 0.011 + 0.207 = 0.218`
- Chunk B (rank #50, 10 accesos): `0.7 × 0.009 + 0.3 × log(11) ≈ 0.006 + 0.719 = 0.725`

**El componente de memoria aplasta la similitud semántica.** El buscador
devolvería chunks populares en vez de relevantes.

### Solución: Min-Max Scaling del RRF

Normalizar los RRF scores del batch a `[0, 1]` antes de aplicar la fórmula:

```
normalized_rrf = (rrf - min_rrf) / (max_rrf - min_rrf)
```

Esto fuerza a que el mejor resultado semántico del batch tenga peso `1.0 × α = 0.7`,
compitiendo justamente contra el `log` de accesos.

### Implementación

```python
import math
from datetime import datetime

def rerank_with_decay(
    candidates: list[dict],
    *,
    alpha: float = 0.7,
    beta: float = 0.3,
    decay_lambda: float = 0.07,
) -> list[dict]:
    """Re-rank un batch de candidatos post-RRF con frequency + decay.

    Normaliza RRF scores a [0, 1] via min-max scaling para que los
    pesos α/β operen sobre escalas comparables.

    Args:
        candidates: lista de dicts con keys: rrf_score, access_count,
                    last_accessed_at, created_at
    """
    if not candidates:
        return candidates

    # Min-max scaling del RRF score dentro del batch
    rrf_scores = [c["rrf_score"] for c in candidates]
    min_rrf = min(rrf_scores)
    max_rrf = max(rrf_scores)
    rrf_range = max_rrf - min_rrf

    now = datetime.now()

    for c in candidates:
        # Normalizar RRF a [0, 1]
        normalized_rrf = (c["rrf_score"] - min_rrf) / rrf_range if rrf_range > 0 else 1.0

        # Temporal decay (clamp to 0 to guard against clock skew / future dates)
        ref = c.get("last_accessed_at") or c.get("created_at")
        if ref is None:
            days_ago = 0.0
        else:
            days_ago = max(0.0, (now - ref).total_seconds() / 86400)

        freq_score = c["access_count"] * math.exp(-decay_lambda * days_ago)
        c["final_score"] = alpha * normalized_rrf + beta * math.log(1 + freq_score)

    candidates.sort(key=lambda c: c["final_score"], reverse=True)
    return candidates
```

### Ejemplo con normalización

Mismo escenario, ahora con min-max scaling (min=0.009, max=0.016, range=0.007):

| Chunk | RRF | Norm. RRF | Accesos | Decay Score | Final |
|-------|-----|-----------|---------|-------------|-------|
| A (rank #1, nuevo) | 0.016 | 1.0 | 1 | log(2)=0.69 | 0.7×1.0 + 0.3×0.69 = **0.907** |
| B (rank #50, 10 acc.) | 0.009 | 0.0 | 10 | log(11)=2.40 | 0.7×0.0 + 0.3×2.40 = **0.719** |

Ahora la similitud semántica domina correctamente: A > B.

### Notas de diseño
- **Min-max scaling por batch** asegura que α/β operen sobre escalas comparables
- Si todos los RRF scores son iguales (`rrf_range = 0`), todos reciben `1.0`
- Usa `created_at` como fallback en vez de `days_ago=999`
- Usa `total_seconds() / 86400` para granularidad sub-día
- `decay_lambda` evita conflicto con keyword `lambda` de Python
- Opera sobre el batch completo (no por chunk individual) para poder normalizar

---

## Paso 3 — Access Tracking

Batch UPDATE en una sola transacción con todos los IDs del resultado final
(los `top_k` que se retornan al usuario, no los `over_fetch` candidatos).
Si `top_k=5`, 1 transacción en lugar de 5 — mantiene latencia sub-milisegundo.

```sql
UPDATE chunks
SET access_count = access_count + 1,
    last_accessed_at = CURRENT_TIMESTAMP
WHERE id IN (?, ?, ?, ?, ?)
```

```python
def track_access(db, result_ids: list[int]):
    if not result_ids:
        return
    placeholders = ",".join("?" * len(result_ids))
    db.execute(
        f"UPDATE chunks SET access_count = access_count + 1, "
        f"last_accessed_at = CURRENT_TIMESTAMP WHERE id IN ({placeholders})",
        result_ids
    )
    db.commit()
```

### Consideraciones de side effects

`search()` actualmente es **read-only** (0 API calls, 0 side effects). El tracking
introduce escrituras en cada búsqueda. Para preservar el contrato:

- **Opt-in via config:** `track_access = true` en `[scoring]` (default: `true` cuando scoring está enabled)
- **Solo se trackea cuando scoring está habilitado** — si `scoring.enabled = false`, search sigue siendo read-only
- El tracking se ejecuta **después** de construir la respuesta, así un fallo en el UPDATE no afecta los resultados
- Solo se trackean los `top_k` resultados finales, no los `over_fetch` candidatos

---

## Paso 4 — Over-fetching y Two-Stage Retrieval

### El problema

Si buscas `top_k=5` usando solo similitud semántica (ANN) en SQLite, y luego
aplicas `compute_score()` en Python a esos 5 resultados, el frequency/decay
**no está influyendo en la búsqueda real**. Solo reordenas los 5 que ya ganaron
por similitud pura. Un chunk con menor similitud pero altísima relevancia
histórica nunca llega a Python para ser evaluado.

### Solución: Over-fetch → Re-rank → Truncate

```
Query
  → embed_query()                    ← existente
  → sqlite-vec similarity (top_n)   ← over-fetch: n = scoring.over_fetch (default 50)
  → FTS5 keyword search             ← existente
  → RRF fusion (sobre top_n)        ← existente, pero sobre pool expandido
  → frequency+decay re-rank         ← NUEVO (post-RRF, sobre top_n candidatos)
  → truncate a top_k                ← retornar solo los k solicitados
  → track_access(top_k ids)         ← NUEVO (opt-in, solo los k finales)
  → return results
```

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `top_k` | 5 | Resultados finales retornados al usuario |
| `over_fetch` | 50 | Candidatos recuperados de SQLite para re-ranking |

**Trade-off de latencia:** recuperar 50 vs 5 de sqlite-vec es ~10x más datos,
pero sigue siendo sub-milisegundo para bases < 1M chunks. El `compute_score()`
en Python sobre 50 items es negligible. Para bases más grandes, el `over_fetch`
es configurable.

### Archivos a modificar

- `vstash/store.py` — migration, `track_access()`, over-fetch + re-rank en `search()`
- `vstash/config.py` — nueva sección `ScoringConfig`
- `vstash/memory.py` / `vstash/mcp.py` — pasar config de scoring

### Optimización futura: scoring nativo en SQLite

Para escalar más allá de 1M chunks, el `compute_score()` puede implementarse
como extensión nativa de SQLite (C/Rust), permitiendo `ORDER BY final_score`
directamente en la query sin pasar datos a Python:

```sql
-- Hipotético con extensión nativa
SELECT *, vstash_score(rrf_score, access_count, last_accessed_at, created_at,
                       0.7, 0.3, 0.07) AS final_score
FROM candidates
ORDER BY final_score DESC
LIMIT ?
```

Esto elimina la serialización SQLite→Python→sort→truncate y mantiene todo
dentro del motor de la base de datos. Prioridad: v2 (post-validación del modelo).

---

## Paso 5 — Simulación con Papers

Usar fechas de publicación como proxy temporal para simular decay histórico
sin datos de uso reales.

### Setup de la simulación

Para que la fórmula tenga valores reales que decaer, el setup debe:

1. Mapear `publication_date` → `created_at` **y** `last_accessed_at`
2. Inicializar `access_count = 1` (el ingestion count por defecto)

**Por qué ambos campos:** si solo se mapea `created_at` y `last_accessed_at`
queda en NULL, el fallback a `created_at` haría que todos los papers tuvieran
el mismo `days_ago` base (su fecha de publicación) y `access_count = 1` — el
scoring sería monótono con la fecha, sin variación interesante.

La simulación debe:
- Papers fase 1: todos con `access_count = 1`, `last_accessed_at = publication_date`
- Papers fase 2: simular uso diferencial (LoCoMo accedido 10x recientemente,
  MemGPT 1x hace meses) para observar el efecto del frequency+decay

### Protocolo

1. Ingestar los 25 papers en vstash con `publication_date` → `created_at` y `last_accessed_at`
2. Correr queries baseline con RRF puro — registrar ranking
3. Activar frequency+decay — correr mismas queries — registrar nuevo ranking
4. Simular patrones de uso diferencial:
   - `track_access()` en LoCoMo 10 veces (simula uso frecuente reciente)
   - `track_access()` en MemGPT 1 vez (simula consulta única antigua)
5. Re-query y observar si LoCoMo sube relativo a MemGPT con similitud semántica igual
6. Variar λ entre 0.01 y 0.5 — plotear score vs λ por paper

### Patrones esperados

- Papers 2026 (PAM, E-mem) > papers 2023 en queries temporales con similitud semántica igual
- LoCoMo (2024) permanece alto por co-ocurrencia universal con todas las queries de memoria
- MemGPT (2023) decae lento — conceptualmente foundational pero menos frecuente
- Papers de scope estrecho decaen más rápido que benchmarks transversales

---

## Corpus de Papers (25)

### Foundational (2023)
| Año | Título | URL |
|-----|--------|-----|
| 2023 | MemGPT: Towards LLMs as Operating Systems | arxiv.org/abs/2310.08560 |
| 2023 | Generative Agents: Interactive Simulacra of Human Behavior | arxiv.org/abs/2304.03442 |
| 2023 | RET-LLM: Towards a General Read-Write Memory for LLMs | arxiv.org/abs/2305.14322 |
| 2023 | SCM: Self-Controlled Memory Framework | arxiv.org/abs/2304.13343 |

### Benchmarks & Evaluation (2024–2025)
| Año | Título | URL |
|-----|--------|-----|
| 2024 | LoCoMo: Evaluating Very Long-Term Conversational Memory | arxiv.org/abs/2402.17753 |
| 2024 | SORT: Sequence Order Recall Tasks for Episodic Memory | arxiv.org/abs/2410.08133 |
| 2025 | Episodic Memories Generation and Evaluation Benchmark | arxiv.org/abs/2501.13121 |

### Memory Architectures (2024–2025)
| Año | Título | URL |
|-----|--------|-----|
| 2024 | MemoryBank: Enhancing LLMs with Long-Term Memory | arxiv.org/abs/2305.10250 |
| 2024 | Dynamic Tree Memory Representation for LLMs | arxiv.org/abs/2410.14052 |
| 2025 | A-MEM: Agentic Memory for LLM Agents | arxiv.org/abs/2502.12110 |
| 2025 | Mem0: Production-Ready AI Agents with Scalable Long-Term Memory | arxiv.org/abs/2504.19413 |
| 2025 | Memoria: Scalable Agentic Memory for Personalized Conversational AI | arxiv.org/abs/2512.12686 |
| 2025 | Memori: Persistent Memory Layer for Context-Aware LLM Agents | arxiv.org/abs/2603.19935 |
| 2025 | Nemori: Self-Organizing Agent Memory Inspired by Cognitive Science | arxiv.org/abs/2508.03341 |
| 2025 | SeCom: Memory Construction for Personalized Conversational Agents | arxiv.org/abs/2502.05589 |
| 2025 | R³Mem: Memory Retention and Retrieval via Reversible Compression | arxiv.org/abs/2502.15957 |

### Temporal & Decay (2025–2026)
| Año | Título | URL |
|-----|--------|-----|
| 2025 | Zep: Temporal Knowledge Graph Architecture for Agent Memory | arxiv.org/abs/2501.13956 |
| 2025 | WMR: Memory Retrieval via LLM-Trained Cross Attention Networks | frontiersin.org/articles/10.3389/fpsyg.2025.1591618 |
| 2025 | MaRS: Forgetful but Faithful — Cognitive Memory Architecture | arxiv.org/abs/2512.12856 |
| 2026 | PAM: Predictive Associative Memory via Temporal Co-occurrence | arxiv.org/abs/2602.11322 |

### Advanced Systems (2025–2026)
| Año | Título | URL |
|-----|--------|-----|
| 2026 | E-mem: Multi-agent Episodic Context Reconstruction | arxiv.org/abs/2601.21714 |
| 2026 | Beyond Fact Retrieval: Episodic Memory for RAG with GSW | arxiv.org/abs/2511.07587 |
| 2026 | MAGMA: Multi-Graph based Agentic Memory Architecture | arxiv.org/abs/2601.03236 |
| 2026 | EverMemOS: Self-Organizing Memory OS for Long-Horizon Reasoning | arxiv.org/abs/2601.02163 |
| 2024 | AI PERSONA: Towards Life-long Personalization of LLMs | arxiv.org/abs/2412.13103 |

---

## Prior Art — Dónde está el gap

| Sistema | ¿Tiene decay? | Gap |
|---------|--------------|-----|
| MemoryBank (2024) | Sí — Ebbinghaus | Sin frecuencia de acceso |
| Generative Agents (2023) | Parcial — recencia | Sin conteo acumulado |
| Zep (2025) | KG temporal | Pesado, cloud-dependent |
| WMR (2025) | Sí — 0.995/hora | Sin frecuencia; requiere LLM para importance |
| Mem0 (2025) | No | Solo extracción y consolidación |
| **vstash + esta propuesta** | **Sí — frecuencia × decay** | **Local, SQLite, sin LLM calls, sub-ms** |

---

## Trade-offs y Mitigaciones

### Confirmation Bias

**Riesgo:** chunks frecuentes suben → se consultan más → suben más.

**Mitigaciones:**
1. **Decay temporal** rompe el ciclo — un chunk con `access_count` alto pero sin acceso en 60 días decae significativamente, cediendo espacio a contexto nuevo relevante.
2. **`log(1 + freq_score)`** satura el boost — la diferencia entre 100 y 1000 accesos es mucho menor que entre 0 y 10.
3. **α=0.8** (post-experimento con PDFs completos) da prioridad a la semántica — con corpus ruidosos, la similitud necesita dominar para filtrar antes de que la memoria ajuste.

### Over-fetch vs Latencia

**Riesgo:** recuperar 50 candidatos en vez de 5 incrementa I/O y memoria.

**Mitigación:** sqlite-vec es sub-milisegundo para pools < 1M chunks. El
re-ranking en Python sobre 50 floats es ~microsegundos. El `over_fetch` es
configurable para ajustar el trade-off según el tamaño del corpus.

### Cold Start

**Riesgo:** chunks recién ingestados sin historial quedan en desventaja.

**Mitigación:** `access_count = 1` al crear + `created_at` como referencia
temporal dan una ventana de competitividad inicial que decae naturalmente.

---

## Resultados Experimentales (Marzo 2026)

### Configuración del Experimento

- **Corpus:** 24 papers de arXiv (2023–2026), **786 chunks** (PDFs completos, no abstracts)
- **Queries:** 10 queries con relevance judgments manuales (NDCG@10)
- **Escenarios:** 5 patrones de acceso (uniform, recent_heavy_use, stale_favorites, mixed_recency, benchmark_focused)
- **Grid:** 16 configuraciones (α: 0.4–0.9, β: 0.1–0.6, λ: 0.01–0.20, over_fetch: 20–100)
- **Baseline NDCG@10:** 0.6081 (RRF puro, sin scoring)

### Defaults Óptimos (validados por grid search — PDFs completos)

```toml
[scoring]
enabled = true
alpha = 0.8           # avg NDCG 0.6357 (+4.5% vs baseline 0.6081)
beta = 0.2
decay_lambda = 0.05   # decay conservador — semanas, no días
over_fetch = 50
track_access = true
```

> **Metodología:** Defaults seleccionados por **avg NDCG across 5 scenarios**
> (uniform, recent_heavy_use, stale_favorites, mixed_recency, benchmark_focused),
> no por el mejor escenario individual. El +4.5% avg es la métrica principal.
>
> **Corpus:** 786 chunks de 24 PDFs completos de arXiv (no abstracts).
> El baseline es más bajo (0.6081 vs 0.8225 con abstracts) porque PDFs
> completos incluyen tablas, referencias y headers que agregan ruido.
> Con más ruido, la semántica necesita mayor peso (α=0.8) para filtrar
> antes de que la memoria ajuste.
>
> **Caso especial — benchmark_focused:** Cuando los papers con access_count
> elevado coinciden con los relevance judgments, el lift puede alcanzar +18.8%.
> Este escenario es partly circular (frequency boost alineado con ground truth)
> y no debe usarse como métrica principal para elegir defaults.

### Comportamiento en los Bordes — Guía de Configuración

> **Nota:** Los valores de NDCG a continuación provienen del grid search con PDFs
> completos (baseline 0.6081). Los valores cualitativos (comportamiento, cuándo
> elegir) se mantienen válidos independientemente del corpus.

El espectro α/β controla el equilibrio entre **"qué es relevante"** (semántica)
y **"qué ha sido útil"** (memoria). Cada extremo tiene un perfil de usuario distinto:

#### α = 0.9, β = 0.1 — Modo Semántico Casi Puro

```
final_score = 0.9 · sim + 0.1 · memory
```

| Métrica | Valor observado (avg 5 scenarios) |
|---------|-----------------------------------|
| Avg NDCG@10 | 0.6273 (+3.2% vs baseline) |
| Displacement | 1.25–1.51 |
| Recency corr. | +0.549–0.670 |

**Comportamiento:** La semántica domina. La memoria es un desempate muy sutil.
Resultados casi idénticos a RRF puro, pero chunks con uso reciente ganan en
empates semánticos.

**Cuándo elegir:** Corpus estático o cuando la reproducibilidad importa.

#### α = 0.8, β = 0.2 — Modo Semántica Dominante (default óptimo) ★

```
final_score = 0.8 · sim + 0.2 · memory
```

| Métrica | Valor observado (avg 5 scenarios) |
|---------|-----------------------------------|
| Avg NDCG@10 | 0.6357 (+4.5% vs baseline) |
| Min NDCG | 0.5910 (mixed_recency) |
| Max NDCG | 0.7220 (benchmark_focused) |
| Displacement | 1.34–1.96 |
| Recency corr. | +0.576–0.740 |

**Comportamiento:** La semántica determina el ranking base. La memoria actúa
como boost moderado — suficiente para subir chunks frecuentemente accedidos
1-2 posiciones, pero no para enterrar resultados semánticamente relevantes.

**Cuándo elegir:** Default recomendado. Equilibrio robusto entre relevancia
semántica y señal de uso, validado con corpus ruidoso de PDFs completos.

#### α = 0.7, β = 0.3 — Modo Balanceado

```
final_score = 0.7 · sim + 0.3 · memory
```

| Métrica | Valor observado (avg 5 scenarios) |
|---------|-----------------------------------|
| Avg NDCG@10 | 0.5990–0.6393 (varía por λ y scenario) |
| Displacement | 1.37–2.30 |
| Recency corr. | +0.575–0.704 |

**Comportamiento:** Equilibrio entre semántica y memoria. Más sensible a
patrones de acceso que α=0.8. Funciona bien con historial de uso maduro,
pero puede empeorar vs baseline en uniform scenario.

**Cuándo elegir:** Usuarios activos cuyo patrón de acceso es señal confiable.

#### α = 0.5, β = 0.5 — Modo Memoria Fuerte

```
final_score = 0.5 · sim + 0.5 · memory
```

| Métrica | Valor observado (avg 5 scenarios) |
|---------|-----------------------------------|
| Avg NDCG@10 | 0.5478–0.6410 (alta varianza por scenario) |
| Displacement | 1.57–2.57 |
| Recency corr. | +0.614–0.796 |

**Comportamiento:** La memoria compite de igual a igual con la semántica.
Papers recientes con uso frecuente dominan. Riesgo de confirmation bias
significativo en corpus ruidosos.

**Cuándo elegir:** Solo con historial de uso maduro y corpus limpio.
**No recomendado** como default con PDFs — el ruido del contenido necesita
más filtrado semántico.

#### α = 0.4, β = 0.6 — Modo Memoria Dominante

```
final_score = 0.4 · sim + 0.6 · memory
```

| Métrica | Valor observado (avg 5 scenarios) |
|---------|-----------------------------------|
| Avg NDCG@10 | 0.5523 (-9.2% vs baseline) |
| Displacement | 1.92 (máximo) |
| Recency corr. | +0.764 (máximo) |

**Comportamiento:** La memoria domina el ranking. Papers populares enterran
papers relevantes no consultados. Con corpus ruidoso esto **empeora** vs
baseline — el ruido semántico no se filtra y el frequency boost amplifica
chunks irrelevantes pero frecuentes.

**Cuándo elegir:** Flujos donde recencia es señal fuerte (noticias, notas
de investigación acotada). **No recomendado** como default.

#### Efecto de λ (decay rate)

Con acceso uniforme (access_count=1 para todos), λ tiene efecto **nulo** — todos
los papers decaen por igual. λ solo se diferencia cuando hay historial variado:

| λ | Efecto | Analogía |
|---|--------|----------|
| 0.01 | Decay muy lento (meses) | Memoria de largo plazo — nada se olvida rápido |
| 0.05 | Decay semanal ★ | Balance: contenido de hace 2 semanas aún pesa |
| 0.10 | Decay diario | Memoria de trabajo — lo de ayer ya pierde fuerza |
| 0.20 | Decay agresivo | Solo lo de hoy importa; historia es irrelevante |
| 0.50 | Decay extremo | Efectivamente anula la frecuencia histórica |

**Hallazgo:** λ=0.05 es conservador y seguro como default. Con datos reales
de uso, λ=0.07–0.10 podría ser mejor para usuarios que quieren frescura
agresiva. El valor óptimo depende del dominio:
- **Código/docs técnicos:** λ=0.03–0.05 (el conocimiento envejece lento)
- **Notas de investigación:** λ=0.07–0.10 (el contexto cambia rápido)
- **Noticias/feeds:** λ=0.15–0.20 (solo importa lo reciente)

#### Efecto de over_fetch

Con 251 chunks, over_fetch=20/50/100 producen resultados **idénticos**. El
over_fetch solo se diferencia con corpus grandes (>10K chunks) donde el pool
de candidatos semánticos es más competitivo.

| over_fetch | Cuándo importa |
|-----------|----------------|
| 20 | Corpus < 1K chunks (bajo overhead, mismo resultado) |
| 50 ★ | Default seguro para la mayoría de los casos |
| 100+ | Corpus > 50K chunks donde similitudes están más comprimidas |

---

## Oportunidades de Mejora Identificadas

### 1. Importance Scoring — más allá de frecuencia

**Problema:** La fórmula actual asume que todos los accesos son iguales. Un
acceso casual ("vi el título en un listado") pesa igual que un acceso profundo
("leí el chunk completo y lo usé en mi respuesta").

**Propuesta:** Agregar un peso por tipo de acceso:
```
weighted_access = sum(weight_i * e^(-lambda * days_ago_i))
```
Donde `weight_i` varía según contexto: `search_result=1.0`, `ask_context=2.0`,
`explicit_bookmark=5.0`. Requiere extender el schema con un log de accesos
tipados en vez de un simple counter.

### 2. Per-Collection Scoring Params

**Problema:** Un solo set de α/β/λ para todo el store. Un usuario puede
tener una colección de "referencia permanente" (α alto) y otra de "notas
de investigación" (β alto).

**Propuesta:** Permitir override de scoring params por colección:
```toml
[scoring.collections.reference]
alpha = 0.8
beta = 0.2

[scoring.collections.research]
alpha = 0.4
beta = 0.6
decay_lambda = 0.10
```

### 3. Adaptive α/β via Feedback Loop

**Problema:** Los defaults óptimos dependen del dominio y el patrón de uso
del usuario. No hay una configuración universalmente óptima.

**Propuesta:** Tracking implícito de "satisfacción": si el usuario hace
`ask()` y recibe chunks que usa (medido por si el LLM los cita en la respuesta),
eso es señal positiva. Ajustar α/β gradualmente según la señal:
- Si chunks con alto score de memoria son citados → incrementar β
- Si chunks solo semánticamente relevantes son citados → incrementar α

### 4. Decay Reset on Re-ingestion

**Problema:** Si un documento se actualiza y se re-ingesta (`force=True`),
los chunks nuevos pierden todo el historial de acceso. El usuario pierde
la "memoria" de que ese contenido era frecuentemente consultado.

**Propuesta:** En re-ingestion, transferir `access_count` del chunk más
similar (por embedding distance) del documento anterior al nuevo chunk.
Preserva la señal de frecuencia a través de actualizaciones de contenido.

### 5. Título Real en Documentos Ingestados por URL

**Hallazgo del experimento:** `vstash` almacena la URL como `title` cuando
se ingesta desde HTTP. Esto dificulta la legibilidad de resultados y el
matching en evaluación. Extraer el `<title>` del HTML/PDF durante la ingesta
mejoraría UX y permitiría matching semántico por título.

### 6. Scoring Nativo en SQLite (v2 — ya planificado)

Para corpus >1M chunks, mover `rerank_with_decay()` a una extensión C/Rust
de SQLite eliminaría la serialización Python y permitiría `ORDER BY final_score`
directamente en la query.

### 7. Chunk Size / Overlap Grid Search

**Problema:** Los defaults de chunking (1024 tokens, 128 overlap) están validados
por el benchmark de chunking pero nunca se han evaluado en interacción con el
scoring de frequency+decay. Es posible que chunks más pequeños (mayor granularidad)
o más grandes (mayor contexto) interactúen de forma no obvia con el re-ranking.

**Propuesta:** Evaluar 3–4 configuraciones de chunking como experimento aislado:

| Config | size | overlap | Chunks esperados (~24 papers) |
|--------|------|---------|-------------------------------|
| Granular | 512 | 64 | ~500 |
| Default | 1024 | 128 | ~250 |
| Amplio | 2048 | 256 | ~130 |

**Protocolo:**
1. Re-ingestar el corpus completo con cada configuración de chunking
2. Medir retrieval base (recall@10, MRR) con RRF puro — sin scoring
3. Aplicar scoring óptimo (α=0.5, β=0.5, λ=0.05) y medir NDCG@10
4. Cruzar resultados para identificar si hay interacción chunking × scoring

**Nota:** Cada configuración requiere re-ingesta completa (nuevos embeddings),
por lo que este experimento es significativamente más costoso que el grid search
de scoring. Debe ejecutarse como proyecto separado, no mezclado con el tuning
de α/β/λ.

---

## Next Steps

- [x] Verificar URLs placeholder del corpus
- [x] Aplicar schema migration
- [x] Agregar `ScoringConfig` a `vstash/config.py`
- [x] Implementar over-fetch en `store.py`
- [x] Implementar `rerank_with_decay()` como post-RRF re-ranker en `store.py`
- [x] Implementar `track_access()` en `store.py`
- [x] Ingestar 24 papers con `publication_date` → `created_at` + `last_accessed_at`
- [x] Correr queries baseline (RRF puro) vs frequency+decay — comparar rankings
- [x] Grid search: 16 configs × 5 escenarios × 10 queries = 800 evaluaciones
- [x] Actualizar defaults a α=0.8, β=0.2, λ=0.05, enabled=true (re-validado con PDFs)
- [ ] PR a vstash + publicar findings
- [ ] v2: importance scoring (weighted access types)
- [ ] v2: per-collection scoring params
- [ ] v2: título real en documentos ingestados por URL
- [ ] v2: chunk size/overlap grid search (interacción con scoring)
- [ ] v3: scoring nativo en SQLite (extensión C/Rust)

---

*Generado: Marzo 2026 — Actualizado: Marzo 2026*
