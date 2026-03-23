# vstash Python SDK — Plan de Implementación
> Phase 4: `from vstash import Memory`

---

## Objetivo

Exponer vstash como un building block reutilizable para agentes y pipelines Python.
La interfaz tiene que ser lo suficientemente simple para usarse en 5 líneas,
y lo suficientemente poderosa para escalar a multi-agent shared memory.

---

## API Surface (diseño final)

```python
from vstash import Memory

# --- Caso básico ---
mem = Memory()
mem.add("docs/spec.pdf")
answer = mem.ask("¿Cuáles son los requisitos del sistema?")

# --- Con scoping por proyecto ---
mem = Memory(project="agent_planner")
mem.add("notes/meeting.md")
chunks = mem.search("decisiones de arquitectura", top_k=5)

# --- Context manager ---
with Memory(project="analysis", collection="q1") as mem:
    mem.add("reports/q1.pdf")
    answer = mem.ask("¿Cuál fue el revenue?")

# --- Para agentes: búsqueda sin LLM ---
chunks = mem.search("deployment strategy")
for c in chunks:
    print(c.text, c.score)   # SearchResult: text, title, path, chunk, score

# --- Gestión ---
mem.remove("docs/old.pdf")
mem.list()      # → list[DocumentInfo]
mem.stats()     # → StoreStats
```

---

## Clase `Memory` — Especificación

### Constructor

```python
Memory(
    config: str | Path | None = None,   # path a vstash.toml, default: auto-detect
    project: str | None = None,         # filtra y etiqueta todos los add/search
    collection: str = "default",        # colección por defecto
    db: str | Path | None = None,       # override del path al .db (útil en tests)
)
```

**Regla de resolución de config** (misma lógica que la CLI):
1. Argumento `config` explícito
2. `VSTASH_CONFIG` env var
3. `./vstash.toml` en el directorio actual
4. `~/.vstash/vstash.toml`

### Métodos públicos

| Método | Firma | Descripción |
|--------|-------|-------------|
| `add` | `(source, *, collection?, project?, layer?, tags?) → IngestResult` | Ingesta un archivo o URL |
| `search` | `(query, *, top_k=5, collection?, project?, layer?) → list[SearchResult]` | Búsqueda semántica sin LLM |
| `ask` | `(query, *, top_k=5, collection?, project?, layer?, history?) → str` | Búsqueda + respuesta LLM |
| `remove` | `(source) → bool` | Elimina un documento del store |
| `list` | `(*, collection?, project?, layer?) → list[DocumentInfo]` | Lista documentos ingrestos |
| `stats` | `() → StoreStats` | Estadísticas del store |

**Nota de diseño:** los parámetros `collection`, `project`, `layer` en cada método
*sobreescriben* los defaults del constructor. Esto permite un objeto `Memory` con
`project="agent_x"` pero con la flexibilidad de hacer búsquedas cross-project pasando
`project=None` explícitamente.

### Context manager

`Memory` implementa `__enter__` / `__exit__` y cierra el `VstashStore` al salir.
Para uso en scripts de larga vida (agentes, servidores), instanciar sin context manager
y llamar `mem.close()` manualmente.

---

## Decisiones de diseño

### 1. Sync-first, async luego
La implementación inicial es completamente síncrona — igual que la CLI y el MCP server.
Async (`async def add`, `async def ask`) se agrega en una segunda iteración una vez
que la API síncrona esté validada. La razón: los frameworks de agentes más usados
(LangGraph, Agno) tienen wrappers sync→async propios; no vale la pena la complejidad
hasta tener un caso real que lo requiera.

### 2. `Memory` no es un Singleton
Cada instancia abre su propia conexión al `.db`. El `VstashStore` ya tiene WAL mode
activado, que soporta múltiples readers concurrentes + un writer. Para multi-agent,
varios procesos pueden instanciar `Memory` sobre el mismo `.db` de forma segura.

### 3. Los filtros son opcionales, no obligatorios
`search("query")` sin filtros busca en todo el store. Esto es intencional: el usuario
puede elegir el nivel de aislamiento que necesita. No forzar `project` simplifica el
onboarding.

### 4. `IngestResult` en lugar de `None` en `add`
`add()` retorna el `IngestResult` existente (chunks ingrestos, título detectado, etc.)
en lugar de `None`. Útil para pipelines que necesitan saber cuántos chunks se
generaron o si el documento fue re-ingresado.

### 5. No re-exportar internals en `__init__.py`
Solo `Memory` y los modelos de resultado (`SearchResult`, `DocumentInfo`, `StoreStats`,
`IngestResult`) se exponen en el paquete público. El resto (`VstashStore`, `embed`,
`chat`) quedan como API privada — permite refactoring interno sin breaking changes.

---

## Estructura de archivos

```
vstash/
├── __init__.py          # expone Memory + modelos públicos  ← MODIFICAR
├── memory.py            # clase Memory                      ← CREAR
├── store.py             # VstashStore (sin cambios)
├── ingest.py            # pipeline de ingesta (sin cambios)
├── chat.py              # inference backends (sin cambios)
├── embed.py             # FastEmbed wrapper (sin cambios)
├── config.py            # VstashConfig (sin cambios)
├── models.py            # Pydantic models (sin cambios)
├── cli.py               # CLI typer (sin cambios)
└── mcp.py               # MCP server (sin cambios)
```

Solo dos archivos cambian: `memory.py` (nuevo) y `__init__.py` (agrega exports).
Cero breaking changes en la CLI ni en el MCP server.

---

## Implementación de `memory.py` — Esqueleto

```python
from __future__ import annotations

from pathlib import Path
from types import TracebackType

from .config import VstashConfig
from .embed import get_embedder
from .ingest import ingest_file
from .models import DocumentInfo, IngestResult, SearchResult, StoreStats
from .store import VstashStore
from . import chat


class Memory:
    """High-level Python SDK for vstash.

    Drop any document. Ask anything. Get an answer in under a second.

    Args:
        config: Path to vstash.toml. Auto-detected if not provided.
        project: Default project tag for add/search operations.
        collection: Default collection name (default: "default").
        db: Override path to the SQLite database file.

    Example::

        from vstash import Memory

        mem = Memory(project="my_agent")
        mem.add("docs/spec.pdf")
        answer = mem.ask("What are the system requirements?")
    """

    def __init__(
        self,
        config: str | Path | None = None,
        *,
        project: str | None = None,
        collection: str = "default",
        db: str | Path | None = None,
    ) -> None:
        self._cfg = VstashConfig.load(config)
        if db:
            self._cfg.db_path = str(db)
        self._project = project
        self._collection = collection
        self._store = VstashStore(self._cfg.db_path)
        self._embedder = get_embedder(self._cfg)

    # ------------------------------------------------------------------ #
    # Context manager                                                      #
    # ------------------------------------------------------------------ #

    def __enter__(self) -> Memory:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def add(
        self,
        source: str | Path,
        *,
        collection: str | None = None,
        project: str | None = None,
        layer: str | None = None,
        tags: str | None = None,
    ) -> IngestResult:
        """Ingest a file or URL into memory."""
        ...

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        collection: str | None = None,
        project: str | None = None,
        layer: str | None = None,
    ) -> list[SearchResult]:
        """Semantic search without LLM inference."""
        ...

    def ask(
        self,
        query: str,
        *,
        top_k: int = 5,
        collection: str | None = None,
        project: str | None = None,
        layer: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """Search + LLM answer generation."""
        ...

    def remove(self, source: str | Path) -> bool:
        """Remove a document from memory."""
        return self._store.delete_document(str(source))

    def list(
        self,
        *,
        collection: str | None = None,
        project: str | None = None,
        layer: str | None = None,
    ) -> list[DocumentInfo]:
        """List ingested documents."""
        return self._store.list_documents(
            collection=collection or self._collection,
            project=project or self._project,
            layer=layer,
        )

    def stats(self) -> StoreStats:
        """Return memory statistics."""
        return self._store.stats()

    def close(self) -> None:
        """Close the database connection."""
        self._store.close()
```

---

## Tests a agregar

Los tests existentes (`tests/`) cubren el store, embed, ingest y MCP.
Para el SDK agregar `tests/test_memory.py` con:

- `test_memory_add_and_ask` — ingest + ask sobre un string temporal
- `test_memory_search_returns_results` — search retorna `SearchResult`
- `test_memory_project_filter` — búsqueda scoped por project no retorna docs de otro project
- `test_memory_context_manager` — el `with Memory()` cierra la conexión correctamente
- `test_memory_remove` — remove devuelve True y el doc desaparece de list()
- `test_memory_db_override` — `db=tmp_path` usa el path correcto

---

## Exports públicos en `__init__.py`

```python
from .memory import Memory
from .models import DocumentInfo, IngestResult, SearchResult, StoreStats

__all__ = [
    "Memory",
    "SearchResult",
    "DocumentInfo",
    "IngestResult",
    "StoreStats",
]
```

---

## Roadmap de iteraciones

| Iteración | Qué se agrega | Prioridad |
|-----------|---------------|-----------|
| v0.3.0 | `Memory` síncrona, exports públicos, 6 tests nuevos | Alta |
| v0.3.1 | `mem.stream(query)` — generator de tokens para UIs | Media |
| v0.3.2 | `AsyncMemory` — wrapper async para agentes event-loop | Media |
| v0.4.0 | Demo: agente LangGraph usando vstash como memoria | Baja |

---

## Criterio de "listo"

- [ ] `from vstash import Memory` funciona sin imports adicionales
- [ ] `mem.add("file.pdf"); mem.ask("question")` funciona en < 10 líneas
- [ ] Los 6 tests nuevos pasan junto con los 72 existentes
- [ ] Documentado en README con un ejemplo de agent integration
- [ ] Sin breaking changes: CLI y MCP siguen funcionando idéntico
