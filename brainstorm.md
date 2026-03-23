# vstash — Brainstorm de Aplicaciones

Ideas exploradas el 2026-03-23.

---

## 🌐 1. Web App sin Backend (vstash como BaaS)
Frontend (HTML/JS) habla directo al MCP server. vstash ES el backend:
- `vstash_add` → sube documentos
- `vstash_search` → semantic search
- `vstash_ask` → chat con tus docs
- `vstash_export` → descarga datasets

Solo necesita un bridge MCP→HTTP (FastAPI ~50 líneas o `mcp-ui-server`).

## 🤖 2. Knowledge Base para Customer Support
Soporte carga FAQs, docs, playbooks → agentes buscan respuestas instantáneamente.
Con `project=cliente_x` cada agente ve solo los docs de su cliente.

## 📚 3. Research Assistant
Ingestas papers (PDF), notas, transcripciones de meetings.
Búsqueda semántica: "¿Qué papers mencionan multi-agent architectures?"

## 🧪 4. Test Case Library
QA teams guardan test cases con metadata (`layer=regression`, `tags=login,auth`).
Deduplicación inteligente de escenarios similares.

## 📝 5. Smart Journal / Daily Notes
Notas diarias con frontmatter → búsqueda semántica sobre tu historial personal.
"¿Qué ideas tuve sobre X en las últimas semanas?"

## 🎓 6. Course/Study Companion
Slides, textbooks, apuntes de un curso → estudio asistido por RAG.

## 🏗️ 7. Architecture Decision Records (ADR)
Teams guardan decisiones arquitectónicas. Meses después:
"¿Por qué elegimos PostgreSQL sobre MongoDB?"

## 🔧 8. Training Data Curation Pipeline
vstash como front-end de curación para fine-tuning:
ingest docs → curate con filtros → `vstash export` → HF Dataset → fine-tune.

## 🤝 9. Multi-Agent Shared Memory
Equipo de agentes (Agno/LangGraph/Temporal) compartiendo vstash como memoria común.
`project` per agent, WAL mode para concurrencia.
