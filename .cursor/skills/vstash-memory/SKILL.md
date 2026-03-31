---
name: vstash-memory
description: >-
  Queries the user's vstash local document memory (hybrid semantic search over
  ~/.vstash/memory.db). Use when the user asks about saved notes, specs, prior
  decisions, ingested docs, "what did I stash", vstash, or anything that should
  be answered from their personal knowledge base rather than only the current
  repo. Prefer MCP vstash_* tools when available; otherwise run the vstash CLI
  in the terminal with the user's configured Python environment.
---

# vstash — memoria local del usuario

## Objetivo

Usar la misma base que Claude Desktop / MCP: **vstash** (SQLite + embeddings + FTS). No inventar contenido de memoria; **buscar o preguntar** y citar fuentes.

## Prioridad de herramientas

1. **Si existen herramientas MCP** `vstash_search`, `vstash_ask`, `vstash_list`, `vstash_stats`, etc., úsalas primero (misma integración que Claude).
2. **Si no hay MCP**, ejecuta el CLI en terminal (ver abajo). Asegúrate de que `vstash` sea el del entorno donde está instalado (`which vstash`).

## CLI (fallback)

Desde el directorio del proyecto o cualquier cwd (la DB es global salvo `VSTASH_CONFIG` / `vstash.toml`):

- **Solo recuperación, sin LLM**: `vstash search "pregunta o keywords" [--json]`
- **Pregunta con respuesta generada** (requiere backend en `~/.vstash/vstash.toml`): `vstash ask "pregunta"`
- **Estado de la memoria**: `vstash stats`, `vstash list`
- **Recordar texto**: `vstash remember "..." --title "slug"` o pipe por stdin

Opciones útiles de filtrado: `--collection`, `--project`, `--layer`, `--top-k`.

## Cuándo usar

- Preguntas de conocimiento sobre documentos que el usuario guardó en vstash.
- "Busca en mi memoria / en vstash / lo que guardé sobre X".

## Cuándo no usar

- Estado actual de git, builds, o código abierto en el editor (usa lectura del repo).
- Órdenes puramente mecánicas (commit, deploy) sin necesidad de contexto documental.
- El usuario no usa vstash y no ha pedido consultar memoria externa.

## Configuración MCP en Cursor (para igualar a Claude)

En **Cursor → Settings → MCP**, añade un servidor que ejecute el mismo binario que Claude Desktop:

- Comando: `vstash-mcp` (o ruta absoluta si usas pyenv, p. ej. `~/.pyenv/versions/3.12.x/bin/vstash-mcp`).
- Requiere: `pip install vstash[mcp]` (o el editable del repo con extra `mcp`).

Las API keys para `vstash_ask` deben estar en `~/.vstash/vstash.toml`; MCP no hereda siempre el entorno del shell.

## Copiar la skill a todos los proyectos

Esta skill está en `.cursor/skills/vstash-memory/` del repo. Para que aplique en **cualquier carpeta**, copia o enlaza el directorio a `~/.cursor/skills/vstash-memory/`.
