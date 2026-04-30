"""
retrain_synth.py -- LLM-based query synthesis for the retrain pipeline.

Chunk prefixes make terrible training queries: they look nothing like
what a real user types, so the fine-tuned model only learns to match
chunks to their own prefix. This module replaces the prefix pseudo-
queries with short, realistic queries produced by a local LLM
(Ollama, LM Studio, OpenAI-compatible) via the backend configured in
vstash.toml.

Results are JSONL-cached by (chunk_id, prompt_hash, model) so
repeated runs over the same corpus are free.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from collections.abc import Iterable

from .config import VstashConfig

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = (
    "You are helping build training data for a retrieval system.\n\n"
    "Given the following passage, write exactly {n} short natural queries "
    "(5 to 15 words each) that a user would type into a search engine to "
    "find this passage. Each query should be distinct and cover a different "
    "angle of the passage.\n\n"
    "Output ONLY a JSON array of {n} strings, nothing else. Do not include "
    "any preamble, explanation, or markdown fences.\n\n"
    'Passage:\n"""\n{text}\n"""\n\n'
    "JSON array of queries:"
)

_JSON_ARRAY_RE = re.compile(r"\[[^\[\]]*\]", re.DOTALL)

# Conservative ceilings. Individual LLMs can misbehave on huge passages
# (truncation, refusal to answer) so we cap input at 1500 chars.
_MAX_PASSAGE_CHARS = 1500
_MAX_COMPLETION_TOKENS = 400


# ---------------------------------------------------------------------- #
# LLM wire calls                                                          #
# ---------------------------------------------------------------------- #


def _llm_complete(cfg: VstashConfig, prompt: str, model: str | None = None) -> str:
    """Single-turn LLM completion using the backend configured in ``cfg``.

    This is intentionally thin: no retrieval context, no chat history,
    no streaming. Synthesis is a pure prompt-in, text-out operation.

    Args:
        cfg: Resolved VstashConfig. ``inference.backend`` must resolve
            to openai-compatible, ollama, or cerebras.
        prompt: The full user prompt to send.
        model: Optional model override. Falls back to the backend's
            configured default.

    Returns:
        Raw completion text from the model.

    Raises:
        ImportError: If the openai SDK is not installed.
        ValueError: If the backend is unsupported or missing credentials.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "Query synthesis requires the openai SDK. Install with: pip install openai"
        ) from exc

    from .chat import _resolve_local_config  # keep chat.py the source of truth

    if cfg.inference.backend.lower() == "local":
        cfg = _resolve_local_config(cfg)

    backend = cfg.inference.backend.lower()
    if backend == "openai":
        client = OpenAI(api_key=cfg.openai_api_key, base_url=cfg.openai.base_url)
        use_model = model or cfg.openai.model
        extra_body = cfg.openai.extra_body
    elif backend == "ollama":
        # OllamaConfig exposes ``host`` (matching chat.py); ensure /v1
        # so the OpenAI client targets Ollama's openai-compat endpoint.
        base_url = cfg.ollama.host.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = base_url + "/v1"
        client = OpenAI(api_key="ollama", base_url=base_url)
        use_model = model or cfg.ollama.model
        extra_body = None
    elif backend == "cerebras":
        # CerebrasConfig only carries ``api_key``; the model lives at
        # ``cfg.inference.model`` and Cerebras exposes an OpenAI-compatible
        # endpoint at api.cerebras.ai/v1 (#294).
        client = OpenAI(api_key=cfg.cerebras_api_key, base_url="https://api.cerebras.ai/v1")
        use_model = model or cfg.inference.model
        extra_body = None
    else:
        raise ValueError(
            f"Query synthesis does not support backend '{backend}'. "
            "Use local, ollama, openai, or cerebras in vstash.toml."
        )

    kwargs: dict = {
        "model": use_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_completion_tokens": _MAX_COMPLETION_TOKENS,
    }
    if extra_body:
        kwargs["extra_body"] = extra_body

    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------- #
# Prompt + parsing                                                        #
# ---------------------------------------------------------------------- #


def _build_prompt(text: str, n: int) -> str:
    truncated = text[:_MAX_PASSAGE_CHARS]
    return _PROMPT_TEMPLATE.format(n=n, text=truncated)


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _parse_queries(raw: str, n: int) -> list[str]:
    """Extract up to ``n`` query strings from an LLM response.

    The model is asked to return a JSON array; real-world responses
    often wrap it in markdown fences or add a preamble. We try a
    strict JSON parse first, then a regex-extracted JSON array
    fallback. On any failure we return [].
    """
    raw = raw.strip()
    # Strip markdown fences if present.
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    queries: list[str] = []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = _JSON_ARRAY_RE.search(raw)
        if match is None:
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

    if not isinstance(data, list):
        return []

    for item in data:
        if isinstance(item, str) and item.strip():
            queries.append(item.strip())
        if len(queries) >= n:
            break
    return queries


# ---------------------------------------------------------------------- #
# JSONL cache                                                             #
# ---------------------------------------------------------------------- #


def _cache_key(chunk_id: int, prompt_hash_: str, model: str) -> str:
    return f"{chunk_id}|{prompt_hash_}|{model}"


def _load_cache(cache_path: Path | None) -> dict[str, list[str]]:
    if cache_path is None or not cache_path.exists():
        return {}
    out: dict[str, list[str]] = {}
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk_id = record.get("chunk_id")
            prompt_hash_ = record.get("prompt_hash")
            model = record.get("model")
            queries = record.get("queries")
            if (
                not isinstance(chunk_id, int)
                or not isinstance(prompt_hash_, str)
                or not isinstance(model, str)
                or not isinstance(queries, list)
            ):
                continue
            out[_cache_key(chunk_id, prompt_hash_, model)] = [
                q for q in queries if isinstance(q, str)
            ]
    return out


def _append_cache(
    cache_path: Path | None,
    chunk_id: int,
    prompt_hash_: str,
    model: str,
    queries: list[str],
) -> None:
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "chunk_id": chunk_id,
        "prompt_hash": prompt_hash_,
        "model": model,
        "queries": queries,
    }
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------- #
# Public entry point                                                      #
# ---------------------------------------------------------------------- #


def synthesize_queries(
    chunks: Iterable[dict],
    cfg: VstashConfig,
    n_per_chunk: int = 2,
    cache_path: str | Path | None = None,
    model: str | None = None,
    on_progress: callable | None = None,
) -> dict[int, list[str]]:
    """Generate short realistic queries for each chunk via local LLM.

    Args:
        chunks: Iterable of dicts, each with at least ``id`` (int) and
            ``text`` (str). Extra keys are ignored.
        cfg: Vstash config. Uses ``cfg.inference.backend`` + the
            corresponding backend section.
        n_per_chunk: Target number of queries per chunk.
        cache_path: Optional JSONL file. When set, results are appended
            line-by-line and reused on re-runs. Keys are
            (chunk_id, prompt_hash, model) so changing the prompt or
            the model invalidates cached entries automatically.
        model: Optional model name override (e.g., "qwen2.5:7b-instruct"
            for Ollama). Defaults to whatever the backend's config says.
        on_progress: Optional callback ``(done, total) -> None`` for
            progress reporting. Called once per chunk after it is
            handled (cache hit or LLM miss).

    Returns:
        Mapping ``{chunk_id: [query, ...]}`` with up to ``n_per_chunk``
        queries per chunk. Chunks for which the LLM failed or returned
        unparseable output are simply absent from the map (skip-and-
        continue semantics).
    """
    chunk_list = list(chunks)
    cache_path_obj = Path(cache_path).expanduser() if cache_path else None
    cache = _load_cache(cache_path_obj)

    resolved_model = model
    if resolved_model is None:
        backend = cfg.inference.backend.lower()
        if backend == "local":
            from .chat import _resolve_local_config

            resolved_cfg = _resolve_local_config(cfg)
            backend = resolved_cfg.inference.backend.lower()
            if backend == "openai":
                resolved_model = resolved_cfg.openai.model
            elif backend == "ollama":
                resolved_model = resolved_cfg.ollama.model
            elif backend == "cerebras":
                # CerebrasConfig only carries api_key; model lives at
                # cfg.inference.model (#294).
                resolved_model = resolved_cfg.inference.model
        elif backend == "openai":
            resolved_model = cfg.openai.model
        elif backend == "ollama":
            resolved_model = cfg.ollama.model
        elif backend == "cerebras":
            resolved_model = cfg.inference.model
    if resolved_model is None:
        resolved_model = "unknown"

    out: dict[int, list[str]] = {}
    total = len(chunk_list)
    t0 = time.perf_counter()
    llm_calls = 0
    cache_hits = 0
    parse_failures = 0

    for i, chunk in enumerate(chunk_list, start=1):
        chunk_id = chunk["id"]
        text = chunk["text"]
        prompt = _build_prompt(text, n_per_chunk)
        prompt_hash_ = _prompt_hash(prompt)
        key = _cache_key(chunk_id, prompt_hash_, resolved_model)

        if key in cache:
            out[chunk_id] = cache[key]
            cache_hits += 1
        else:
            try:
                raw = _llm_complete(cfg, prompt, model=resolved_model)
            except Exception as exc:
                logger.warning("LLM call failed for chunk %d: %s", chunk_id, exc)
                if on_progress:
                    on_progress(i, total)
                continue
            queries = _parse_queries(raw, n_per_chunk)
            llm_calls += 1
            if not queries:
                parse_failures += 1
                logger.debug("Unparseable LLM output for chunk %d: %r", chunk_id, raw[:200])
                if on_progress:
                    on_progress(i, total)
                continue
            out[chunk_id] = queries
            _append_cache(cache_path_obj, chunk_id, prompt_hash_, resolved_model, queries)

        if on_progress:
            on_progress(i, total)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Query synthesis: %d chunks -> %d with queries "
        "(%d cache hits, %d LLM calls, %d parse failures, %.1fs)",
        total,
        len(out),
        cache_hits,
        llm_calls,
        parse_failures,
        elapsed,
    )
    return out
