"""
chat.py — Inference backend abstraction.

Three backends, same interface:
  cerebras → ~2000 tok/s, sends chunks to Cerebras API
  ollama   → fully local, nothing leaves your machine
  openai   → OpenAI API or any compatible endpoint

The prompt is assembled here: system context + history + retrieved chunks + user query.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable, Generator
from typing import ParamSpec, TypeVar

from .config import VstashConfig
from .models import AskResult, SearchResult

T = TypeVar("T")
P = ParamSpec("P")

logger = logging.getLogger(__name__)

# Retry configuration
_MAX_RETRIES = 3
_BASE_DELAY = 1.0  # seconds
_MAX_DELAY = 10.0  # seconds

# Transient error strings that warrant a retry (case-insensitive substring match)
_RETRYABLE_PATTERNS = (
    "rate limit",
    "429",
    "503",
    "502",
    "timeout",
    "timed out",
    "connection reset",
    "connection refused",
    "temporarily unavailable",
    "server error",
    "overloaded",
)

# Exception types that are always retryable regardless of message content
_RETRYABLE_TYPES = (TimeoutError,)


def _is_retryable(exc: Exception) -> bool:
    """Check if an exception represents a transient failure worth retrying."""
    if isinstance(exc, _RETRYABLE_TYPES):
        return True
    msg = str(exc).lower()
    return any(p in msg for p in _RETRYABLE_PATTERNS)


def _retry_call(fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """Retry a function with exponential backoff on transient errors."""
    for attempt in range(_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except (ConnectionError, TimeoutError, OSError) as exc:
            if attempt < _MAX_RETRIES - 1 and _is_retryable(exc):
                delay = min(_BASE_DELAY * (2**attempt), _MAX_DELAY)
                logger.warning(
                    "Retry %d/%d for %s after %.1fs: %s",
                    attempt + 1,
                    _MAX_RETRIES,
                    fn.__name__,
                    delay,
                    exc,
                )
                print(
                    f"\r⟳ Retry {attempt + 1}/{_MAX_RETRIES} in {delay:.0f}s...",
                    end="",
                    flush=True,
                    file=sys.stderr,
                )
                time.sleep(delay)
            else:
                raise
    # Unreachable: loop always returns or raises
    raise RuntimeError("unreachable")  # pragma: no cover


SYSTEM_PROMPT = """You are a precise document assistant. Answer questions based strictly on the provided context.

Rules:
- Answer only from the context. Do not invent information.
- If the context doesn't contain the answer, say so clearly.
- Always cite which source document each fact comes from (use the document title shown in brackets).
- If the user's question mentions a specific document but the answer comes from a different one, explicitly note the correction (e.g., "That information is not in [X] but in [Y]").
- For code questions, provide working code examples from the context."""


def _extract_reasoning(message: object) -> str | None:
    """Pull the reasoning trace from a chat-completions ``message`` object.

    Different OpenAI-compatible servers expose it under different field
    names: Cerebras and the original anthropic-flavoured proposals use
    ``message.reasoning``; vLLM, DeepSeek, Together, xAI Grok and OpenAI
    o1/o3 Chat Completions all use ``message.reasoning_content``.  We
    accept either and collapse empty strings to ``None`` so a present-
    but-empty channel doesn't masquerade as a real trace.
    """
    candidate = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
    return candidate or None


def _normalize_usage(raw: object) -> dict[str, int] | None:
    """Coerce an SDK ``usage`` object into ``{prompt, completion, total}_tokens``.

    Cerebras / OpenAI return a Pydantic-ish object; some compat servers
    return a plain dict; older SDKs may omit it entirely. We accept any
    of those shapes and keep only int fields we recognise.
    """
    if raw is None:
        return None
    out: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if isinstance(raw, dict):
            value = raw.get(key)
        else:
            value = getattr(raw, key, None)
        if isinstance(value, int):
            out[key] = value
    return out or None


def _build_prompt(query: str, chunks: list[SearchResult]) -> str:
    """Build user prompt from query and retrieved context chunks.

    Args:
        query: The user's question.
        chunks: Retrieved search results with context.

    Returns:
        Formatted prompt string combining context and question.
    """
    if not chunks:
        return f"No relevant context found in memory.\n\nQuestion: {query}"

    context_parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        source = f"[{chunk.title}] (from: {chunk.path})"
        context_parts.append(f"--- Context {i} {source} ---\n{chunk.text}")

    context = "\n\n".join(context_parts)
    return f"{context}\n\n---\nQuestion: {query}"


def _build_messages(
    query: str,
    chunks: list[SearchResult],
    history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Build the full message list for the LLM.

    Args:
        query: The user's current question.
        chunks: Retrieved search results with context.
        history: Previous conversation turns as role/content dicts.

    Returns:
        List of message dicts with system, history, and user messages.
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        messages.extend(history)

    prompt = _build_prompt(query, chunks)
    messages.append({"role": "user", "content": prompt})

    return messages


# ------------------------------------------------------------------ #
# Cerebras                                                            #
# ------------------------------------------------------------------ #


def _ask_cerebras(
    query: str,
    chunks: list[SearchResult],
    cfg: VstashConfig,
    history: list[dict[str, str]] | None = None,
) -> AskResult:
    """Send query to Cerebras API and return the response.

    Args:
        query: The user's question.
        chunks: Retrieved context chunks.
        cfg: Vex configuration with Cerebras settings.
        history: Prior conversation turns.

    Returns:
        AskResult with content + (when present) reasoning + usage.

    Raises:
        ValueError: If API key is missing.
        ConnectionError: If the API request fails.
    """
    try:
        from cerebras.cloud.sdk import Cerebras
    except ImportError as exc:
        raise ImportError(
            "Cerebras backend requires the cerebras SDK. "
            "Install it with: pip install vstash[cerebras]"
        ) from exc

    api_key = cfg.cerebras_api_key
    if not api_key:
        raise ValueError(
            "Cerebras API key not found. Set CEREBRAS_API_KEY env var "
            "or add it to vstash.toml under [cerebras] api_key."
        )

    client = Cerebras(api_key=api_key)
    messages = _build_messages(query, chunks, history)
    model = cfg.inference.model

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=2048,
            temperature=0.2,
        )
        message = response.choices[0].message
        # gpt-oss-* surfaces hidden CoT under ``reasoning``; future
        # Cerebras builds may follow the broader ``reasoning_content``
        # convention -- accept either.
        return AskResult(
            content=message.content or "",
            reasoning=_extract_reasoning(message),
            usage=_normalize_usage(getattr(response, "usage", None)),
            backend="cerebras",
            model=model,
        )
    except Exception as exc:
        raise ConnectionError(f"Cerebras API error: {exc}") from exc


def _stream_cerebras(
    query: str,
    chunks: list[SearchResult],
    cfg: VstashConfig,
    history: list[dict[str, str]] | None = None,
) -> Generator[str, None, None]:
    """Stream tokens from Cerebras API.

    Args:
        query: The user's question.
        chunks: Retrieved context chunks.
        cfg: Vex configuration with Cerebras settings.
        history: Prior conversation turns.

    Yields:
        Token strings as they arrive.

    Raises:
        ValueError: If API key is missing.
        ConnectionError: If the API request fails.
    """
    try:
        from cerebras.cloud.sdk import Cerebras
    except ImportError as exc:
        raise ImportError(
            "Cerebras backend requires the cerebras SDK. "
            "Install it with: pip install vstash[cerebras]"
        ) from exc

    api_key = cfg.cerebras_api_key
    if not api_key:
        raise ValueError("Cerebras API key not found.")

    client = Cerebras(api_key=api_key)
    messages = _build_messages(query, chunks, history)

    try:
        stream = client.chat.completions.create(
            model=cfg.inference.model,
            messages=messages,
            max_tokens=2048,
            temperature=0.2,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as exc:
        raise ConnectionError(f"Cerebras streaming error: {exc}") from exc


# ------------------------------------------------------------------ #
# Ollama                                                              #
# ------------------------------------------------------------------ #


def _ask_ollama(
    query: str,
    chunks: list[SearchResult],
    cfg: VstashConfig,
    history: list[dict[str, str]] | None = None,
) -> AskResult:
    """Send query to Ollama local server and return the response.

    Args:
        query: The user's question.
        chunks: Retrieved context chunks.
        cfg: Vex configuration with Ollama settings.
        history: Prior conversation turns.

    Returns:
        AskResult with content + (qwen3 thinking-mode) reasoning + usage.

    Raises:
        ConnectionError: If Ollama server is unreachable.
    """
    try:
        import ollama
    except ImportError as exc:
        raise ImportError(
            "Ollama backend requires the ollama package. "
            "Install it with: pip install vstash[ollama]"
        ) from exc

    client = ollama.Client(host=cfg.ollama.host)
    messages = _build_messages(query, chunks, history)
    model = cfg.ollama.model

    try:
        response = client.chat(
            model=model,
            messages=messages,
            options={"temperature": 0.2},
        )
        # The ollama python client returns dict-shaped responses on
        # current versions, but older / pinned releases occasionally
        # return a Pydantic-ish object.  Stay permissive so an SDK bump
        # doesn't crash the call site.
        if isinstance(response, dict):
            message = response.get("message")
            prompt = response.get("prompt_eval_count")
            completion = response.get("eval_count")
        else:
            message = getattr(response, "message", None)
            prompt = getattr(response, "prompt_eval_count", None)
            completion = getattr(response, "eval_count", None)
        # qwen3 thinking-enabled models surface CoT here; absent otherwise.
        if isinstance(message, dict):
            content = message.get("content") or ""
            reasoning = message.get("thinking") or None
        else:
            content = getattr(message, "content", "") or ""
            reasoning = getattr(message, "thinking", None) or None
        usage: dict[str, int] | None = None
        if isinstance(prompt, int) or isinstance(completion, int):
            p = prompt if isinstance(prompt, int) else 0
            c = completion if isinstance(completion, int) else 0
            usage = {
                "prompt_tokens": p,
                "completion_tokens": c,
                "total_tokens": p + c,
            }
        return AskResult(
            content=content,
            reasoning=reasoning,
            usage=usage,
            backend="ollama",
            model=model,
        )
    except Exception as exc:
        raise ConnectionError(f"Ollama error: {exc}") from exc


def _stream_ollama(
    query: str,
    chunks: list[SearchResult],
    cfg: VstashConfig,
    history: list[dict[str, str]] | None = None,
) -> Generator[str, None, None]:
    """Stream tokens from Ollama local server.

    Args:
        query: The user's question.
        chunks: Retrieved context chunks.
        cfg: Vex configuration with Ollama settings.
        history: Prior conversation turns.

    Yields:
        Token strings as they arrive.

    Raises:
        ConnectionError: If Ollama server is unreachable.
    """
    try:
        import ollama
    except ImportError as exc:
        raise ImportError(
            "Ollama backend requires the ollama package. "
            "Install it with: pip install vstash[ollama]"
        ) from exc

    client = ollama.Client(host=cfg.ollama.host)
    messages = _build_messages(query, chunks, history)

    try:
        for chunk in client.chat(
            model=cfg.ollama.model,
            messages=messages,
            stream=True,
            options={"temperature": 0.2},
        ):
            content = chunk.get("message", {}).get("content", "")  # type: ignore[union-attr]
            if content:
                yield content
    except Exception as exc:
        raise ConnectionError(f"Ollama streaming error: {exc}") from exc


# ------------------------------------------------------------------ #
# OpenAI                                                              #
# ------------------------------------------------------------------ #


def _ask_openai(
    query: str,
    chunks: list[SearchResult],
    cfg: VstashConfig,
    history: list[dict[str, str]] | None = None,
) -> AskResult:
    """Send query to OpenAI API (or compatible endpoint) and return the response.

    Args:
        query: The user's question.
        chunks: Retrieved context chunks.
        cfg: Vex configuration with OpenAI settings.
        history: Prior conversation turns.

    Returns:
        AskResult. ``reasoning`` is populated only when the underlying
        endpoint surfaces a non-standard reasoning channel (some
        OpenAI-compatible servers do; vanilla GPT-4 does not).

    Raises:
        ValueError: If API key is missing.
        ConnectionError: If the API request fails.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "OpenAI backend requires the openai package. "
            "Install it with: pip install vstash[openai]"
        ) from exc

    api_key = cfg.openai_api_key
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Set OPENAI_API_KEY env var "
            "or add it to vstash.toml under [openai] api_key."
        )

    client = OpenAI(api_key=api_key, base_url=cfg.openai.base_url)
    messages = _build_messages(query, chunks, history)
    model = cfg.openai.model

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            max_completion_tokens=2048,
            temperature=0.2,
            extra_body=cfg.openai.extra_body,
        )
        message = response.choices[0].message
        # OpenAI o1/o3 Chat Completions, vLLM, DeepSeek, Together, and
        # xAI Grok all surface reasoning under ``reasoning_content``;
        # accept ``reasoning`` too for older / Cerebras-aligned servers.
        return AskResult(
            content=message.content or "",
            reasoning=_extract_reasoning(message),
            usage=_normalize_usage(getattr(response, "usage", None)),
            backend="openai",
            model=model,
        )
    except Exception as exc:
        raise ConnectionError(f"OpenAI API error: {exc}") from exc


def _stream_openai(
    query: str,
    chunks: list[SearchResult],
    cfg: VstashConfig,
    history: list[dict[str, str]] | None = None,
) -> Generator[str, None, None]:
    """Stream tokens from OpenAI API (or compatible endpoint).

    Args:
        query: The user's question.
        chunks: Retrieved context chunks.
        cfg: Vex configuration with OpenAI settings.
        history: Prior conversation turns.

    Yields:
        Token strings as they arrive.

    Raises:
        ValueError: If API key is missing.
        ConnectionError: If the API request fails.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "OpenAI backend requires the openai package. "
            "Install it with: pip install vstash[openai]"
        ) from exc

    api_key = cfg.openai_api_key
    if not api_key:
        raise ValueError("OpenAI API key not found.")

    client = OpenAI(api_key=api_key, base_url=cfg.openai.base_url)
    messages = _build_messages(query, chunks, history)
    model = cfg.openai.model

    try:
        response_stream = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            max_completion_tokens=2048,
            temperature=0.2,
            stream=True,
            extra_body=cfg.openai.extra_body,
        )
        for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as exc:
        raise ConnectionError(f"OpenAI streaming error: {exc}") from exc


# ------------------------------------------------------------------ #
# Public interface                                                    #
# ------------------------------------------------------------------ #

_BACKENDS = {
    "cerebras": (_ask_cerebras, _stream_cerebras),
    "ollama": (_ask_ollama, _stream_ollama),
    "openai": (_ask_openai, _stream_openai),
}

# --- Local auto-detect ---------------------------------------------------

# Endpoints probed in order.  Each entry is (name, url, probe_path, backend).
# "backend" is which _BACKENDS key to delegate to once detected.
_LOCAL_ENDPOINTS: list[tuple[str, str, str, str]] = [
    ("Ollama", "http://localhost:11434", "/api/tags", "ollama"),
    ("LM Studio", "http://localhost:1234", "/v1/models", "openai"),
    ("LM Studio (alt)", "http://localhost:8080", "/v1/models", "openai"),
    ("LocalAI", "http://localhost:8081", "/v1/models", "openai"),
]

# Cache so we only probe once per process.
_local_cache: dict[str, object] | None = None


def _detect_local() -> dict[str, object]:
    """Probe local endpoints and return detection info.

    Returns:
        Dict with keys: backend, name, base_url, model (or None if nothing found).
    """
    global _local_cache  # noqa: PLW0603
    if _local_cache is not None:
        return _local_cache

    import urllib.request

    for name, base_url, probe_path, backend in _LOCAL_ENDPOINTS:
        try:
            req = urllib.request.Request(base_url + probe_path, method="GET")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                if resp.status == 200:
                    import json

                    data = json.loads(resp.read())
                    # Extract first available model name
                    model = None
                    if backend == "ollama":
                        # Ollama: {"models": [{"name": "qwen3.5:4b", ...}]}
                        models = data.get("models", [])
                        if models:
                            model = models[0].get("name")
                    else:
                        # OpenAI-compatible: {"data": [{"id": "model-name"}]}
                        items = data.get("data", [])
                        if items:
                            model = items[0].get("id")

                    if model:
                        _local_cache = {
                            "backend": backend,
                            "name": name,
                            "base_url": base_url,
                            "model": model,
                        }
                        logger.info(
                            "Local LLM detected: %s at %s (model: %s)",
                            name,
                            base_url,
                            model,
                        )
                        return _local_cache
        except Exception:
            continue

    _local_cache = {"backend": None, "name": None, "base_url": None, "model": None}
    return _local_cache


def _resolve_local_config(cfg: VstashConfig) -> VstashConfig:
    """Detect a local LLM server and return a config patched to use it."""
    detected = _detect_local()
    backend = detected["backend"]

    if backend is None:
        raise ConnectionError(
            "No local LLM server found. Start Ollama, LM Studio, or any "
            "OpenAI-compatible server, or set inference.backend explicitly in vstash.toml."
        )

    logger.info("Using local backend: %s (%s)", detected["name"], detected["model"])

    if backend == "ollama":
        return cfg.model_copy(
            update={
                "inference": cfg.inference.model_copy(
                    update={"backend": "ollama", "model": str(detected["model"])}
                ),
                "ollama": cfg.ollama.model_copy(
                    update={"model": str(detected["model"]), "host": str(detected["base_url"])}
                ),
            }
        )
    else:
        # OpenAI-compatible (LM Studio, LocalAI, etc.)
        return cfg.model_copy(
            update={
                "inference": cfg.inference.model_copy(
                    update={"backend": "openai", "model": str(detected["model"])}
                ),
                "openai": cfg.openai.model_copy(
                    update={
                        "model": str(detected["model"]),
                        "base_url": str(detected["base_url"]) + "/v1",
                        "api_key": "not-needed",
                    }
                ),
            }
        )


def ask_full(
    query: str,
    chunks: list[SearchResult],
    cfg: VstashConfig,
    history: list[dict[str, str]] | None = None,
) -> AskResult:
    """Like :func:`ask`, but returns the full normalized result.

    Surfaces the reasoning channel and token usage that ``ask()`` drops.
    Driven by Merken Phase 2 distillation (see issue #303); generally
    useful for trace-aware logging or evaluation. ``ask()`` is now a
    thin wrapper that returns ``ask_full(...).content``.

    Args:
        query: The user's question.
        chunks: Retrieved context chunks.
        cfg: Vex configuration.
        history: Prior conversation turns for multi-turn chat.

    Returns:
        AskResult with content + (when surfaced) reasoning + usage +
        resolved backend / model.

    Raises:
        ValueError: If the configured backend is unknown.
    """
    backend = cfg.inference.backend.lower()
    if backend == "local":
        cfg = _resolve_local_config(cfg)
        backend = cfg.inference.backend.lower()

    backend_funcs = _BACKENDS.get(backend)

    if not backend_funcs:
        raise ValueError(
            f"Unknown inference backend: '{backend}'. "
            "Use 'local', 'cerebras', 'ollama', or 'openai' in vstash.toml."
        )

    ask_fn, _ = backend_funcs
    return _retry_call(ask_fn, query, chunks, cfg, history)


def ask(
    query: str,
    chunks: list[SearchResult],
    cfg: VstashConfig,
    history: list[dict[str, str]] | None = None,
) -> str:
    """Send query + context chunks to the configured inference backend.

    Args:
        query: The user's question.
        chunks: Retrieved context chunks.
        cfg: Vex configuration.
        history: Prior conversation turns for multi-turn chat.

    Returns:
        Model response text. Use :func:`ask_full` if you also need the
        reasoning trace and usage counts.

    Raises:
        ValueError: If the configured backend is unknown.
    """
    return ask_full(query, chunks, cfg, history).content


def stream(
    query: str,
    chunks: list[SearchResult],
    cfg: VstashConfig,
    history: list[dict[str, str]] | None = None,
) -> Generator[str, None, None]:
    """Stream tokens from the configured backend.

    Args:
        query: The user's question.
        chunks: Retrieved context chunks.
        cfg: Vex configuration.
        history: Prior conversation turns for multi-turn chat.

    Yields:
        Token strings as they arrive.

    Raises:
        ValueError: If the configured backend is unknown.
    """
    backend = cfg.inference.backend.lower()
    if backend == "local":
        cfg = _resolve_local_config(cfg)
        backend = cfg.inference.backend.lower()

    backend_funcs = _BACKENDS.get(backend)

    if not backend_funcs:
        raise ValueError(
            f"Unknown inference backend: '{backend}'. "
            "Use 'local', 'cerebras', 'ollama', or 'openai' in vstash.toml."
        )

    _, stream_fn = backend_funcs
    # Retry the initial connection — once the generator starts yielding,
    # a mid-stream failure is not retried (partial output already sent).
    gen = _retry_call(stream_fn, query, chunks, cfg, history)
    yield from gen
