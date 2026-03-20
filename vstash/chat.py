"""
chat.py — Inference backend abstraction.

Three backends, same interface:
  cerebras → ~2000 tok/s, sends chunks to Cerebras API
  ollama   → fully local, nothing leaves your machine
  openai   → OpenAI API or any compatible endpoint

The prompt is assembled here: system context + history + retrieved chunks + user query.
"""

from __future__ import annotations

from collections.abc import Generator

from .config import VstashConfig
from .models import SearchResult

SYSTEM_PROMPT = """You are a precise document assistant. Answer questions based strictly on the provided context.

Rules:
- Answer only from the context. Do not invent information.
- If the context doesn't contain the answer, say so clearly.
- Be concise. Cite the source document when relevant.
- For code questions, provide working code examples from the context."""


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
        source = f"[{chunk.title}]"
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
) -> str:
    """Send query to Cerebras API and return the response.

    Args:
        query: The user's question.
        chunks: Retrieved context chunks.
        cfg: Vex configuration with Cerebras settings.
        history: Prior conversation turns.

    Returns:
        Model response text.

    Raises:
        ValueError: If API key is missing.
        ConnectionError: If the API request fails.
    """
    from cerebras.cloud.sdk import Cerebras

    api_key = cfg.cerebras_api_key
    if not api_key:
        raise ValueError(
            "Cerebras API key not found. Set CEREBRAS_API_KEY env var "
            "or add it to vstash.toml under [cerebras] api_key."
        )

    client = Cerebras(api_key=api_key)
    messages = _build_messages(query, chunks, history)

    try:
        response = client.chat.completions.create(
            model=cfg.inference.model,
            messages=messages,
            max_tokens=2048,
            temperature=0.2,
        )
        return response.choices[0].message.content  # type: ignore[return-value]
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
    from cerebras.cloud.sdk import Cerebras

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
) -> str:
    """Send query to Ollama local server and return the response.

    Args:
        query: The user's question.
        chunks: Retrieved context chunks.
        cfg: Vex configuration with Ollama settings.
        history: Prior conversation turns.

    Returns:
        Model response text.

    Raises:
        ConnectionError: If Ollama server is unreachable.
    """
    import ollama

    client = ollama.Client(host=cfg.ollama.host)
    messages = _build_messages(query, chunks, history)

    try:
        response = client.chat(
            model=cfg.ollama.model,
            messages=messages,
            options={"temperature": 0.2},
        )
        return response["message"]["content"]  # type: ignore[index]
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
    import ollama

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
) -> str:
    """Send query to OpenAI API (or compatible endpoint) and return the response.

    Args:
        query: The user's question.
        chunks: Retrieved context chunks.
        cfg: Vex configuration with OpenAI settings.
        history: Prior conversation turns.

    Returns:
        Model response text.

    Raises:
        ValueError: If API key is missing.
        ConnectionError: If the API request fails.
    """
    from openai import OpenAI

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
        )
        return response.choices[0].message.content or ""
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
    from openai import OpenAI

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
        Model response text.

    Raises:
        ValueError: If the configured backend is unknown.
    """
    backend = cfg.inference.backend.lower()
    backend_funcs = _BACKENDS.get(backend)

    if not backend_funcs:
        raise ValueError(
            f"Unknown inference backend: '{backend}'. "
            "Use 'cerebras', 'ollama', or 'openai' in vstash.toml."
        )

    ask_fn, _ = backend_funcs
    return ask_fn(query, chunks, cfg, history)


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
    backend_funcs = _BACKENDS.get(backend)

    if not backend_funcs:
        raise ValueError(
            f"Unknown inference backend: '{backend}'. "
            "Use 'cerebras', 'ollama', or 'openai' in vstash.toml."
        )

    _, stream_fn = backend_funcs
    yield from stream_fn(query, chunks, cfg, history)
