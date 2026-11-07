"""vstash.judge -- a local decision model that checks ``ask`` before and after the LLM.

Retrieval says which chunks are *near* a question; it cannot say whether they *answer* it, and the LLM
that writes the reply cannot be trusted to say when it is guessing. A small decision model (Liquid AI's
open d1, ~2B parameters, one forward pass per question, calibrated probabilities) closes both gaps:

- **Gate** (before): P(the retrieved context holds enough to answer). Below
  ``[judge] gate_threshold`` vstash replies that memory does not hold the answer and skips the LLM call.
- **Verify** (after): the reply is split into claims and each gets P(supported by the context), so a
  caller can see which sentences the memory backs and which the LLM added.

Everything runs locally (PyTorch on Apple Silicon / CUDA / CPU). Optional: ``pip install vstash[judge]``
and ``[judge] enabled = true``. Without it ``ask`` behaves exactly as before.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from .models import ClaimCheck, Judgment

if TYPE_CHECKING:
    from .config import JudgeConfig
    from .models import SearchResult

ANSWERABLE = (
    "Does the CONTEXT contain the information needed to answer the QUESTION? Answer yes only if a "
    "careful reader could answer it from the context alone, without outside knowledge or guessing."
)
SUPPORTED = (
    "Is every piece of information in the CLAIM supported by the CONTEXT? Answer yes only if the "
    "context states or directly implies all of it; answer no if any part is contradicted, missing, "
    "or unverifiable from the context alone."
)
NOT_IN_MEMORY = (
    "I couldn't find this in your memory: the retrieved documents don't seem to contain the answer "
    "(judge confidence {p:.0%}). Try rephrasing, or add the relevant documents with `vstash add`."
)

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ0-9\"'(\[])")
_MIN_CLAIM_CHARS = 25


class Judge(Protocol):
    """What ``ask`` needs from a decision model: two calibrated yes/no reads."""

    def answerable(self, query: str, context: str) -> float: ...

    def supported(self, claims: list[str], context: str) -> list[float]: ...


def context_text(chunks: list[SearchResult]) -> str:
    """The retrieved chunks as the judge reads them: numbered, with their source titles."""
    return "\n\n".join(f"[{i}] {c.title}\n{c.text}" for i, c in enumerate(chunks, 1))


def split_claims(answer: str) -> list[str]:
    """The reply's checkable statements: sentences and list items, without markdown furniture.

    Headings, code fences and very short fragments ("Sure.", "In short:") are not claims about the
    documents, so they are left out rather than judged.
    """
    text = re.sub(r"```.*?```", " ", answer, flags=re.S)
    claims: list[str] = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(#+|[-*+]|\d+[.)])\s*", "", line).strip()
        line = re.sub(r"\s*\[\d+\]|\*\*|__|`", "", line).strip()
        if not line:
            continue
        for sent in _SENTENCE.split(line):
            sent = sent.strip()
            if len(sent) >= _MIN_CLAIM_CHARS and not sent.endswith(":"):
                claims.append(sent)
    return claims


@dataclass
class D1Judge:
    """Liquid AI's d1 decision models through their own ``system_one_batch`` call."""

    model_name: str = "LiquidAI/d1-2B"
    device: str = "auto"

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "the judge needs PyTorch and transformers: pip install 'vstash[judge]'"
            ) from exc
        dev = self.device
        if dev == "auto":
            dev = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        dtype = torch.float32 if dev == "cpu" else torch.bfloat16
        self._torch = torch
        self._model = (
            AutoModel.from_pretrained(self.model_name, trust_remote_code=True, dtype=dtype)
            .to(dev)
            .eval()
        )

    def _noul(self, requests: list[tuple[str, dict]]) -> list[float]:
        with self._torch.inference_mode():
            out = self._model.system_one_batch(requests)
        return [float(o["answers"]["q"]["noul"]) for o in out]

    def answerable(self, query: str, context: str) -> float:
        state = f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"
        return self._noul([(state, {"q": {"type": "noul", "instructions": ANSWERABLE}})])[0]

    def supported(self, claims: list[str], context: str) -> list[float]:
        if not claims:
            return []
        q = {"q": {"type": "noul", "instructions": SUPPORTED}}
        return self._noul([(f"CONTEXT:\n{context}\n\nCLAIM:\n{c}", q) for c in claims])


_cache: dict[tuple[str, str], Judge] = {}
_lock = threading.Lock()


def get_judge(cfg: JudgeConfig) -> Judge:
    """One loaded judge per (model, device) for the life of the process: loading costs seconds."""
    key = (cfg.model, cfg.device)
    with _lock:
        if key not in _cache:
            _cache[key] = D1Judge(cfg.model, cfg.device)
        return _cache[key]


def gate(judge: Judge, query: str, chunks: list[SearchResult], cfg: JudgeConfig) -> Judgment:
    """The pre-LLM read: is the answer in the retrieved context at all?"""
    p = judge.answerable(query, context_text(chunks)) if chunks else 0.0
    return Judgment(answerable=p, gated=cfg.gate and p < cfg.gate_threshold, model=cfg.model)


def verify(
    judge: Judge, judgment: Judgment, answer: str, chunks: list[SearchResult], cfg: JudgeConfig
) -> Judgment:
    """The post-LLM read: each claim of ``answer`` against the same context the LLM saw."""
    claims = split_claims(answer)[: cfg.max_claims]
    ps = judge.supported(claims, context_text(chunks)) if chunks else [0.0] * len(claims)
    checks = [
        ClaimCheck(text=c, supported=p, flagged=p < cfg.claim_threshold)
        for c, p in zip(claims, ps, strict=True)
    ]
    return judgment.model_copy(update={"claims": checks})
