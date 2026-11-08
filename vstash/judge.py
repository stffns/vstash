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
# Worded exactly as the fact-checking adapters were trained (destila's LLM-AggreFact recipe), so a
# ``verify_adapter`` reads the question it learnt; the base model reads it just as well.
SUPPORTED = (
    "Is every piece of information in the CLAIM supported by the DOCUMENT? Answer yes only if the "
    "document states or directly implies all of it; answer no if any part is contradicted, missing, "
    "or unverifiable from the document alone."
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
    verify_adapter: str | None = None

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
        # One forward pass at a time: the verify adapter is switched on and off on the shared
        # model, so a concurrent gate read must not run while it is on (web / MCP threads).
        self._infer_lock = threading.Lock()
        self._model = AutoModel.from_pretrained(
            self.model_name, trust_remote_code=True, dtype=dtype
        ).to(dev)
        self._lora: list = []
        if self.verify_adapter:
            self._attach(self.verify_adapter)
        self._model.eval()

    def _attach(self, adapter: str) -> None:
        """Load a LoRA fact-checking adapter, switched on only while claims are verified.

        It must go in before the first ``system_one`` call: d1 fuses q/k/v (and the feed-forward
        gates) into single matmuls the first time it builds its engine, and only plain ``Linear``
        layers are fused, so an adapter trained on the unfused layers has to be injected first.
        """
        import json
        from pathlib import Path

        try:
            from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
            from peft.tuners.tuners_utils import BaseTunerLayer
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError("verify_adapter needs peft: pip install 'vstash[judge]'") from exc
        root = Path(adapter).expanduser()
        cfg = LoraConfig(**json.loads((root / "lora.json").read_text()))
        # Building the LoRA layers runs CPU init kernels on torch's OpenMP pool. snapvec (imported by
        # vstash) bundles its own libomp, and two OpenMP runtimes in one process segfault there on
        # macOS; one thread keeps the init off the pool. Inference afterwards is unaffected.
        threads = self._torch.get_num_threads()
        self._torch.set_num_threads(1)
        try:
            inject_adapter_in_model(cfg, self._model)
            set_peft_model_state_dict(self._model, load_file(str(root / "adapter.safetensors")))
        finally:
            self._torch.set_num_threads(threads)
        self._lora = [m for m in self._model.modules() if isinstance(m, BaseTunerLayer)]
        self._model.to(next(self._model.parameters()).device)
        self._use_adapter(False)

    def _use_adapter(self, on: bool) -> None:
        for m in self._lora:
            m.enable_adapters(on)

    def _noul(self, requests: list[tuple[str, dict]], *, adapter: bool = False) -> list[float]:
        with self._infer_lock, self._torch.inference_mode():
            if adapter and self._lora:
                self._use_adapter(True)
                try:
                    out = self._model.system_one_batch(requests)
                finally:
                    self._use_adapter(False)
            else:
                out = self._model.system_one_batch(requests)
        return [float(o["answers"]["q"]["noul"]) for o in out]

    def answerable(self, query: str, context: str) -> float:
        state = f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"
        return self._noul([(state, {"q": {"type": "noul", "instructions": ANSWERABLE}})])[0]

    def supported(self, claims: list[str], context: str) -> list[float]:
        if not claims:
            return []
        q = {"q": {"type": "noul", "instructions": SUPPORTED}}
        reqs = [(f"DOCUMENT:\n{context}\n\nCLAIM:\n{c}", q) for c in claims]
        return self._noul(reqs, adapter=True)


_cache: dict[tuple[str, str, str | None], Judge] = {}
_lock = threading.Lock()


def get_judge(cfg: JudgeConfig) -> Judge:
    """One loaded judge per (model, device) for the life of the process: loading costs seconds."""
    key = (cfg.model, cfg.device, cfg.verify_adapter)
    with _lock:
        if key not in _cache:
            _cache[key] = D1Judge(cfg.model, cfg.device, cfg.verify_adapter)
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
