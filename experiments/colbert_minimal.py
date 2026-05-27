"""Minimal ColBERTv2 inference on raw HuggingFace transformers.

Fallback for when pylate / sentence-transformers / fast-plaid dependency
churn breaks Colab installs.  Loads ``colbert-ir/colbertv2.0`` as a plain
BertModel + linear projection (the architecture documented in Khattab
& Zaharia 2022) and computes MaxSim via batched matmul.

Used by ``longmemeval_colbert.py`` / ``beir_colbert.py`` when
``--engine minimal`` is passed; pylate stays the default.

Why this exists: pylate's transitive dep ``fast-plaid`` hard-pins
``torch==2.9.0``.  On Colab T4 the preinstalled torch is 2.10.0+cu128
matched to torchvision 0.25.0+cu128, and forcing the downgrade
breaks the entire transformers import chain.  The pure-HF path here
needs only ``torch`` + ``transformers`` (both pre-shipped on Colab),
so it works regardless of what pylate / sentence-transformers do
to the env.

Reference: ColBERTv2: Effective and Efficient Retrieval via
Lightweight Late Interaction (Santhanam et al., NAACL 2022).
"""

from __future__ import annotations

import torch
from huggingface_hub import hf_hub_download


# Special tokens hardcoded at the values used in the published
# colbert-ir/colbertv2.0 checkpoint -- we mark queries with [unused0]
# (id 1) and docs with [unused1] (id 2) immediately after [CLS].
_QUERY_MARKER_ID = 1
_DOC_MARKER_ID = 2


class MinimalColBERT:
    """Tiny ColBERTv2 inference wrapper.

    Loads the published checkpoint as a BertModel, applies the 768->128
    linear projection from ``linear.weight``, L2-normalises per token,
    and exposes ``encode_queries`` / ``encode_documents`` plus
    ``maxsim_scores`` for retrieval.
    """

    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        device: str = "cuda",
        query_max_len: int = 32,
        doc_max_len: int = 220,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device(
            device if torch.cuda.is_available() or device == "cpu" else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # AutoModel returns BertModel; the linear projection is stored
        # under the ``linear`` key in the checkpoint but is NOT loaded
        # automatically by AutoModel.  We pull it from safetensors.
        self.bert = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.linear = self._load_linear(model_name).to(self.device).eval()
        self.query_max_len = query_max_len
        self.doc_max_len = doc_max_len

    @staticmethod
    def _load_linear(model_name: str) -> torch.nn.Linear:
        """Pull the 768 -> 128 projection layer from the HF checkpoint."""
        from safetensors.torch import load_file

        path = hf_hub_download(model_name, "model.safetensors")
        state = load_file(path)
        # ColBERTv2 stores it as ``linear.weight`` (no bias).
        if "linear.weight" not in state:
            # Some HF mirrors prefix with model. ; try both layouts.
            for prefix in ("", "model."):
                key = f"{prefix}linear.weight"
                if key in state:
                    weight = state[key]
                    break
            else:
                raise RuntimeError(
                    f"Could not find ColBERT linear projection in {path}. "
                    f"Keys present: {list(state)[:5]}..."
                )
        else:
            weight = state["linear.weight"]
        layer = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        layer.weight.data = weight
        return layer

    @torch.no_grad()
    def _encode(self, texts: list[str], marker_id: int, max_length: int, batch_size: int):
        """Tokenize, prepend the marker token, run BERT + linear + L2 normalize."""
        all_embs = []
        all_masks = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length - 1,  # leave room for marker token
                return_tensors="pt",
            )
            ids = enc["input_ids"].to(self.device)
            attn = enc["attention_mask"].to(self.device)
            # Insert the marker AFTER the [CLS] token (position 1).
            cls = ids[:, :1]
            rest = ids[:, 1:]
            marker = torch.full_like(cls, marker_id)
            ids = torch.cat([cls, marker, rest], dim=1)
            attn = torch.cat([attn[:, :1], torch.ones_like(attn[:, :1]), attn[:, 1:]], dim=1)

            out = self.bert(input_ids=ids, attention_mask=attn).last_hidden_state
            proj = self.linear(out)
            proj = torch.nn.functional.normalize(proj, p=2, dim=2)
            # Zero out padding token vectors so they contribute nothing
            # to the MaxSim sum.
            proj = proj * attn.unsqueeze(-1).float()
            all_embs.append(proj)
            all_masks.append(attn)
        return all_embs, all_masks

    def encode_queries(self, queries: list[str], batch_size: int = 32):
        """Returns one (seq_len, 128) tensor per query."""
        embs, _masks = self._encode(queries, _QUERY_MARKER_ID, self.query_max_len, batch_size)
        return [e[i] for e in embs for i in range(e.shape[0])]

    def encode_documents(self, docs: list[str], batch_size: int = 32):
        """Returns (n_docs, max_seq_len, 128) padded tensor + mask.

        Padding lets us build one big tensor for batched MaxSim instead
        of looping over docs.  Masked positions are zero so they
        contribute nothing to the cosine product.
        """
        embs, masks = self._encode(docs, _DOC_MARKER_ID, self.doc_max_len, batch_size)
        if not embs:
            return torch.empty(0, 0, 128, device=self.device), torch.empty(0, 0, device=self.device)
        # Right-pad each batch to max seq_len in the corpus.
        max_len = max(e.shape[1] for e in embs)
        padded = []
        padded_masks = []
        for e, m in zip(embs, masks):
            pad = max_len - e.shape[1]
            if pad:
                e = torch.nn.functional.pad(e, (0, 0, 0, pad))
                m = torch.nn.functional.pad(m, (0, pad))
            padded.append(e)
            padded_masks.append(m)
        return torch.cat(padded, dim=0), torch.cat(padded_masks, dim=0)

    @staticmethod
    def maxsim_scores(
        query_emb: torch.Tensor,  # (seq_q, 128)
        doc_embs: torch.Tensor,  # (n_docs, seq_d, 128)
    ) -> torch.Tensor:
        """ColBERT MaxSim: for each query token, take max cosine over
        doc tokens; sum across query tokens.

        Returns (n_docs,) score vector.
        """
        # (n_docs, seq_q, seq_d) cosine matrix
        scores = torch.einsum("qh,dkh->dqk", query_emb, doc_embs)
        # Max over doc tokens (dim=2), sum over query tokens (dim=1)
        return scores.max(dim=2).values.sum(dim=1)
