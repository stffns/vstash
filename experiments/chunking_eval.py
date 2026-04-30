"""Chunking Evaluation — Code-Aware vs Naive Chunking

Compares retrieval quality when using:
  1. Code-aware chunking (function/class boundary detection)
  2. Naive fixed-window chunking (ignores code structure)

Uses real code files with known function locations as ground truth.

Usage:
    python -m experiments.chunking_eval [--output results/chunking.json]
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from vstash.ingest import chunk_code
from vstash.embed import embed_query, get_embedding_dim
from vstash.store import VstashStore

MODEL = "BAAI/bge-small-en-v1.5"

# ------------------------------------------------------------------ #
# Synthetic code corpus with known functions                           #
# ------------------------------------------------------------------ #

PYTHON_CODE = '''"""User authentication module."""

import hashlib
import secrets
from datetime import datetime, timedelta


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """Hash a password with PBKDF2-SHA256.

    Args:
        password: The plaintext password to hash.
        salt: Optional salt. Generated if not provided.

    Returns:
        Tuple of (hashed_password, salt).
    """
    if salt is None:
        salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return hashed.hex(), salt


def verify_password(password: str, hashed: str, salt: str) -> bool:
    """Verify a password against its hash."""
    computed, _ = hash_password(password, salt)
    return secrets.compare_digest(computed, hashed)


class TokenManager:
    """Manages JWT-like authentication tokens."""

    def __init__(self, secret: str, expiry_hours: int = 24):
        self.secret = secret
        self.expiry_hours = expiry_hours
        self._revoked: set[str] = set()

    def create_token(self, user_id: str, roles: list[str] | None = None) -> dict:
        """Create a new authentication token.

        Args:
            user_id: The user's unique identifier.
            roles: Optional list of role strings.

        Returns:
            Token dict with payload and expiry.
        """
        now = datetime.utcnow()
        payload = {
            "sub": user_id,
            "roles": roles or [],
            "iat": now.isoformat(),
            "exp": (now + timedelta(hours=self.expiry_hours)).isoformat(),
            "jti": secrets.token_urlsafe(32),
        }
        return {"token": self._sign(payload), "payload": payload}

    def validate_token(self, token: str) -> dict | None:
        """Validate a token and return its payload if valid."""
        payload = self._decode(token)
        if payload is None:
            return None
        if payload.get("jti") in self._revoked:
            return None
        exp = datetime.fromisoformat(payload["exp"])
        if datetime.utcnow() > exp:
            return None
        return payload

    def revoke_token(self, jti: str) -> None:
        """Revoke a token by its JTI."""
        self._revoked.add(jti)

    def _sign(self, payload: dict) -> str:
        """Sign a payload (simplified — real impl would use JWT)."""
        import json as _json
        raw = _json.dumps(payload, sort_keys=True)
        sig = hashlib.sha256((raw + self.secret).encode()).hexdigest()[:16]
        return f"{raw}.{sig}"

    def _decode(self, token: str) -> dict | None:
        """Decode and verify a signed token."""
        import json as _json
        try:
            raw, sig = token.rsplit(".", 1)
            expected = hashlib.sha256((raw + self.secret).encode()).hexdigest()[:16]
            if not secrets.compare_digest(sig, expected):
                return None
            return _json.loads(raw)
        except (ValueError, _json.JSONDecodeError):
            return None


def create_session(user_id: str, token_manager: TokenManager) -> dict:
    """Create a new user session with token."""
    token_data = token_manager.create_token(user_id)
    return {
        "session_id": secrets.token_urlsafe(16),
        "user_id": user_id,
        "token": token_data["token"],
        "created_at": datetime.utcnow().isoformat(),
    }


class RateLimiter:
    """Simple in-memory rate limiter using sliding window."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = {}

    def is_allowed(self, client_id: str) -> bool:
        """Check if a request from client_id is allowed."""
        import time
        now = time.time()
        cutoff = now - self.window_seconds

        if client_id not in self._requests:
            self._requests[client_id] = []

        # Remove expired entries
        self._requests[client_id] = [
            t for t in self._requests[client_id] if t > cutoff
        ]

        if len(self._requests[client_id]) >= self.max_requests:
            return False

        self._requests[client_id].append(now)
        return True

    def reset(self, client_id: str) -> None:
        """Reset rate limit for a client."""
        self._requests.pop(client_id, None)

    def get_remaining(self, client_id: str) -> int:
        """Get remaining requests in current window."""
        import time
        now = time.time()
        cutoff = now - self.window_seconds
        current = [t for t in self._requests.get(client_id, []) if t > cutoff]
        return max(0, self.max_requests - len(current))
'''

GO_CODE = """package cache

import (
\t"container/list"
\t"sync"
\t"time"
)

// Entry represents a single cache entry with TTL support.
type Entry struct {
\tKey       string
\tValue     interface{}
\tExpiresAt time.Time
\tSize      int
}

// LRUCache implements a thread-safe LRU cache with TTL expiration.
type LRUCache struct {
\tmu       sync.RWMutex
\tcapacity int
\titems    map[string]*list.Element
\torder    *list.List
\tttl      time.Duration
\tstats    CacheStats
}

// CacheStats tracks cache hit/miss statistics.
type CacheStats struct {
\tHits       int64
\tMisses     int64
\tEvictions  int64
\tExpired    int64
}

func NewLRUCache(capacity int, ttl time.Duration) *LRUCache {
\treturn &LRUCache{
\t\tcapacity: capacity,
\t\titems:    make(map[string]*list.Element),
\t\torder:    list.New(),
\t\tttl:      ttl,
\t}
}

func (c *LRUCache) Get(key string) (interface{}, bool) {
\tc.mu.Lock()
\tdefer c.mu.Unlock()

\telem, ok := c.items[key]
\tif !ok {
\t\tc.stats.Misses++
\t\treturn nil, false
\t}

\tentry := elem.Value.(*Entry)
\tif time.Now().After(entry.ExpiresAt) {
\t\tc.removeElement(elem)
\t\tc.stats.Expired++
\t\tc.stats.Misses++
\t\treturn nil, false
\t}

\tc.order.MoveToFront(elem)
\tc.stats.Hits++
\treturn entry.Value, true
}

func (c *LRUCache) Set(key string, value interface{}, size int) {
\tc.mu.Lock()
\tdefer c.mu.Unlock()

\tif elem, ok := c.items[key]; ok {
\t\tc.order.MoveToFront(elem)
\t\tentry := elem.Value.(*Entry)
\t\tentry.Value = value
\t\tentry.ExpiresAt = time.Now().Add(c.ttl)
\t\tentry.Size = size
\t\treturn
\t}

\tfor c.order.Len() >= c.capacity {
\t\tc.evict()
\t}

\tentry := &Entry{
\t\tKey:       key,
\t\tValue:     value,
\t\tExpiresAt: time.Now().Add(c.ttl),
\t\tSize:      size,
\t}
\telem := c.order.PushFront(entry)
\tc.items[key] = elem
}

func (c *LRUCache) Delete(key string) bool {
\tc.mu.Lock()
\tdefer c.mu.Unlock()

\telem, ok := c.items[key]
\tif !ok {
\t\treturn false
\t}
\tc.removeElement(elem)
\treturn true
}

func (c *LRUCache) evict() {
\telem := c.order.Back()
\tif elem != nil {
\t\tc.removeElement(elem)
\t\tc.stats.Evictions++
\t}
}

func (c *LRUCache) removeElement(elem *list.Element) {
\tc.order.Remove(elem)
\tentry := elem.Value.(*Entry)
\tdelete(c.items, entry.Key)
}

func (c *LRUCache) Stats() CacheStats {
\tc.mu.RLock()
\tdefer c.mu.RUnlock()
\treturn c.stats
}

func (c *LRUCache) Len() int {
\tc.mu.RLock()
\tdefer c.mu.RUnlock()
\treturn c.order.Len()
}

func (c *LRUCache) Purge() {
\tc.mu.Lock()
\tdefer c.mu.Unlock()

\tc.items = make(map[string]*list.Element)
\tc.order.Init()
}
"""

# Ground truth: queries and which functions they should retrieve
CODE_QUERIES = [
    {
        "query": "how to hash and verify passwords securely",
        "file": "auth.py",
        "language": "python",
        "expected_functions": ["hash_password", "verify_password"],
        "code": PYTHON_CODE,
    },
    {
        "query": "create and validate authentication tokens",
        "file": "auth.py",
        "language": "python",
        "expected_functions": ["create_token", "validate_token"],
        "code": PYTHON_CODE,
    },
    {
        "query": "rate limiting with sliding window",
        "file": "auth.py",
        "language": "python",
        "expected_functions": ["is_allowed", "get_remaining", "RateLimiter"],
        "code": PYTHON_CODE,
    },
    {
        "query": "revoke a JWT token",
        "file": "auth.py",
        "language": "python",
        "expected_functions": ["revoke_token", "validate_token"],
        "code": PYTHON_CODE,
    },
    {
        "query": "LRU cache get and set operations",
        "file": "cache.go",
        "language": "go",
        "expected_functions": ["Get", "Set"],
        "code": GO_CODE,
    },
    {
        "query": "cache eviction and memory management",
        "file": "cache.go",
        "language": "go",
        "expected_functions": ["evict", "removeElement", "Delete"],
        "code": GO_CODE,
    },
    {
        "query": "cache hit miss statistics tracking",
        "file": "cache.go",
        "language": "go",
        "expected_functions": ["Stats", "CacheStats"],
        "code": GO_CODE,
    },
    {
        "query": "create new session with token manager",
        "file": "auth.py",
        "language": "python",
        "expected_functions": ["create_session", "create_token"],
        "code": PYTHON_CODE,
    },
]


# ------------------------------------------------------------------ #
# Chunking strategies                                                  #
# ------------------------------------------------------------------ #


def chunk_naive(text: str, chunk_size: int = 1024, overlap: int = 128) -> list[str]:
    """Fixed-window chunking that ignores code structure."""
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        if chunk_text.strip():
            chunks.append(chunk_text)
        start += chunk_size - overlap

    return chunks


def chunk_code_aware(
    text: str, language: str, chunk_size: int = 1024, overlap: int = 128
) -> list[str]:
    """Code-aware chunking using vstash's chunk_code."""
    chunks = chunk_code(text, chunk_size=chunk_size, overlap=overlap, language=language)
    return [c for c in chunks if c.strip()]


# ------------------------------------------------------------------ #
# Evaluation                                                           #
# ------------------------------------------------------------------ #


@dataclass
class ChunkingResult:
    query: str
    strategy: str
    n_chunks: int
    avg_chunk_tokens: float
    function_recall: float  # fraction of expected functions found in top-k
    function_precision: float  # fraction of top-k chunks that contain expected functions
    top_chunks_preview: list[str]  # first 80 chars of each top chunk
    boundary_violations: int  # chunks that split a function mid-body


def count_tokens(text: str) -> int:
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def check_function_in_chunk(chunk: str, function_name: str) -> bool:
    """Check if a function/method definition or significant usage is in the chunk."""
    # Look for definition
    patterns = [
        f"def {function_name}",
        f"func {function_name}",
        f"func ({function_name}",  # Go method
        f"class {function_name}",
        f"type {function_name}",
    ]
    for p in patterns:
        if p in chunk:
            return True

    # For Go methods like (c *LRUCache) Get
    return f") {function_name}(" in chunk


def count_boundary_violations(chunks: list[str], language: str) -> int:
    """Count chunks that appear to split a function mid-body."""
    violations = 0

    if language == "python":
        for chunk in chunks:
            lines = chunk.strip().split("\n")
            if not lines:
                continue
            first_line = lines[0].strip()
            # If first line is indented code (not a def/class/import/comment)
            if (
                first_line
                and not first_line.startswith(
                    ("def ", "class ", "import ", "from ", "#", '"""', "'''", "@")
                )
                and first_line[0] == " "
            ) or first_line.startswith("    "):
                violations += 1

    elif language == "go":
        for chunk in chunks:
            lines = chunk.strip().split("\n")
            if not lines:
                continue
            first_line = lines[0].strip()
            # If starts mid-function (indented, not a top-level decl)
            if (
                first_line
                and not first_line.startswith(
                    ("package ", "import ", "func ", "type ", "//", "var ", "const ")
                )
                and first_line not in (")", "}")
            ):
                violations += 1

    return violations


def evaluate_chunking(
    store_naive: VstashStore,
    store_code: VstashStore,
    top_k: int = 3,
) -> tuple[list[ChunkingResult], list[ChunkingResult]]:
    """Evaluate both chunking strategies on all code queries."""
    naive_results = []
    code_results = []

    for cq in CODE_QUERIES:
        q = cq["query"]
        expected_fns = cq["expected_functions"]
        emb = embed_query(q, MODEL)

        # Naive search
        naive_chunks = store_naive.search(
            query_embedding=emb,
            query_text=q,
            top_k=top_k,
            scoring=None,
        )
        naive_texts = [c.text for c in naive_chunks]
        naive_recall = sum(
            1 for fn in expected_fns if any(check_function_in_chunk(t, fn) for t in naive_texts)
        ) / len(expected_fns)
        naive_precision = sum(
            1 for t in naive_texts if any(check_function_in_chunk(t, fn) for fn in expected_fns)
        ) / max(len(naive_texts), 1)

        naive_results.append(
            ChunkingResult(
                query=q,
                strategy="naive",
                n_chunks=len(naive_chunks),
                avg_chunk_tokens=sum(count_tokens(c.text) for c in naive_chunks)
                / max(len(naive_chunks), 1),
                function_recall=naive_recall,
                function_precision=naive_precision,
                top_chunks_preview=[c.text[:80].replace("\n", " ") for c in naive_chunks],
                boundary_violations=0,
            )
        )

        # Code-aware search
        code_chunks = store_code.search(
            query_embedding=emb,
            query_text=q,
            top_k=top_k,
            scoring=None,
        )
        code_texts = [c.text for c in code_chunks]
        code_recall = sum(
            1 for fn in expected_fns if any(check_function_in_chunk(t, fn) for t in code_texts)
        ) / len(expected_fns)
        code_precision = sum(
            1 for t in code_texts if any(check_function_in_chunk(t, fn) for fn in expected_fns)
        ) / max(len(code_texts), 1)

        code_results.append(
            ChunkingResult(
                query=q,
                strategy="code_aware",
                n_chunks=len(code_chunks),
                avg_chunk_tokens=sum(count_tokens(c.text) for c in code_chunks)
                / max(len(code_chunks), 1),
                function_recall=code_recall,
                function_precision=code_precision,
                top_chunks_preview=[c.text[:80].replace("\n", " ") for c in code_chunks],
                boundary_violations=0,
            )
        )

    return naive_results, code_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Code chunking evaluation")
    parser.add_argument("--output", type=str, default="experiments/results/chunking.json")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    dim = get_embedding_dim(MODEL)
    sep = "=" * 80

    # ============================================================== #
    # Phase 1: Chunk analysis (no search)                              #
    # ============================================================== #
    print(f"\n{sep}")
    print("  PHASE 1: Chunking Strategy Comparison")
    print(f"{sep}\n")

    files = {
        "auth.py": {"code": PYTHON_CODE, "language": "python"},
        "cache.go": {"code": GO_CODE, "language": "go"},
    }

    chunking_stats = {}

    for filename, info in files.items():
        code = info["code"]
        lang = info["language"]

        naive_chunks = chunk_naive(code)
        code_chunks = chunk_code_aware(code, lang)

        naive_violations = count_boundary_violations(naive_chunks, lang)
        code_violations = count_boundary_violations(code_chunks, lang)

        naive_tokens = [count_tokens(c) for c in naive_chunks]
        code_tokens = [count_tokens(c) for c in code_chunks]

        stats = {
            "naive": {
                "n_chunks": len(naive_chunks),
                "avg_tokens": sum(naive_tokens) / len(naive_tokens) if naive_tokens else 0,
                "min_tokens": min(naive_tokens) if naive_tokens else 0,
                "max_tokens": max(naive_tokens) if naive_tokens else 0,
                "boundary_violations": naive_violations,
            },
            "code_aware": {
                "n_chunks": len(code_chunks),
                "avg_tokens": sum(code_tokens) / len(code_tokens) if code_tokens else 0,
                "min_tokens": min(code_tokens) if code_tokens else 0,
                "max_tokens": max(code_tokens) if code_tokens else 0,
                "boundary_violations": code_violations,
            },
        }
        chunking_stats[filename] = stats

        print(f"  {filename} ({lang}):")
        print(
            f"    Naive:      {len(naive_chunks)} chunks, "
            f"avg {stats['naive']['avg_tokens']:.0f} tokens, "
            f"{naive_violations} boundary violations"
        )
        print(
            f"    Code-aware: {len(code_chunks)} chunks, "
            f"avg {stats['code_aware']['avg_tokens']:.0f} tokens, "
            f"{code_violations} boundary violations"
        )
        print()

        # Show chunk boundaries
        print("    Naive chunks first lines:")
        for i, c in enumerate(naive_chunks):
            first_line = c.strip().split("\n")[0][:70]
            print(f"      [{i}] {first_line}")
        print("    Code-aware chunks first lines:")
        for i, c in enumerate(code_chunks):
            first_line = c.strip().split("\n")[0][:70]
            print(f"      [{i}] {first_line}")
        print()

    # ============================================================== #
    # Phase 2: Retrieval evaluation                                    #
    # ============================================================== #
    print(f"\n{sep}")
    print("  PHASE 2: Retrieval Quality Comparison")
    print(f"{sep}\n")

    # Create two separate DBs
    tmp_dir = tempfile.mkdtemp(prefix="vstash_chunk_eval_")

    # Naive DB
    db_naive_path = str(Path(tmp_dir) / "naive.db")
    store_naive = VstashStore(db_naive_path, embedding_dim=dim)

    # Code-aware DB
    db_code_path = str(Path(tmp_dir) / "code_aware.db")
    store_code = VstashStore(db_code_path, embedding_dim=dim)

    # Ingest with naive chunking
    import struct

    from vstash.embed import embed_texts

    def _ingest_chunks(store, doc_id, filename, code, chunks_list, embeddings_list):
        """Insert chunks with vector + FTS5 indexes."""
        store._conn.execute(
            "INSERT OR REPLACE INTO documents (id, path, title, source_type, collection, char_count, chunk_count, added_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))",
            [doc_id, filename, filename, "code", "eval", len(code), len(chunks_list)],
        )
        for i, (chunk, emb) in enumerate(zip(chunks_list, embeddings_list)):
            store._conn.execute(
                "INSERT INTO chunks (doc_id, seq, text) VALUES (?, ?, ?)",
                [doc_id, i, chunk],
            )
            chunk_id = store._conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            blob = struct.pack(f"{dim}f", *emb)
            store._conn.execute(
                "INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)",
                [chunk_id, blob],
            )
            store._conn.execute(
                "INSERT INTO fts_chunks (rowid, text) VALUES (?, ?)",
                [chunk_id, chunk],
            )
        store._conn.commit()

    for filename, info in files.items():
        code = info["code"]
        lang = info["language"]

        # Naive
        naive_chunks = chunk_naive(code)
        naive_embeddings = embed_texts(naive_chunks, MODEL)
        _ingest_chunks(
            store_naive, f"naive_{filename}", filename, code, naive_chunks, naive_embeddings
        )

        # Code-aware
        code_chunks = chunk_code_aware(code, lang)
        code_embeddings = embed_texts(code_chunks, MODEL)
        _ingest_chunks(store_code, f"code_{filename}", filename, code, code_chunks, code_embeddings)

    # Evaluate at multiple top_k values
    for k in [3, 5, 7]:
        n_res, c_res = evaluate_chunking(store_naive, store_code, top_k=k)
        n_avg = sum(r.function_recall for r in n_res) / len(n_res)
        c_avg = sum(r.function_recall for r in c_res) / len(c_res)
        print(f"  top_k={k}: naive recall={n_avg:.3f}, code-aware recall={c_avg:.3f}")

    # Primary evaluation
    naive_results, code_results = evaluate_chunking(store_naive, store_code, top_k=args.top_k)

    # Print results
    print(f"  {'Query':<50} {'Strategy':<12} {'Recall':>8} {'Prec':>8}")
    print(f"  {'─' * 82}")

    for nr, cr in zip(naive_results, code_results):
        print(
            f"  {nr.query[:50]:<50} {'naive':<12} {nr.function_recall:>8.2f} {nr.function_precision:>8.2f}"
        )
        print(
            f"  {'':<50} {'code-aware':<12} {cr.function_recall:>8.2f} {cr.function_precision:>8.2f}"
        )
        print()

    # Summary
    avg_naive_recall = sum(r.function_recall for r in naive_results) / len(naive_results)
    avg_code_recall = sum(r.function_recall for r in code_results) / len(code_results)
    avg_naive_precision = sum(r.function_precision for r in naive_results) / len(naive_results)
    avg_code_precision = sum(r.function_precision for r in code_results) / len(code_results)

    print(f"\n{sep}")
    print("  SUMMARY")
    print(f"{sep}\n")
    print(f"  {'Strategy':<15} {'Avg Recall':>12} {'Avg Precision':>15}")
    print(f"  {'─' * 42}")
    print(f"  {'Naive':<15} {avg_naive_recall:>12.4f} {avg_naive_precision:>15.4f}")
    print(f"  {'Code-aware':<15} {avg_code_recall:>12.4f} {avg_code_precision:>15.4f}")

    if avg_naive_recall > 0:
        recall_improvement = ((avg_code_recall / avg_naive_recall) - 1) * 100
        print(f"\n  Code-aware recall improvement: {recall_improvement:+.1f}%")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "experiment": "chunking_eval",
        "top_k": args.top_k,
        "files": list(files.keys()),
        "chunking_stats": chunking_stats,
        "naive_results": [asdict(r) for r in naive_results],
        "code_aware_results": [asdict(r) for r in code_results],
        "summary": {
            "naive_avg_recall": avg_naive_recall,
            "code_avg_recall": avg_code_recall,
            "naive_avg_precision": avg_naive_precision,
            "code_avg_precision": avg_code_precision,
        },
    }
    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"\n  Results saved to: {output_path}")

    store_naive.close()
    store_code.close()


if __name__ == "__main__":
    main()
