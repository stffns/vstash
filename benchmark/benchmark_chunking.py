#!/usr/bin/env python3
"""
benchmark_chunking.py — A/B comparison: fixed-window vs semantic chunking.

Compares chunk quality, count, and coherence on the benchmark corpus.
Outputs a markdown report with side-by-side results.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import tiktoken

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the chunking functions directly to avoid store.py import chain
# which requires Python 3.11+ for datetime.UTC
import importlib.util

_ingest_spec = importlib.util.spec_from_file_location(
    "_ingest_chunking",
    str(Path(__file__).parent.parent / "vstash" / "ingest.py"),
    submodule_search_locations=[],
)

# We can't import ingest.py directly due to relative imports — extract functions inline
# Copy the logic here to keep benchmark self-contained on Python 3.10

_HEADER_RE = re.compile(r"^(#{1,6})\s+", re.MULTILINE)
_MIN_CHUNK_TOKENS = 80


def _split_by_headers(text: str) -> list[str]:
    positions = [m.start() for m in _HEADER_RE.finditer(text)]
    if not positions:
        return [text]
    sections: list[str] = []
    if positions[0] > 0:
        preamble = text[: positions[0]].strip()
        if preamble:
            sections.append(preamble)
    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        section = text[start:end].strip()
        if section:
            sections.append(section)
    return sections


def _fixed_window_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    tokens = enc.encode(text)
    if not tokens:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_text_str = enc.decode(tokens[start:end]).strip()
        if chunk_text_str and len(chunk_text_str) > 20:
            chunks.append(chunk_text_str)
        if end >= len(tokens):
            break
        start += chunk_size - overlap
    return chunks


def _split_by_paragraphs(text: str, chunk_size: int, overlap: int) -> list[str]:
    paragraphs = re.split(r"\n{2,}", text)
    result: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_tokens = len(enc.encode(para))
        if para_tokens > chunk_size:
            if current:
                result.append("\n\n".join(current))
                current, current_tokens = [], 0
            result.extend(_fixed_window_chunks(para, chunk_size, overlap))
            continue
        if current_tokens + para_tokens > chunk_size and current:
            result.append("\n\n".join(current))
            current, current_tokens = [], 0
        current.append(para)
        current_tokens += para_tokens
    if current:
        result.append("\n\n".join(current))
    return result


def _merge_small_chunks(chunks: list[str], chunk_size: int) -> list[str]:
    if not chunks:
        return []
    merged: list[str] = []
    current = chunks[0]
    current_tokens = len(enc.encode(current))
    for chunk in chunks[1:]:
        chunk_tokens = len(enc.encode(chunk))
        if current_tokens < _MIN_CHUNK_TOKENS and current_tokens + chunk_tokens <= chunk_size:
            current = current + "\n\n" + chunk
            current_tokens += chunk_tokens
        else:
            merged.append(current)
            current = chunk
            current_tokens = chunk_tokens
    merged.append(current)
    return merged


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if not text or not text.strip():
        return []
    sections = _split_by_headers(text)
    sized_chunks: list[str] = []
    for section in sections:
        tc = len(enc.encode(section))
        if tc <= chunk_size:
            sized_chunks.append(section)
        else:
            sized_chunks.extend(_split_by_paragraphs(section, chunk_size, overlap))
    merged = _merge_small_chunks(sized_chunks, chunk_size)
    return [c for c in merged if c.strip() and len(c.strip()) > 20]

enc = tiktoken.get_encoding("cl100k_base")

CORPUS_DIR = Path(__file__).parent / "corpus"
REPORT_PATH = Path(__file__).parent / "CHUNKING_BENCHMARK.md"

# Default config values
CHUNK_SIZE = 1024
OVERLAP = 128


def token_count(text: str) -> int:
    return len(enc.encode(text))


def analyze_chunks(chunks: list[str]) -> dict:
    """Compute quality metrics for a list of chunks."""
    if not chunks:
        return {"count": 0, "avg_tokens": 0, "min_tokens": 0, "max_tokens": 0,
                "header_preserved": 0, "header_torn": 0}

    token_counts = [token_count(c) for c in chunks]
    # Count headers that appear with their body content
    header_preserved = 0
    header_torn = 0
    for c in chunks:
        headers = re.findall(r"^(#{1,6})\s+(.+)$", c, re.MULTILINE)
        for _, title in headers:
            # Check if there's body text after the header in the same chunk
            # (at least 50 chars of non-header content)
            non_header = re.sub(r"^#{1,6}\s+.*$", "", c, flags=re.MULTILINE).strip()
            if len(non_header) > 50:
                header_preserved += 1
            else:
                header_torn += 1

    return {
        "count": len(chunks),
        "avg_tokens": round(sum(token_counts) / len(token_counts), 1),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
        "header_preserved": header_preserved,
        "header_torn": header_torn,
    }


def benchmark_file(filepath: Path) -> dict:
    """Run both chunking strategies on a single file and compare."""
    text = filepath.read_text()
    total_tokens = token_count(text)

    # --- Fixed window (old) ---
    t0 = time.perf_counter()
    fixed_chunks = _fixed_window_chunks(text, CHUNK_SIZE, OVERLAP)
    fixed_time = time.perf_counter() - t0

    # --- Semantic (new) ---
    t0 = time.perf_counter()
    semantic_chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)
    semantic_time = time.perf_counter() - t0

    return {
        "file": filepath.name,
        "total_tokens": total_tokens,
        "fixed": {**analyze_chunks(fixed_chunks), "time_ms": round(fixed_time * 1000, 2)},
        "semantic": {**analyze_chunks(semantic_chunks), "time_ms": round(semantic_time * 1000, 2)},
        "fixed_chunks_sample": fixed_chunks[:2] if fixed_chunks else [],
        "semantic_chunks_sample": semantic_chunks[:2] if semantic_chunks else [],
    }


def main() -> None:
    print("=" * 70)
    print("  Chunking Benchmark: Fixed Window vs Semantic")
    print("=" * 70)

    # Find corpus files
    files = sorted(CORPUS_DIR.glob("*.md")) + sorted(CORPUS_DIR.glob("*.txt"))
    if not files:
        print("No corpus files found. Run benchmark.py first to generate them.")
        sys.exit(1)

    results = []
    for f in files:
        print(f"\n  Processing: {f.name}")
        r = benchmark_file(f)
        results.append(r)
        print(f"    Fixed:    {r['fixed']['count']} chunks, "
              f"avg {r['fixed']['avg_tokens']} tok, {r['fixed']['time_ms']}ms")
        print(f"    Semantic: {r['semantic']['count']} chunks, "
              f"avg {r['semantic']['avg_tokens']} tok, {r['semantic']['time_ms']}ms")

    # --- Generate report ---
    lines: list[str] = []
    lines.append("# Chunking Benchmark: Fixed Window vs Semantic\n")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Config:** chunk_size={CHUNK_SIZE}, overlap={OVERLAP}\n")

    # Summary table
    lines.append("\n## Summary\n")
    lines.append("| File | Tokens | Fixed Chunks | Semantic Chunks | Fixed Avg Tok | Semantic Avg Tok | Headers Preserved (Semantic) |")
    lines.append("|------|--------|-------------|----------------|--------------|-----------------|------------------------------|")

    total_fixed = 0
    total_semantic = 0
    total_preserved = 0
    total_torn = 0

    for r in results:
        total_fixed += r["fixed"]["count"]
        total_semantic += r["semantic"]["count"]
        total_preserved += r["semantic"]["header_preserved"]
        total_torn += r["semantic"]["header_torn"]
        lines.append(
            f"| {r['file']} | {r['total_tokens']} | "
            f"{r['fixed']['count']} | {r['semantic']['count']} | "
            f"{r['fixed']['avg_tokens']} | {r['semantic']['avg_tokens']} | "
            f"{r['semantic']['header_preserved']}/{r['semantic']['header_preserved'] + r['semantic']['header_torn']} |"
        )

    lines.append(
        f"| **Total** | — | **{total_fixed}** | **{total_semantic}** | "
        f"— | — | **{total_preserved}/{total_preserved + total_torn}** |"
    )

    # Performance
    lines.append("\n## Performance\n")
    lines.append("| File | Fixed Time | Semantic Time | Overhead |")
    lines.append("|------|-----------|--------------|----------|")
    for r in results:
        ft = r["fixed"]["time_ms"]
        st = r["semantic"]["time_ms"]
        overhead = f"{(st / ft - 1) * 100:.0f}%" if ft > 0 else "N/A"
        lines.append(f"| {r['file']} | {ft}ms | {st}ms | {overhead} |")

    # Sample comparison
    lines.append("\n## Sample Chunks (first chunk of each file)\n")
    for r in results:
        lines.append(f"### {r['file']}\n")
        if r["fixed_chunks_sample"]:
            first_fixed = r["fixed_chunks_sample"][0][:300].replace("\n", " ")
            lines.append(f"**Fixed (first chunk):**\n> {first_fixed}...\n")
        if r["semantic_chunks_sample"]:
            first_semantic = r["semantic_chunks_sample"][0][:300].replace("\n", " ")
            lines.append(f"**Semantic (first chunk):**\n> {first_semantic}...\n")

    # Verdict
    lines.append("\n## Verdict\n")
    lines.append(f"- Semantic chunking preserved **{total_preserved}/{total_preserved + total_torn}** "
                 f"headers with their body content")
    lines.append(f"- Fixed window: **{total_fixed}** chunks vs Semantic: **{total_semantic}** chunks")
    if total_semantic < total_fixed:
        pct = round((1 - total_semantic / total_fixed) * 100)
        lines.append(f"- **{pct}% fewer chunks** with semantic — less noise, better context per embedding")
    elif total_semantic > total_fixed:
        pct = round((total_semantic / total_fixed - 1) * 100)
        lines.append(f"- **{pct}% more chunks** with semantic — more granular, but each chunk is more coherent")

    report = "\n".join(lines)
    REPORT_PATH.write_text(report)
    print(f"\n📊 Report saved to: {REPORT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
