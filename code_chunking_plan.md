# Code-Aware Chunking — Implementation Plan

**Issue:** #11
**Branch:** `feat/code-aware-chunking`
**Phase:** 1 (regex-based, zero new dependencies)

---

## Problem

Code files (`.py`, `.js`, `.ts`, `.go`, `.rs`, `.java`) pass through markitdown + generic
markdown chunking. This destroys code structure — functions split mid-body, classes separated
from methods, imports scattered. Searching "SSRF validation" returns docs instead of `_validate_url()`.

## Architecture

```
ingest()
  │
  source_type == "code" && !is_url && code_aware?
  ├── YES → _read_raw_code() → chunk_code() → embed + store
  └── NO  → _parse(markitdown) → chunk_text() → embed + store
```

`chunk_code()` reuses existing `_split_by_paragraphs` and `_merge_small_chunks` for
token-bounding — only the initial split strategy is new.

---

## Commits (each leaves codebase working)

### Commit 1: `_split_code_blocks()` + regex patterns

**File:** `vstash/ingest.py`

Add language-specific regex patterns that match top-level definitions at column 0:

| Language | Pattern matches |
|----------|----------------|
| Python | `class `, `def `, `async def ` at col 0 |
| JavaScript | `function `, `class `, `const x = (`, `export default` |
| TypeScript | Above + `interface `, `type X ` |
| Go | `func `, `type X struct/interface` |
| Rust | `[pub] fn/struct/enum/impl/trait/mod` |
| Java | `[public/private/...] class/interface/enum/void/...` |

**Key:** `^(?=def )` with `re.MULTILINE` does NOT match `    def method():` because
`^` matches after `\n`, but lookahead sees `    def` (with spaces), not `def `.
Indented methods are safe.

**Extension → language map:**
```python
_EXT_TO_LANG = {".py": "python", ".js": "javascript", ".ts": "typescript",
                ".go": "go", ".rs": "rust", ".java": "java"}
```

### Commit 2: `chunk_code()` function

**File:** `vstash/ingest.py`

```
chunk_code(text, chunk_size, overlap, language) → list[str]
  1. _split_code_blocks(text, language)     # structure-aware split
  2. oversized blocks → _split_by_paragraphs fallback
  3. _merge_small_chunks                    # reuse existing
```

### Commit 3: Wire into `ingest()`

**File:** `vstash/ingest.py`

- `is_code = source_type == "code" and not _is_url(source)`
- Code path: `_read_raw_code(source)` → `chunk_code()`
- Non-code path: unchanged (`_parse()` → `chunk_text()`)
- Raw read failure: fallback to markitdown path

### Commit 4: `code_aware` config toggle

**File:** `vstash/config.py`

Add `code_aware: bool = True` to `ChunkingConfig`.
Guard: `is_code = ... and cfg.chunking.code_aware`

### Commit 5: Comprehensive tests

**File:** `tests/test_code_chunking.py`

```
TestSplitCodeBlocks:
  - python: functions, classes, imports preamble, indented methods (CRITICAL), async def
  - javascript: function, class, arrow, export default
  - go: func, type struct
  - rust: pub fn, impl
  - java: class, method
  - fallback: unknown language, empty input

TestChunkCode:
  - small file → single chunk
  - large function → paragraph/window fallback
  - multiple functions stay intact
  - token limits respected

TestIngestCodeRouting:
  - .py → chunk_code
  - .md → chunk_text (unchanged)
  - code_aware=false → chunk_text
  - URL code → chunk_text (no raw read)
```

---

## Known Limitations (Phase 1)

| Limitation | Impact | Phase 2 fix |
|------------|--------|-------------|
| Decorators may separate from their function | Low — decorator becomes trailing content of previous block | Post-process: move `@` lines to next block |
| No AST parsing — regex only | Low — handles 95% of real code | tree-sitter optional dep |
| No chunk metadata (symbol name, signature) | Medium — search works but no enriched FTS5 | Step 4 from issue |
| No `source_type` filter on search | Low | Step 5 from issue |

---

## Success Criteria

- [ ] Python file chunks by function/class boundaries
- [ ] Searching function name returns that function's chunk as #1
- [ ] Searching docstring content returns semantic match
- [ ] `.md` files chunk identically to before (regression)
- [ ] Self-ingestion: "SSRF validation" returns `_validate_url()` in top 3
- [ ] All existing tests pass unchanged
