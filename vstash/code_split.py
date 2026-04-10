"""Code-aware splitting with graceful degradation.

Resolution order:
  1. tree-sitter-language-pack  (AST-level, 165+ languages)
  2. parso                      (AST-level, Python only)
  3. regex                      (pattern-based, 6 languages)

The public entry point is :func:`split_code_blocks`.  Callers don't need
to know which backend is active — the API is identical.
"""

from __future__ import annotations

import re
from typing import Literal

# ------------------------------------------------------------------ #
# Backend availability probing (done once at import time)             #
# ------------------------------------------------------------------ #

_HAS_TREE_SITTER = False
_HAS_PARSO = False

try:
    from tree_sitter_language_pack import get_parser  # noqa: F401

    _HAS_TREE_SITTER = True
except ImportError:
    pass

try:
    import parso as _parso  # noqa: F401

    _HAS_PARSO = True
except ImportError:
    pass

# ------------------------------------------------------------------ #
# Language / extension mapping                                        #
# ------------------------------------------------------------------ #

EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".lua": "lua",
    ".r": "r",
    ".cs": "c_sharp",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".zig": "zig",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".dart": "dart",
    ".vue": "vue",
    ".svelte": "svelte",
}

# tree-sitter language names that differ from our keys
_TS_LANG_MAP: dict[str, str] = {
    "javascript": "javascript",
    "typescript": "typescript",
    "python": "python",
    "go": "go",
    "rust": "rust",
    "java": "java",
    "c": "c",
    "cpp": "cpp",
    "ruby": "ruby",
    "php": "php",
    "swift": "swift",
    "kotlin": "kotlin",
    "scala": "scala",
    "lua": "lua",
    "r": "r",
    "c_sharp": "c_sharp",
    "bash": "bash",
    "zig": "zig",
    "elixir": "elixir",
    "erlang": "erlang",
    "haskell": "haskell",
    "ocaml": "ocaml",
    "dart": "dart",
    "vue": "vue",
    "svelte": "svelte",
}

# Node types that represent top-level definitions per language.
# tree-sitter uses these as node names in the CST.
_DEFINITION_NODE_TYPES: dict[str, set[str]] = {
    "python": {
        "function_definition",
        "class_definition",
        "decorated_definition",
    },
    "javascript": {
        "function_declaration",
        "class_declaration",
        "export_statement",
        "lexical_declaration",
    },
    "typescript": {
        "function_declaration",
        "class_declaration",
        "export_statement",
        "lexical_declaration",
        "interface_declaration",
        "type_alias_declaration",
    },
    "go": {
        "function_declaration",
        "method_declaration",
        "type_declaration",
    },
    "rust": {
        "function_item",
        "struct_item",
        "enum_item",
        "impl_item",
        "trait_item",
        "mod_item",
    },
    "java": {
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
        "method_declaration",
    },
    "c": {
        "function_definition",
        "declaration",
        "struct_specifier",
        "enum_specifier",
    },
    "cpp": {
        "function_definition",
        "declaration",
        "class_specifier",
        "struct_specifier",
        "namespace_definition",
    },
    "ruby": {
        "method",
        "class",
        "module",
    },
    "php": {
        "function_definition",
        "class_declaration",
        "method_declaration",
    },
    "swift": {
        "function_declaration",
        "class_declaration",
        "struct_declaration",
        "protocol_declaration",
    },
    "kotlin": {
        "function_declaration",
        "class_declaration",
        "object_declaration",
    },
}

# ------------------------------------------------------------------ #
# Regex fallback (original vstash patterns)                           #
# ------------------------------------------------------------------ #

_CODE_SPLIT_PATTERNS: dict[str, re.Pattern[str]] = {
    "python": re.compile(r"^(?=class |def |async def )", re.MULTILINE),
    "javascript": re.compile(
        r"^(?=function |class |const \w+ = (?:async )?\(|export (?:default )?(?:function |class |const ))",
        re.MULTILINE,
    ),
    "typescript": re.compile(
        r"^(?=function |class |const \w+ = (?:async )?\(|export (?:default )?(?:function |class |const )|interface |type \w+ )",
        re.MULTILINE,
    ),
    "go": re.compile(r"^(?=func |type \w+ (?:struct|interface))", re.MULTILINE),
    "rust": re.compile(r"^(?=(?:pub\s+)?(?:fn |struct |enum |impl |trait |mod ))", re.MULTILINE),
    "java": re.compile(
        r"^(?=(?:public |private |protected |static |abstract |final )*(?:class |interface |enum |void |int |String |boolean |long |double |float )\w)",
        re.MULTILINE,
    ),
}


# ------------------------------------------------------------------ #
# tree-sitter backend                                                 #
# ------------------------------------------------------------------ #


def _split_tree_sitter(text: str, language: str) -> list[str] | None:
    """Split using tree-sitter AST.  Returns None if unavailable."""
    if not _HAS_TREE_SITTER:
        return None

    ts_lang = _TS_LANG_MAP.get(language)
    if ts_lang is None:
        return None

    try:
        parser = get_parser(ts_lang)
    except Exception:
        return None

    encoded = text.encode("utf-8")
    tree = parser.parse(encoded)
    root = tree.root_node

    definition_types = _DEFINITION_NODE_TYPES.get(language)
    if definition_types is None:
        return None

    # Collect byte-offset boundaries for definition nodes
    def_starts: list[int] = []
    for child in root.children:
        if child.type in definition_types:
            def_starts.append(child.start_byte)

    if not def_starts:
        return None

    blocks: list[str] = []

    # Preamble: everything before the first definition
    if def_starts[0] > 0:
        preamble = encoded[: def_starts[0]].decode("utf-8").strip()
        if preamble:
            blocks.append(preamble)

    # Each definition block spans from its start to the next definition's start
    # (or end of file for the last one). This preserves inter-definition text.
    for i, start in enumerate(def_starts):
        end = def_starts[i + 1] if i + 1 < len(def_starts) else len(encoded)
        block = encoded[start:end].decode("utf-8").strip()
        if block:
            blocks.append(block)

    return blocks or None


# ------------------------------------------------------------------ #
# parso backend (Python only)                                         #
# ------------------------------------------------------------------ #


def _split_parso(text: str) -> list[str] | None:
    """Split Python code using parso AST.  Returns None if unavailable."""
    if not _HAS_PARSO:
        return None

    try:
        module = _parso.parse(text)
    except Exception:
        return None

    blocks: list[str] = []
    preamble_lines: list[str] = []

    for child in module.children:
        if child.type in ("funcdef", "classdef", "async_funcdef", "decorated"):
            if preamble_lines and not blocks:
                preamble = "".join(preamble_lines).strip()
                if preamble:
                    blocks.append(preamble)
                preamble_lines = []

            block_text = child.get_code().strip()
            if block_text:
                blocks.append(block_text)
        else:
            if not blocks:
                preamble_lines.append(child.get_code())
            else:
                # Trailing non-definition code: attach to last block
                blocks[-1] += "\n" + child.get_code().rstrip()

    if not blocks:
        return None

    return blocks


# ------------------------------------------------------------------ #
# regex fallback                                                      #
# ------------------------------------------------------------------ #


def _split_regex(text: str, language: str) -> list[str]:
    """Split using regex patterns (original vstash approach)."""
    pattern = _CODE_SPLIT_PATTERNS.get(language)
    if pattern is None:
        return [text]

    positions = [m.start() for m in pattern.finditer(text)]
    if not positions:
        return [text]

    blocks: list[str] = []

    if positions[0] > 0:
        preamble = text[: positions[0]].strip()
        if preamble:
            blocks.append(preamble)

    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        block = text[start:end].strip()
        if block:
            blocks.append(block)

    if language in ("python", "java") and len(blocks) > 1:
        blocks = _attach_decorators(blocks)

    return blocks


def _attach_decorators(blocks: list[str]) -> list[str]:
    """Move trailing @decorator lines from each block to the next block."""
    result: list[str] = []
    for i, block in enumerate(blocks):
        lines = block.split("\n")
        decorator_start = len(lines)
        for j in range(len(lines) - 1, -1, -1):
            stripped = lines[j].strip()
            if stripped.startswith("@") and not stripped.startswith("@="):
                decorator_start = j
            elif stripped:
                break

        if decorator_start < len(lines) and i + 1 < len(blocks):
            main_part = "\n".join(lines[:decorator_start]).strip()
            decorator_part = "\n".join(lines[decorator_start:]).strip()
            if main_part:
                result.append(main_part)
            blocks[i + 1] = decorator_part + "\n" + blocks[i + 1]
        else:
            result.append(block)

    return result


# ------------------------------------------------------------------ #
# Public API                                                          #
# ------------------------------------------------------------------ #

Backend = Literal["tree_sitter", "parso", "regex"]


def get_backend(language: str) -> Backend:
    """Return which backend will be used for the given language."""
    if _HAS_TREE_SITTER and _TS_LANG_MAP.get(language):
        return "tree_sitter"
    if language == "python" and _HAS_PARSO:
        return "parso"
    if language in _CODE_SPLIT_PATTERNS:
        return "regex"
    return "regex"


def split_code_blocks(text: str, language: str) -> list[str]:
    """Split source code at top-level definition boundaries.

    Tries backends in order: tree-sitter → parso → regex.
    Always returns at least ``[text]`` (never empty for non-empty input).
    """
    if not text:
        return [text]

    # 1. tree-sitter
    if _HAS_TREE_SITTER:
        result = _split_tree_sitter(text, language)
        if result:
            return result

    # 2. parso (Python only)
    if language == "python" and _HAS_PARSO:
        result = _split_parso(text)
        if result:
            return result

    # 3. regex fallback
    return _split_regex(text, language)
