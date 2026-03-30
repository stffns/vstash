"""Tests for code-aware chunking (hybrid: tree-sitter → parso → regex)."""

from __future__ import annotations


from vstash.code_split import EXT_TO_LANG as _EXT_TO_LANG
from vstash.code_split import split_code_blocks as _split_code_blocks
from vstash.ingest import chunk_code


# ------------------------------------------------------------------ #
# _split_code_blocks — Python                                         #
# ------------------------------------------------------------------ #


class TestSplitCodeBlocksPython:
    """Test Python-specific code splitting."""

    def test_splits_at_top_level_functions(self) -> None:
        code = "def foo():\n    pass\n\ndef bar():\n    pass\n"
        blocks = _split_code_blocks(code, "python")
        assert len(blocks) == 2
        assert "def foo" in blocks[0]
        assert "def bar" in blocks[1]

    def test_splits_at_classes(self) -> None:
        code = "class A:\n    pass\n\nclass B:\n    pass\n"
        blocks = _split_code_blocks(code, "python")
        assert len(blocks) == 2
        assert "class A" in blocks[0]
        assert "class B" in blocks[1]

    def test_keeps_imports_as_preamble(self) -> None:
        code = "import os\nimport sys\n\ndef main():\n    pass\n"
        blocks = _split_code_blocks(code, "python")
        assert len(blocks) == 2
        assert "import os" in blocks[0]
        assert "def main" in blocks[1]

    def test_indented_methods_not_split(self) -> None:
        """CRITICAL: indented methods inside a class must NOT cause extra splits."""
        code = (
            "class MyClass:\n"
            "    def method_one(self):\n"
            "        pass\n"
            "\n"
            "    def method_two(self):\n"
            "        pass\n"
        )
        blocks = _split_code_blocks(code.strip(), "python")
        assert len(blocks) == 1
        assert "class MyClass" in blocks[0]
        assert "method_one" in blocks[0]
        assert "method_two" in blocks[0]

    def test_async_def(self) -> None:
        code = "async def fetch():\n    pass\n\ndef sync():\n    pass\n"
        blocks = _split_code_blocks(code, "python")
        assert len(blocks) == 2
        assert "async def fetch" in blocks[0]

    def test_decorator_starts_new_block(self) -> None:
        code = "def first():\n    pass\n\n@app.route('/api')\ndef handler():\n    return 'ok'\n"
        blocks = _split_code_blocks(code.strip(), "python")
        assert len(blocks) == 2
        assert "@app.route" in blocks[1]
        assert "def handler" in blocks[1]

    def test_class_with_decorator(self) -> None:
        code = "@dataclass\nclass Config:\n    name: str\n"
        blocks = _split_code_blocks(code.strip(), "python")
        assert len(blocks) == 1
        assert "@dataclass" in blocks[0]
        assert "class Config" in blocks[0]


# ------------------------------------------------------------------ #
# _split_code_blocks — JavaScript                                     #
# ------------------------------------------------------------------ #


class TestSplitCodeBlocksJavaScript:
    def test_function_declarations(self) -> None:
        code = "function foo() {\n  return 1;\n}\n\nfunction bar() {\n  return 2;\n}\n"
        blocks = _split_code_blocks(code, "javascript")
        assert len(blocks) == 2

    def test_class_declarations(self) -> None:
        code = "class Foo {\n  constructor() {}\n}\n\nclass Bar {\n  constructor() {}\n}\n"
        blocks = _split_code_blocks(code, "javascript")
        assert len(blocks) == 2

    def test_export_default(self) -> None:
        code = "export default function main() {\n  return 1;\n}\n"
        blocks = _split_code_blocks(code, "javascript")
        assert len(blocks) == 1
        assert "export default" in blocks[0]

    def test_preamble_preserved(self) -> None:
        code = "const x = 1;\n\nfunction foo() {}\n"
        blocks = _split_code_blocks(code, "javascript")
        assert len(blocks) == 2
        assert "const x" in blocks[0]


# ------------------------------------------------------------------ #
# _split_code_blocks — Go                                             #
# ------------------------------------------------------------------ #


class TestSplitCodeBlocksGo:
    def test_func_declarations(self) -> None:
        code = 'package main\n\nimport "fmt"\n\nfunc main() {\n    fmt.Println("hi")\n}\n\nfunc helper() int {\n    return 1\n}\n'
        blocks = _split_code_blocks(code, "go")
        assert len(blocks) == 3  # preamble + 2 funcs
        assert "package main" in blocks[0]
        assert "func main" in blocks[1]
        assert "func helper" in blocks[2]

    def test_type_struct(self) -> None:
        code = "type Config struct {\n    Name string\n}\n\nfunc New() Config {\n    return Config{}\n}\n"
        blocks = _split_code_blocks(code, "go")
        assert len(blocks) == 2
        assert "type Config struct" in blocks[0]
        assert "func New" in blocks[1]


# ------------------------------------------------------------------ #
# _split_code_blocks — Rust                                           #
# ------------------------------------------------------------------ #


class TestSplitCodeBlocksRust:
    def test_fn_declarations(self) -> None:
        code = 'fn main() {\n    println!("hello");\n}\n\nfn helper() -> i32 {\n    42\n}\n'
        blocks = _split_code_blocks(code, "rust")
        assert len(blocks) == 2

    def test_pub_fn(self) -> None:
        code = "pub fn public_func() {}\n\nfn private_func() {}\n"
        blocks = _split_code_blocks(code, "rust")
        assert len(blocks) == 2
        assert "pub fn" in blocks[0]

    def test_impl_block(self) -> None:
        code = "struct Foo;\n\nimpl Foo {\n    fn new() -> Self { Foo }\n}\n"
        blocks = _split_code_blocks(code, "rust")
        assert len(blocks) == 2
        assert "struct Foo" in blocks[0]
        assert "impl Foo" in blocks[1]


# ------------------------------------------------------------------ #
# _split_code_blocks — Java                                           #
# ------------------------------------------------------------------ #


class TestSplitCodeBlocksJava:
    def test_class_declaration(self) -> None:
        code = "public class Main {\n    public static void main(String[] args) {}\n}\n"
        blocks = _split_code_blocks(code, "java")
        assert len(blocks) == 1
        assert "public class Main" in blocks[0]

    def test_interface(self) -> None:
        code = "public interface Repo {\n    void save();\n}\n\npublic class RepoImpl {\n    void save() {}\n}\n"
        blocks = _split_code_blocks(code, "java")
        assert len(blocks) == 2


# ------------------------------------------------------------------ #
# _split_code_blocks — Edge cases                                     #
# ------------------------------------------------------------------ #


class TestSplitCodeBlocksEdgeCases:
    def test_unknown_language_returns_whole_text(self) -> None:
        blocks = _split_code_blocks("some code here", "cobol")
        assert blocks == ["some code here"]

    def test_empty_input(self) -> None:
        blocks = _split_code_blocks("", "python")
        assert blocks == [""]

    def test_no_definitions_returns_whole_text(self) -> None:
        code = "# just a comment\nx = 1\ny = 2\n"
        blocks = _split_code_blocks(code, "python")
        assert len(blocks) == 1
        assert "x = 1" in blocks[0]


# ------------------------------------------------------------------ #
# chunk_code                                                           #
# ------------------------------------------------------------------ #


class TestChunkCode:
    def test_small_file_single_chunk(self) -> None:
        code = "def foo():\n    return 1\n\ndef bar():\n    return 2\n"
        chunks = chunk_code(code, 1024, 128, "python")
        assert len(chunks) >= 1
        # Both functions should be in the output
        all_text = " ".join(chunks)
        assert "def foo" in all_text
        assert "def bar" in all_text

    def test_multiple_functions_preserved(self) -> None:
        """Each function should appear in at least one chunk."""
        funcs = [f"def func_{i}():\n    '''Docstring {i}.'''\n    return {i}\n" for i in range(10)]
        code = "\n\n".join(funcs)
        chunks = chunk_code(code, 1024, 128, "python")
        all_text = " ".join(chunks)
        for i in range(10):
            assert f"func_{i}" in all_text

    def test_token_limits_respected(self) -> None:
        """No chunk should exceed chunk_size tokens."""
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        # Generate a large file
        funcs = [
            f"def func_{i}():\n    '''Long docstring {'x' * 200}.'''\n    return {i}\n"
            for i in range(20)
        ]
        code = "\n\n".join(funcs)
        chunks = chunk_code(code, 256, 64, "python")
        for chunk in chunks:
            assert len(enc.encode(chunk)) <= 256 + 10  # small tolerance for merging

    def test_empty_input(self) -> None:
        assert chunk_code("", 1024, 128, "python") == []
        assert chunk_code("   \n  ", 1024, 128, "python") == []

    def test_unknown_language_still_chunks(self) -> None:
        """Unknown language should fall back to paragraph-based splitting."""
        code = "some code\nwith lines\n\nanother paragraph"
        chunks = chunk_code(code, 1024, 128, "unknown")
        assert len(chunks) >= 1


# ------------------------------------------------------------------ #
# Extension → language mapping                                         #
# ------------------------------------------------------------------ #


class TestExtToLang:
    def test_python(self) -> None:
        assert _EXT_TO_LANG[".py"] == "python"

    def test_javascript_variants(self) -> None:
        assert _EXT_TO_LANG[".js"] == "javascript"
        assert _EXT_TO_LANG[".jsx"] == "javascript"

    def test_typescript_variants(self) -> None:
        assert _EXT_TO_LANG[".ts"] == "typescript"
        assert _EXT_TO_LANG[".tsx"] == "typescript"

    def test_go(self) -> None:
        assert _EXT_TO_LANG[".go"] == "go"

    def test_rust(self) -> None:
        assert _EXT_TO_LANG[".rs"] == "rust"

    def test_java(self) -> None:
        assert _EXT_TO_LANG[".java"] == "java"
