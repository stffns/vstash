"""Tests for hybrid code splitting backends (tree-sitter / parso / regex)."""

from __future__ import annotations

import pytest

from vstash.code_split import (
    EXT_TO_LANG,
    _HAS_PARSO,
    _HAS_TREE_SITTER,
    get_backend,
    split_code_blocks,
)


# ------------------------------------------------------------------ #
# Backend detection                                                    #
# ------------------------------------------------------------------ #


class TestBackendDetection:
    def test_get_backend_python(self) -> None:
        backend = get_backend("python")
        if _HAS_TREE_SITTER:
            assert backend == "tree_sitter"
        elif _HAS_PARSO:
            assert backend == "parso"
        else:
            assert backend == "regex"

    def test_get_backend_unknown_language(self) -> None:
        assert get_backend("cobol") == "regex"

    def test_get_backend_go(self) -> None:
        backend = get_backend("go")
        if _HAS_TREE_SITTER:
            assert backend == "tree_sitter"
        else:
            assert backend == "regex"


# ------------------------------------------------------------------ #
# Extended language support (tree-sitter only)                         #
# ------------------------------------------------------------------ #


class TestExtendedLanguages:
    """Test languages only supported via tree-sitter."""

    @pytest.mark.skipif(not _HAS_TREE_SITTER, reason="tree-sitter not installed")
    def test_c_function_split(self) -> None:
        code = '#include <stdio.h>\n\nint main() {\n    printf("hi");\n    return 0;\n}\n\nint helper() {\n    return 1;\n}\n'
        blocks = split_code_blocks(code, "c")
        assert len(blocks) >= 2
        all_text = " ".join(blocks)
        assert "main" in all_text
        assert "helper" in all_text

    @pytest.mark.skipif(not _HAS_TREE_SITTER, reason="tree-sitter not installed")
    def test_cpp_class_split(self) -> None:
        code = "#include <string>\n\nclass Foo {\npublic:\n    void bar() {}\n};\n\nvoid standalone() {}\n"
        blocks = split_code_blocks(code, "cpp")
        assert len(blocks) >= 2

    @pytest.mark.skipif(not _HAS_TREE_SITTER, reason="tree-sitter not installed")
    def test_ruby_method_split(self) -> None:
        code = "require 'json'\n\ndef foo\n  puts 'hi'\nend\n\ndef bar\n  puts 'bye'\nend\n"
        blocks = split_code_blocks(code, "ruby")
        assert len(blocks) >= 2

    @pytest.mark.skipif(not _HAS_TREE_SITTER, reason="tree-sitter not installed")
    def test_php_function_split(self) -> None:
        code = "<?php\n\nfunction foo() {\n    return 1;\n}\n\nfunction bar() {\n    return 2;\n}\n"
        blocks = split_code_blocks(code, "php")
        assert len(blocks) >= 2

    @pytest.mark.skipif(not _HAS_TREE_SITTER, reason="tree-sitter not installed")
    def test_kotlin_function_split(self) -> None:
        code = 'fun main() {\n    println("hello")\n}\n\nfun helper(): Int {\n    return 42\n}\n'
        blocks = split_code_blocks(code, "kotlin")
        assert len(blocks) >= 2

    @pytest.mark.skipif(not _HAS_TREE_SITTER, reason="tree-sitter not installed")
    def test_swift_function_split(self) -> None:
        code = 'import Foundation\n\nfunc greet() {\n    print("hello")\n}\n\nfunc farewell() {\n    print("bye")\n}\n'
        blocks = split_code_blocks(code, "swift")
        assert len(blocks) >= 2


# ------------------------------------------------------------------ #
# Extension mapping                                                    #
# ------------------------------------------------------------------ #


class TestExtendedExtMapping:
    def test_c_extensions(self) -> None:
        assert EXT_TO_LANG[".c"] == "c"
        assert EXT_TO_LANG[".h"] == "c"

    def test_cpp_extensions(self) -> None:
        assert EXT_TO_LANG[".cpp"] == "cpp"
        assert EXT_TO_LANG[".hpp"] == "cpp"

    def test_ruby(self) -> None:
        assert EXT_TO_LANG[".rb"] == "ruby"

    def test_php(self) -> None:
        assert EXT_TO_LANG[".php"] == "php"

    def test_swift(self) -> None:
        assert EXT_TO_LANG[".swift"] == "swift"

    def test_kotlin(self) -> None:
        assert EXT_TO_LANG[".kt"] == "kotlin"

    def test_shell(self) -> None:
        assert EXT_TO_LANG[".sh"] == "bash"
        assert EXT_TO_LANG[".bash"] == "bash"

    def test_original_langs_preserved(self) -> None:
        assert EXT_TO_LANG[".py"] == "python"
        assert EXT_TO_LANG[".js"] == "javascript"
        assert EXT_TO_LANG[".ts"] == "typescript"
        assert EXT_TO_LANG[".go"] == "go"
        assert EXT_TO_LANG[".rs"] == "rust"
        assert EXT_TO_LANG[".java"] == "java"


# ------------------------------------------------------------------ #
# Graceful degradation                                                 #
# ------------------------------------------------------------------ #


class TestGracefulDegradation:
    def test_unsupported_language_returns_whole_text(self) -> None:
        blocks = split_code_blocks("some random code", "fortran")
        assert blocks == ["some random code"]

    def test_empty_input(self) -> None:
        blocks = split_code_blocks("", "python")
        assert blocks == [""]

    def test_no_definitions_returns_whole_text(self) -> None:
        code = "# just comments\nx = 1\ny = 2\n"
        blocks = split_code_blocks(code, "python")
        # Should return at least one block
        assert len(blocks) >= 1
        all_text = " ".join(blocks)
        assert "x = 1" in all_text


# ------------------------------------------------------------------ #
# Consistency: tree-sitter vs regex produce same structure             #
# ------------------------------------------------------------------ #


class TestConsistency:
    """Verify tree-sitter and regex agree on basic cases."""

    def test_python_functions_same_count(self) -> None:
        code = "def foo():\n    pass\n\ndef bar():\n    pass\n"
        blocks = split_code_blocks(code, "python")
        assert len(blocks) == 2
        assert "def foo" in blocks[0]
        assert "def bar" in blocks[1]

    def test_python_class_not_split_into_methods(self) -> None:
        code = (
            "class MyClass:\n"
            "    def method_one(self):\n"
            "        pass\n"
            "\n"
            "    def method_two(self):\n"
            "        pass\n"
        )
        blocks = split_code_blocks(code.strip(), "python")
        assert len(blocks) == 1
        assert "class MyClass" in blocks[0]

    def test_python_preamble_preserved(self) -> None:
        code = "import os\nimport sys\n\ndef main():\n    pass\n"
        blocks = split_code_blocks(code, "python")
        assert len(blocks) == 2
        assert "import os" in blocks[0]
        assert "def main" in blocks[1]

    def test_go_preamble_and_funcs(self) -> None:
        code = 'package main\n\nimport "fmt"\n\nfunc main() {\n    fmt.Println("hi")\n}\n\nfunc helper() int {\n    return 1\n}\n'
        blocks = split_code_blocks(code, "go")
        assert len(blocks) == 3
        assert "package main" in blocks[0]
        assert "func main" in blocks[1]
        assert "func helper" in blocks[2]
