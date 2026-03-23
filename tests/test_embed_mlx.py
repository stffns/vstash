"""Tests for vstash.embed — resolve_backend with mocked platform detection."""

from __future__ import annotations

from unittest.mock import patch

from vstash.embed import resolve_backend


class TestResolveBackendAuto:
    """Test resolve_backend('auto') with mocked hardware/library detection."""

    @patch("vstash.embed._mlx_available", return_value=True)
    @patch("vstash.embed._is_apple_silicon", return_value=True)
    def test_auto_returns_mlx_on_apple_silicon_with_mlx(
        self, mock_silicon, mock_mlx
    ) -> None:
        """auto resolves to 'mlx' when on Apple Silicon and mlx is available."""
        assert resolve_backend("auto") == "mlx"

    @patch("vstash.embed._mlx_available", return_value=True)
    @patch("vstash.embed._is_apple_silicon", return_value=False)
    def test_auto_returns_onnx_when_not_apple_silicon(
        self, mock_silicon, mock_mlx
    ) -> None:
        """auto resolves to 'onnx' when not on Apple Silicon."""
        assert resolve_backend("auto") == "onnx"

    @patch("vstash.embed._mlx_available", return_value=False)
    @patch("vstash.embed._is_apple_silicon", return_value=True)
    def test_auto_returns_onnx_when_mlx_unavailable(
        self, mock_silicon, mock_mlx
    ) -> None:
        """auto resolves to 'onnx' when on Apple Silicon but mlx is not installed."""
        assert resolve_backend("auto") == "onnx"


class TestResolveBackendExplicit:
    """Test resolve_backend with explicit backend selection."""

    @patch("vstash.embed._mlx_available", return_value=False)
    @patch("vstash.embed._is_apple_silicon", return_value=False)
    def test_mlx_returns_mlx_regardless(self, mock_silicon, mock_mlx) -> None:
        """Explicit 'mlx' returns 'mlx' even if detection says otherwise."""
        assert resolve_backend("mlx") == "mlx"

    @patch("vstash.embed._mlx_available", return_value=True)
    @patch("vstash.embed._is_apple_silicon", return_value=True)
    def test_onnx_returns_onnx_regardless(self, mock_silicon, mock_mlx) -> None:
        """Explicit 'onnx' returns 'onnx' even if mlx is available."""
        assert resolve_backend("onnx") == "onnx"
