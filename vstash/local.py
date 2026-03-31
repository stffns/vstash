"""
local.py — Managed llama-server subprocess for fully local inference.

Handles the full lifecycle:
  - Model download from Hugging Face Hub (~/.vstash/models/)
  - llama-server binary detection
  - Process start with hardware-optimal flags
  - Health check polling
  - Clean shutdown via atexit

Usage:
    server = LocalServer(cfg.local)
    base_url = server.ensure_running()  # idempotent — safe to call multiple times
"""

from __future__ import annotations

import atexit
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

from .config import LocalConfig

_HEALTH_POLL_INTERVAL = 0.5  # seconds
_HEALTH_TIMEOUT = 60  # seconds to wait for server to be ready


class LocalServerError(RuntimeError):
    pass


class LocalServer:
    """Manages a llama-server subprocess for vstash local inference."""

    def __init__(self, cfg: LocalConfig) -> None:
        self._cfg = cfg
        self._proc: subprocess.Popen | None = None  # type: ignore[type-arg]

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def ensure_running(self) -> str:
        """Ensure a llama-server is running. Returns the base_url.

        Idempotent — safe to call on every request. If a server is already
        reachable on the configured port it is reused, even if started by a
        previous process.
        """
        if self._is_healthy():
            return self._base_url

        model_path = self._ensure_model()
        binary = self._find_binary()
        self._start(binary, model_path)
        return self._base_url

    def shutdown(self) -> None:
        """Terminate the managed server process, if we started it."""
        pid_file = self._pid_file
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                pass
            self._proc = None

        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, ValueError):
                pass
            pid_file.unlink(missing_ok=True)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @property
    def _base_url(self) -> str:
        return f"http://127.0.0.1:{self._cfg.port}/v1"

    @property
    def _pid_file(self) -> Path:
        return Path(self._cfg.models_dir).expanduser() / "server.pid"

    def _is_healthy(self) -> bool:
        """Return True if a server is already responding on our port."""
        try:
            r = httpx.get(
                f"http://127.0.0.1:{self._cfg.port}/health",
                timeout=1.0,
            )
            return r.status_code == 200
        except Exception:
            return False

    def _ensure_model(self) -> Path:
        """Download model GGUF if not already on disk. Returns local path."""
        models_dir = Path(self._cfg.models_dir).expanduser()
        models_dir.mkdir(parents=True, exist_ok=True)

        dest = models_dir / self._cfg.model_file
        if dest.exists():
            return dest

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise LocalServerError(
                "huggingface_hub is required for model download.\n"
                "Install it with: pip install huggingface-hub"
            ) from exc

        print(
            f"[vstash] Downloading {self._cfg.model_file} from "
            f"{self._cfg.model_repo} (~{self._cfg.model_size_hint} GB, one-time)…",
            file=sys.stderr,
        )

        downloaded = hf_hub_download(
            repo_id=self._cfg.model_repo,
            filename=self._cfg.model_file,
            local_dir=str(models_dir),
        )
        return Path(downloaded)

    def _find_binary(self) -> Path:
        """Locate llama-server binary. Raises with install instructions if missing."""
        cfg_path = self._cfg.llama_server_path
        if cfg_path:
            p = Path(cfg_path).expanduser()
            if p.exists():
                return p
            raise LocalServerError(
                f"llama_server_path configured as '{cfg_path}' but not found."
            )

        # Check PATH
        which = shutil.which("llama-server")
        if which:
            return Path(which)

        # Check ~/.vstash/bin/
        local_bin = Path.home() / ".vstash" / "bin" / "llama-server"
        if local_bin.exists():
            return local_bin

        raise LocalServerError(
            "llama-server binary not found.\n\n"
            "Options:\n"
            "  1. Build llama.cpp and add to PATH:\n"
            "     https://github.com/TheTom/llama-cpp-turboquant\n"
            "  2. Set llama_server_path in [local] section of vstash.toml:\n"
            "     llama_server_path = '~/projects/llama-cpp/build/bin/llama-server'\n"
            "  3. Copy the binary to ~/.vstash/bin/llama-server\n"
        )

    def _detect_gpu_layers(self) -> int:
        """Return optimal -ngl value for this machine."""
        if self._cfg.gpu_layers != "auto":
            return int(self._cfg.gpu_layers)

        # Apple Silicon — offload everything
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            return 99

        # Check for NVIDIA/AMD GPU via environment hints
        if os.environ.get("CUDA_VISIBLE_DEVICES") or shutil.which("nvidia-smi"):
            return 99

        # CPU-only fallback
        return 0

    def _start(self, binary: Path, model: Path) -> None:
        """Start llama-server and wait until healthy."""
        ngl = self._detect_gpu_layers()
        cache_type = self._cfg.cache_type

        cmd = [
            str(binary),
            "-m", str(model),
            "--cache-type-k", cache_type,
            "--cache-type-v", cache_type,
            "-ngl", str(ngl),
            "-c", str(self._cfg.context),
            "-fa", "on",
            "--host", "127.0.0.1",
            "--port", str(self._cfg.port),
            "-np", str(self._cfg.n_parallel),
        ]

        print(
            f"[vstash] Starting local inference server (port {self._cfg.port})…",
            file=sys.stderr,
        )

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        pid_file = self._pid_file
        pid_file.write_text(str(self._proc.pid))
        atexit.register(self.shutdown)

        # Poll until healthy or timeout
        deadline = time.monotonic() + _HEALTH_TIMEOUT
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                pid_file.unlink(missing_ok=True)
                raise LocalServerError(
                    f"llama-server exited unexpectedly (exit code {self._proc.returncode}).\n"
                    "Check that the model file is valid and the binary supports turbo quants."
                )
            if self._is_healthy():
                print("[vstash] Local server ready.", file=sys.stderr)
                return
            time.sleep(_HEALTH_POLL_INTERVAL)

        self.shutdown()
        raise LocalServerError(
            f"llama-server did not become healthy within {_HEALTH_TIMEOUT}s."
        )
