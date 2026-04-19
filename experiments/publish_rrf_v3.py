"""
publish_rrf_v3.py -- upload the arm_vol-trained v3 model to HF Hub.

One-shot script. Run it once the model artifact is locally
accessible (either downloaded from Colab or retrained via
``vstash retrain-multi``). Requires:

- The ``huggingface_hub`` package: ``pip install huggingface_hub``.
- A Hugging Face token with ``write`` permission to the target
  namespace. Set via env var (``HF_TOKEN=hf_...``) or the
  ``--token`` flag. Do not commit tokens.
- The model directory from the training run. Default layout
  matches ``vstash retrain-multi`` output: ``.../bge-small-rrf-v3/``
  containing ``config_sentence_transformers.json``,
  ``model.safetensors`` (or ``pytorch_model.bin``), tokenizer
  files, and ``training_meta.json``.

Typical flow:

    # 1. From Colab, after arm_vol completed:
    #    Tools -> Files -> right-click on /content/retrained_hr9_arm_vol
    #    -> Download (zip). Unzip locally.

    # 2. Locally:
    export HF_TOKEN=hf_xxxxxxxxxxxx
    python -m experiments.publish_rrf_v3 \\
        --model-dir ~/Downloads/retrained_hr9_arm_vol \\
        --repo-id Stffens/bge-small-rrf-v3

The script:

1. Verifies the model dir looks right (required files present).
2. Copies ``experiments/rrf_v3_model_card.md`` into the dir as
   ``README.md`` so the HF Hub page renders the card.
3. Creates the repo if it does not exist.
4. Uploads the whole dir via ``HfApi.upload_folder``.
5. Prints the public URL.

No credentials are written or echoed. Run with ``--dry-run`` to
stop after the local validation step (no upload).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


_REQUIRED_FILES = {
    "config_sentence_transformers.json",
    "modules.json",
    "tokenizer.json",
}
# Model weights may appear under any of these names depending on the
# sentence-transformers version used for training.
_WEIGHT_CANDIDATES = {
    "model.safetensors",
    "pytorch_model.bin",
}


def _validate_model_dir(model_dir: Path) -> None:
    missing = [f for f in _REQUIRED_FILES if not (model_dir / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{model_dir} is missing required files: {missing}. "
            "Make sure you are pointing at the retrain-multi output "
            "directory (not its parent, not the .candidate subdir)."
        )
    if not any((model_dir / w).is_file() for w in _WEIGHT_CANDIDATES):
        raise FileNotFoundError(
            f"{model_dir} has no recognised weights file. Expected one of: {_WEIGHT_CANDIDATES}."
        )
    meta_path = model_dir / "training_meta.json"
    if not meta_path.is_file():
        print(
            f"warning: {meta_path} not found; the model card's training "
            "metadata section will look empty.",
            file=sys.stderr,
        )


def _copy_model_card(model_dir: Path, card_src: Path) -> Path:
    if not card_src.is_file():
        raise FileNotFoundError(f"model card not found: {card_src}")
    readme_dst = model_dir / "README.md"
    shutil.copy2(card_src, readme_dst)
    print(f"copied model card -> {readme_dst}")
    return readme_dst


def _upload(
    model_dir: Path,
    repo_id: str,
    token: str,
    private: bool,
    commit_message: str,
) -> str:
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        ) from exc

    api = HfApi(token=token)

    # Create the repo if needed. ``exist_ok=True`` makes this
    # idempotent across re-runs.
    create_repo(repo_id, token=token, private=private, exist_ok=True)

    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        commit_message=commit_message,
        token=token,
    )
    url = f"https://huggingface.co/{repo_id}"
    return url


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to the trained model directory (the one that holds "
        "config_sentence_transformers.json and the weights file).",
    )
    parser.add_argument(
        "--repo-id",
        default="Stffens/bge-small-rrf-v3",
        help="Target Hugging Face repo, ``user/name`` form.",
    )
    parser.add_argument(
        "--card",
        type=Path,
        default=Path("experiments/rrf_v3_model_card.md"),
        help="Path to the model card markdown. Copied into the model "
        "dir as README.md before upload.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face write token. Falls back to $HF_TOKEN.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo private (default: public).",
    )
    parser.add_argument(
        "--commit-message",
        default="upload bge-small-rrf-v3 (H-R9 arm_vol: temperature=0.5, total_triples=60000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stop after local validation; do not push.",
    )
    args = parser.parse_args()

    model_dir: Path = args.model_dir.expanduser().resolve()
    if not model_dir.is_dir():
        print(f"error: {model_dir} is not a directory", file=sys.stderr)
        return 1

    try:
        _validate_model_dir(model_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _copy_model_card(model_dir, args.card.expanduser().resolve())

    if args.dry_run:
        print("dry-run: validation + card copy done; skipping upload")
        return 0

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print(
            "error: no HF token provided. Set $HF_TOKEN or pass --token.",
            file=sys.stderr,
        )
        return 1

    url = _upload(
        model_dir=model_dir,
        repo_id=args.repo_id,
        token=token,
        private=args.private,
        commit_message=args.commit_message,
    )
    print(f"\npublished: {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
