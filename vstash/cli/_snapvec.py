"""``vstash snapvec`` sub-typer (#284): fit.

Defines ``snapvec_app`` and registers it on the shared ``app`` at import time;
``vstash/cli/__init__.py`` imports this module for that side effect.
"""

from __future__ import annotations

import typer

from ._app import _get_store, _profile_from_ctx, _safe_exc, app, console

snapvec_app = typer.Typer(
    name="snapvec",
    help="Manage the snapvec vector backend (IVFPQ index training).",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)
app.add_typer(snapvec_app, name="snapvec")


@snapvec_app.command(name="fit")
def snapvec_fit(
    ctx: typer.Context,
    training_sample: int = typer.Option(
        50_000,
        "--training-sample",
        help="Vectors sampled for IVFPQ codebook training (FAISS rule: >=30 * nlist)",
    ),
) -> None:
    """Train and persist the IVFPQ index from the current corpus.

    Requires ``storage.vector_backend = 'snapvec-ivfpq'`` in vstash.toml.
    Reads every embedding out of vec_chunks, fits the IVF coarse centroids
    + residual PQ codebooks, indexes all rows, and saves the ``.snpi``
    file next to the database. After this completes, searches route
    through the IVFPQ backend with fp16 rerank.
    """
    _, store = _get_store(profile=_profile_from_ctx(ctx))
    try:
        stats = store.fit_ivfpq(training_sample=training_sample)
    except RuntimeError as exc:
        console.print(f"[red]x[/red] {_safe_exc(exc)}")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[green]+[/green] IVFPQ index fit: "
        f"{stats['n_indexed']} vectors indexed, "
        f"nlist={stats['nlist']}, "
        f"training_sample={stats['training_sample']}, "
        f"build={stats['build_seconds']}s"
    )
    console.print(f"  saved to [dim]{stats['path']}[/dim]")
