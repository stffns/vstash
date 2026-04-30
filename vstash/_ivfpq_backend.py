"""Compatibility shim. The real module lives at
:mod:`vstash.vectorbackend.snapvec_ivfpq` since v0.37.

Importing from here continues to work but is deprecated. New code
should use::

    from vstash.vectorbackend import IVFPQBackend
    # or
    from vstash.vectorbackend.snapvec_ivfpq import IVFPQBackend

This shim will be removed in v0.40.
"""

from __future__ import annotations

import warnings

from .vectorbackend.snapvec_ivfpq import IVFPQBackend

warnings.warn(
    "vstash._ivfpq_backend is deprecated; import IVFPQBackend from "
    "vstash.vectorbackend (or vstash.vectorbackend.snapvec_ivfpq). "
    "This shim will be removed in v0.40.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["IVFPQBackend"]
