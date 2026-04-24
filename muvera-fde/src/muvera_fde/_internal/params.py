"""
Per-repetition projection parameter containers for muvera-fde.

This module is not part of the public API.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict

from muvera_fde._internal.sketch import simhash_matrix


class RepParams(BaseModel):
    """Precomputed random-projection parameters for one MUVERA repetition.

    Built once at encoder initialisation and reused across every encode call,
    eliminating repeated RNG sampling and matrix allocation in the hot loop.

    Attributes
    ----------
    cs_indices:
        Count Sketch bucket index for each input dimension, shape (dimension,).
        ``None`` when ``projection_type`` is ``DEFAULT_IDENTITY``.
    cs_signs:
        Count Sketch ±1 sign for each input dimension, shape (dimension,), dtype float32.
        ``None`` when ``projection_type`` is ``DEFAULT_IDENTITY``.
    simhash_mat:
        Gaussian SimHash projection matrix, shape (projection_dim, k), dtype float32.
        ``None`` when ``num_simhash_projections == 0``.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    cs_indices: Optional[np.ndarray]
    cs_signs: Optional[np.ndarray]
    simhash_mat: Optional[np.ndarray]


def build_rep_params(
    rep_seed: int,
    dimension: int,
    projection_dim: int,
    num_simhash_projections: int,
    use_identity: bool,
) -> RepParams:
    """Precompute Count Sketch and SimHash parameters for one repetition.

    Parameters
    ----------
    rep_seed:
        Per-repetition seed (``config.seed + rep``).
    dimension:
        Input embedding dimension.
    projection_dim:
        Width of each partition slot (``dimension`` for identity projection,
        ``config.projection_dimension`` for Count Sketch).
    num_simhash_projections:
        Number of SimHash bits *k*; 0 disables SimHash (single partition).
    use_identity:
        When ``True`` the Count Sketch fields are left as ``None``.

    Returns
    -------
    RepParams
        Immutable container of precomputed projection parameters.
    """
    cs_indices = cs_signs = None
    if not use_identity:
        rng = np.random.default_rng(rep_seed)
        cs_indices = rng.integers(0, projection_dim, size=dimension)
        cs_signs = 2.0 * rng.integers(0, 2, size=dimension).astype(np.float32) - 1.0

    mat = (
        simhash_matrix(rep_seed, projection_dim, num_simhash_projections)
        if num_simhash_projections > 0
        else None
    )
    return RepParams(cs_indices=cs_indices, cs_signs=cs_signs, simhash_mat=mat)
