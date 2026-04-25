"""
Per-repetition projection parameter containers for muvera-fde.

This module is not part of the public API.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

from muvera_fde._internal.sketch import low_rank_simhash_factors, simhash_matrix


class RepParams(BaseModel):
    """Precomputed random-projection parameters for one MUVERA repetition.

    Built once at encoder initialisation and reused across every encode call,
    eliminating repeated RNG sampling and matrix allocation in the hot loop.

    Attributes
    ----------
    cs_indices:
        Count Sketch bucket index for each input dimension, shape (dimension,).
        ``None`` when ``projection_type`` is not ``AMS_SKETCH``.
    cs_signs:
        Count Sketch +-1 sign for each input dimension, shape (dimension,), dtype float32.
        ``None`` when ``projection_type`` is not ``AMS_SKETCH``.
    simhash_mat:
        Full-rank Gaussian SimHash projection matrix, shape (projection_dim, k), dtype float32.
        Set for ``DEFAULT_IDENTITY`` and ``AMS_SKETCH``.
        ``None`` when ``num_simhash_projections == 0`` or when using ``LOW_RANK_GAUSSIAN``.
    simhash_a:
        Low-rank SimHash factor A, shape (projection_dim, rank), dtype float32.
        Set only when ``projection_type`` is ``LOW_RANK_GAUSSIAN`` and
        ``num_simhash_projections > 0``.  ``None`` otherwise.
    simhash_b:
        Low-rank SimHash factor B, shape (num_simhash_projections, rank), dtype float32.
        Set only when ``projection_type`` is ``LOW_RANK_GAUSSIAN`` and
        ``num_simhash_projections > 0``.  ``None`` otherwise.

    Notes
    -----
    Exactly one of ``simhash_mat`` or ``(simhash_a, simhash_b)`` is set (non-None)
    per repetition when ``num_simhash_projections > 0``.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    cs_indices: np.ndarray | None
    cs_signs: np.ndarray | None
    simhash_mat: np.ndarray | None
    simhash_a: np.ndarray | None = None
    simhash_b: np.ndarray | None = None


def build_rep_params(
    rep_seed: int,
    dimension: int,
    projection_dim: int,
    num_simhash_projections: int,
    use_identity: bool,
    use_low_rank_simhash: bool = False,
    simhash_rank: int = 1,
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
        When ``True`` the Count Sketch fields (cs_indices, cs_signs) are
        left as ``None``.
    use_low_rank_simhash:
        When ``True``, build low-rank SimHash factors (A, B) instead of the
        full (projection_dim x k) matrix.  Requires ``num_simhash_projections > 0``
        and ``simhash_rank >= 1``.
    simhash_rank:
        Rank *r* for the low-rank SimHash factorisation.  Only used when
        ``use_low_rank_simhash=True``.

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

    simhash_mat = None
    simhash_a = None
    simhash_b = None

    if num_simhash_projections > 0:
        if use_low_rank_simhash:
            simhash_a, simhash_b = low_rank_simhash_factors(
                rep_seed, projection_dim, num_simhash_projections, simhash_rank
            )
        else:
            simhash_mat = simhash_matrix(rep_seed, projection_dim, num_simhash_projections)

    return RepParams(
        cs_indices=cs_indices,
        cs_signs=cs_signs,
        simhash_mat=simhash_mat,
        simhash_a=simhash_a,
        simhash_b=simhash_b,
    )
