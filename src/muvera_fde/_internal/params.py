"""
Per-repetition projection parameter containers for muvera-fde.

This module is not part of the public API.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

from muvera_fde._internal.sketch import (
    low_rank_simhash_factors,
    simhash_matrix,
    srht_params,
)


class RepParams(BaseModel):
    """Precomputed random-projection parameters for one MUVERA repetition.

    Exactly one SimHash mode is active per repetition (when
    ``num_simhash_projections > 0``):

    * Full-rank Gaussian (``simhash_mat`` set): DEFAULT_IDENTITY / AMS_SKETCH
    * Low-rank Gaussian (``simhash_a`` + ``simhash_b`` set): LOW_RANK_GAUSSIAN
    * SRHT (``srht_d_signs`` + ``srht_sample_indices`` set): SRHT

    All other SimHash fields are ``None`` for the inactive modes.

    Attributes
    ----------
    cs_indices:
        Count Sketch bucket index for each input dimension, shape (dimension,).
        ``None`` when ``projection_type`` is not ``AMS_SKETCH``.
    cs_signs:
        Count Sketch +-1 sign for each input dimension, shape (dimension,), dtype float32.
        ``None`` when ``projection_type`` is not ``AMS_SKETCH``.
    simhash_mat:
        Full-rank Gaussian SimHash matrix, shape (projection_dim, k), dtype float32.
        Set for DEFAULT_IDENTITY and AMS_SKETCH (when num_simhash_projections > 0).
        ``None`` for LOW_RANK_GAUSSIAN, SRHT, or when num_simhash_projections == 0.
    simhash_a:
        Low-rank SimHash factor A, shape (projection_dim, rank), dtype float32.
        Set only for LOW_RANK_GAUSSIAN with num_simhash_projections > 0.
    simhash_b:
        Low-rank SimHash factor B, shape (num_simhash_projections, rank), dtype float32.
        Set only for LOW_RANK_GAUSSIAN with num_simhash_projections > 0.
    srht_d_signs:
        Rademacher +-1 diagonal signs for SRHT, shape (padded_dim,), dtype float32.
        Set only for SRHT with num_simhash_projections > 0.
    srht_sample_indices:
        Sorted indices of the k subsampled Hadamard rows, shape (k,), dtype int64.
        Set only for SRHT with num_simhash_projections > 0.
    srht_padded_dim:
        Zero-padding target (next power of 2 >= projection_dim) for SRHT.
        Set only for SRHT with num_simhash_projections > 0.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    cs_indices: np.ndarray | None
    cs_signs: np.ndarray | None
    simhash_mat: np.ndarray | None
    simhash_a: np.ndarray | None = None
    simhash_b: np.ndarray | None = None
    srht_d_signs: np.ndarray | None = None
    srht_sample_indices: np.ndarray | None = None
    srht_padded_dim: int | None = None


def build_rep_params(
    rep_seed: int,
    dimension: int,
    projection_dim: int,
    num_simhash_projections: int,
    use_identity: bool,
    use_low_rank_simhash: bool = False,
    simhash_rank: int = 1,
    use_srht: bool = False,
) -> RepParams:
    """Precompute projection parameters for one repetition.

    Exactly one SimHash variant is built when ``num_simhash_projections > 0``:

    * ``use_srht=True``: SRHT params (d_signs, sample_indices, padded_dim)
    * ``use_low_rank_simhash=True``: low-rank factors (A, B)
    * otherwise: full-rank Gaussian matrix W

    Parameters
    ----------
    rep_seed:
        Per-repetition seed (``config.seed + rep``).
    dimension:
        Input embedding dimension.
    projection_dim:
        Width of each partition slot after optional Count Sketch.
    num_simhash_projections:
        Number of SimHash bits *k*; 0 disables all SimHash.
    use_identity:
        When ``True``, Count Sketch fields are left as ``None``.
    use_low_rank_simhash:
        When ``True``, build LOW_RANK_GAUSSIAN factors instead of full matrix.
    simhash_rank:
        Rank *r* for LOW_RANK_GAUSSIAN.  Ignored otherwise.
    use_srht:
        When ``True``, build SRHT params instead of full matrix.
        Takes precedence over ``use_low_rank_simhash`` if both are set.

    Returns
    -------
    RepParams
    """
    cs_indices = cs_signs = None
    if not use_identity:
        rng = np.random.default_rng(rep_seed)
        cs_indices = rng.integers(0, projection_dim, size=dimension)
        cs_signs = 2.0 * rng.integers(0, 2, size=dimension).astype(np.float32) - 1.0

    simhash_mat = None
    simhash_a = simhash_b = None
    srht_d_signs = srht_sample_indices = None
    srht_padded_dim = None

    if num_simhash_projections > 0:
        if use_srht:
            srht_d_signs, srht_sample_indices, srht_padded_dim = srht_params(
                rep_seed, projection_dim, num_simhash_projections
            )
        elif use_low_rank_simhash:
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
        srht_d_signs=srht_d_signs,
        srht_sample_indices=srht_sample_indices,
        srht_padded_dim=srht_padded_dim,
    )
