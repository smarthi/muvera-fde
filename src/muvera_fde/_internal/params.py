"""
Per-repetition projection parameter containers for muvera-fde.

This module is not part of the public API.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

from muvera_fde._internal.sketch import (
    cross_polytope_params,
    low_rank_simhash_factors,
    simhash_matrix,
    srht_params,
)


class RepParams(BaseModel):
    """Precomputed random-projection parameters for one MUVERA repetition.

    Exactly one SimHash mode is active per repetition:

    * Full-rank Gaussian  (simhash_mat set):              DEFAULT_IDENTITY / AMS_SKETCH
    * Low-rank Gaussian   (simhash_a + simhash_b set):    LOW_RANK_GAUSSIAN
    * SRHT                (srht_d_signs + indices set):   SRHT
    * Cross-Polytope      (cp_d_signs set):               CROSS_POLYTOPE

    All other SimHash fields are None for the inactive modes.
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
    # Cross-Polytope fields
    cp_d_signs: np.ndarray | None = None
    cp_padded_dim: int | None = None


def build_rep_params(
    rep_seed: int,
    dimension: int,
    projection_dim: int,
    num_simhash_projections: int,
    use_identity: bool,
    use_low_rank_simhash: bool = False,
    simhash_rank: int = 1,
    use_srht: bool = False,
    use_cross_polytope: bool = False,
) -> RepParams:
    """Precompute projection parameters for one repetition.

    Priority order (when multiple flags could be set):
        CROSS_POLYTOPE > SRHT > LOW_RANK_GAUSSIAN > DEFAULT_IDENTITY

    Parameters
    ----------
    rep_seed             : per-repetition seed (config.seed + rep)
    dimension            : input embedding dimension
    projection_dim       : slot width after optional Count Sketch
    num_simhash_projections : k; 0 disables SimHash (ignored for CROSS_POLYTOPE)
    use_identity         : True -> skip Count Sketch
    use_low_rank_simhash : True -> LOW_RANK_GAUSSIAN factors
    simhash_rank         : r for LOW_RANK_GAUSSIAN
    use_srht             : True -> SRHT params
    use_cross_polytope   : True -> Cross-Polytope params
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
    cp_d_signs = None
    cp_padded_dim = None

    if use_cross_polytope:
        # Cross-Polytope: full rotation, no subsampling, num_simhash_projections ignored
        cp_d_signs, cp_padded_dim = cross_polytope_params(rep_seed, projection_dim)
    elif num_simhash_projections > 0:
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
        cp_d_signs=cp_d_signs,
        cp_padded_dim=cp_padded_dim,
    )
