"""
Core FDE generation for muvera-fde.
"""

from __future__ import annotations

import numpy as np

from muvera_fde._internal.params import RepParams, build_rep_params
from muvera_fde._internal.sketch import (
    apply_cross_polytope,
    apply_srht,
    count_sketch,
    densifying_fill,
    simhash_partition_indices,
)
from muvera_fde._internal.validation import (
    _projection_dim_for_config,
    checked_intermediate_fde_length,
    num_partitions_for_config,
    prepare_embeddings,
    validate_config,
)
from muvera_fde.config import FDEConfig, ProjectionType

# ---------------------------------------------------------------------------
# Config-derived helpers (shared with encoder.py)
# ---------------------------------------------------------------------------


def _use_identity(config: FDEConfig) -> bool:
    return config.projection_type != ProjectionType.AMS_SKETCH


def _use_low_rank_simhash(config: FDEConfig) -> bool:
    return config.projection_type == ProjectionType.LOW_RANK_GAUSSIAN


def _use_srht(config: FDEConfig) -> bool:
    return config.projection_type == ProjectionType.SRHT


def _use_cross_polytope(config: FDEConfig) -> bool:
    return config.projection_type == ProjectionType.CROSS_POLYTOPE


def _use_densifying(config: FDEConfig) -> bool:
    """Return True when densifying fill should be used instead of Hamming fill."""
    return _use_cross_polytope(config) or config.densifying_fill


# ---------------------------------------------------------------------------
# Shared projection + partition
# ---------------------------------------------------------------------------


def _project_and_partition(
    embedding_matrix: np.ndarray,
    rep_params: RepParams,
    use_identity: bool,
    projection_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Apply optional Count Sketch and SimHash/Cross-Polytope partitioning.

    Five paths:

    * Full-rank Gaussian (simhash_mat):    sign(projected @ W)
    * Low-rank Gaussian  (simhash_a/b):    sign((projected @ A) @ B.T)
    * SRHT               (srht fields):    sign(S H D projected)
    * Cross-Polytope     (cp fields):      argmax(|H D projected|) + sign bit
    * No partitioning    (all None):       single partition

    Returns
    -------
    projected       : (N, projection_dim)
    partition_indices : (N,) int32
    sketch_matrix   : (N, k) or None  -- None for Cross-Polytope and no-SimHash
    """
    num_points = embedding_matrix.shape[0]
    if use_identity:
        projected = embedding_matrix
    else:
        assert rep_params.cs_indices is not None and rep_params.cs_signs is not None
        projected = np.zeros((num_points, projection_dim), dtype=np.float32)
        np.add.at(projected.T, rep_params.cs_indices, (embedding_matrix * rep_params.cs_signs).T)

    sketch_matrix: np.ndarray | None

    if rep_params.cp_d_signs is not None:
        # Cross-Polytope: argmax-based partition, no sketch matrix
        assert rep_params.cp_padded_dim is not None
        partition_indices = apply_cross_polytope(
            projected, rep_params.cp_d_signs, rep_params.cp_padded_dim
        )
        return projected, partition_indices, None

    if rep_params.simhash_mat is not None:
        sketch_matrix = projected @ rep_params.simhash_mat
    elif rep_params.simhash_a is not None:
        assert rep_params.simhash_b is not None
        sketch_matrix = (projected @ rep_params.simhash_a) @ rep_params.simhash_b.T
    elif rep_params.srht_d_signs is not None:
        assert rep_params.srht_sample_indices is not None
        assert rep_params.srht_padded_dim is not None
        sketch_matrix = apply_srht(
            projected,
            rep_params.srht_d_signs,
            rep_params.srht_sample_indices,
            rep_params.srht_padded_dim,
        )
    else:
        sketch_matrix = None

    partition_indices = (
        simhash_partition_indices(sketch_matrix)
        if sketch_matrix is not None
        else np.zeros(num_points, dtype=np.int32)
    )
    return projected, partition_indices, sketch_matrix


# ---------------------------------------------------------------------------
# Empty-partition fill
# ---------------------------------------------------------------------------


def _hamming_fill(
    rep_slice: np.ndarray,
    projected: np.ndarray,
    empty_pidxs: np.ndarray,
    signs_rev: np.ndarray,
    k: int,
) -> None:
    """Fill empty slots with nearest token by SimHash Hamming distance. O(N*k*empty)."""
    empty_binary = empty_pidxs.copy()
    empty_binary ^= empty_binary >> 1
    empty_binary ^= empty_binary >> 2
    empty_binary ^= empty_binary >> 4
    empty_binary ^= empty_binary >> 8
    empty_binary ^= empty_binary >> 16

    bit_positions = np.arange(k)[np.newaxis, :]
    num_points = signs_rev.shape[0]
    if num_points == 0:
        return

    batch_size = max(1, (1 << 20) // max(1, num_points))
    for start in range(0, len(empty_pidxs), batch_size):
        stop = min(start + batch_size, len(empty_pidxs))
        batch_pidxs = empty_pidxs[start:stop]
        batch_binary = empty_binary[start:stop]
        batch_bits = ((batch_binary[:, np.newaxis] >> bit_positions) & 1).astype(np.int32)
        batch_distances = (batch_bits[:, np.newaxis, :] != signs_rev[np.newaxis, :, :]).sum(axis=2)
        nearest = np.argmin(batch_distances, axis=1)
        rep_slice[batch_pidxs] = projected[nearest]


def _normalize_and_fill_rep(
    rep_slice: np.ndarray,
    partition_indices: np.ndarray,
    projected: np.ndarray,
    sketch_matrix: np.ndarray | None,
    config: FDEConfig,
    num_partitions: int,
    rep_seed: int,
) -> None:
    """Normalize partition sums to centroids; fill empty slots if configured."""
    partition_sizes = np.bincount(partition_indices, minlength=num_partitions).astype(
        np.float32, copy=False
    )
    filled_mask = partition_sizes > 0
    rep_slice[filled_mask] /= partition_sizes[filled_mask, np.newaxis]

    if not config.fill_empty_partitions:
        return

    empty_pidxs = np.nonzero(~filled_mask)[0]
    if len(empty_pidxs) == 0:
        return

    if _use_densifying(config):
        # Densifying LSH: O(num_empty), hash-based, no sketch_matrix needed.
        # Always used for CROSS_POLYTOPE; optional for other modes via densifying_fill=True.
        densifying_fill(rep_slice, projected, empty_pidxs, rep_seed)
    elif sketch_matrix is not None:
        # Hamming nearest-neighbour fill: O(N*k), more geometrically precise.
        signs_rev = (sketch_matrix[:, ::-1] > 0).astype(np.int32)
        _hamming_fill(rep_slice, projected, empty_pidxs, signs_rev, config.num_simhash_projections)


def _maybe_count_sketch(out: np.ndarray, config: FDEConfig) -> np.ndarray:
    if config.final_projection_dimension is not None:
        return count_sketch(out, config.final_projection_dimension, config.seed)
    return out


# ---------------------------------------------------------------------------
# Public generation functions
# ---------------------------------------------------------------------------


def generate_query_fde(
    point_cloud: np.ndarray,
    config: FDEConfig,
    rep_params_list: list[RepParams] | None = None,
) -> np.ndarray:
    """Generate a query-side FDE (SUM aggregation).

    Parameters
    ----------
    point_cloud     : (num_tokens, dimension) or flat 1-D
    config          : encoding configuration; fill_empty_partitions must be False
    rep_params_list : precomputed parameters; built on the fly when None

    Returns
    -------
    np.ndarray, shape (fde_dimension,), dtype float32
    """
    validate_config(config)
    if config.fill_empty_partitions:
        raise ValueError("Query FDE does not support fill_empty_partitions.")

    embedding_matrix = prepare_embeddings(point_cloud, config)
    use_id = _use_identity(config)
    projection_dim = _projection_dim_for_config(config)
    num_partitions = num_partitions_for_config(config)

    out = np.zeros(
        checked_intermediate_fde_length(config, projection_dim, num_partitions), dtype=np.float32
    )

    for rep in range(config.num_repetitions):
        params = (
            rep_params_list[rep]
            if rep_params_list is not None
            else build_rep_params(
                config.seed + rep,
                config.dimension,
                projection_dim,
                config.num_simhash_projections,
                use_id,
                use_low_rank_simhash=_use_low_rank_simhash(config),
                simhash_rank=config.simhash_rank,
                use_srht=_use_srht(config),
                use_cross_polytope=_use_cross_polytope(config),
            )
        )
        projected, partition_indices, _ = _project_and_partition(
            embedding_matrix, params, use_id, projection_dim
        )
        rep_offset = rep * num_partitions * projection_dim
        rep_slice = out[rep_offset : rep_offset + num_partitions * projection_dim].reshape(
            num_partitions, projection_dim
        )
        np.add.at(rep_slice, partition_indices, projected)

    return _maybe_count_sketch(out, config)


def generate_document_fde(
    point_cloud: np.ndarray,
    config: FDEConfig,
    rep_params_list: list[RepParams] | None = None,
) -> np.ndarray:
    """Generate a document-side FDE (AVERAGE aggregation).

    Parameters
    ----------
    point_cloud     : (num_tokens, dimension) or flat 1-D
    config          : encoding configuration
    rep_params_list : precomputed parameters; built on the fly when None

    Returns
    -------
    np.ndarray, shape (fde_dimension,), dtype float32
    """
    validate_config(config)
    embedding_matrix = prepare_embeddings(point_cloud, config)
    use_id = _use_identity(config)
    projection_dim = _projection_dim_for_config(config)
    num_partitions = num_partitions_for_config(config)

    out = np.zeros(
        checked_intermediate_fde_length(config, projection_dim, num_partitions), dtype=np.float32
    )

    for rep in range(config.num_repetitions):
        params = (
            rep_params_list[rep]
            if rep_params_list is not None
            else build_rep_params(
                config.seed + rep,
                config.dimension,
                projection_dim,
                config.num_simhash_projections,
                use_id,
                use_low_rank_simhash=_use_low_rank_simhash(config),
                simhash_rank=config.simhash_rank,
                use_srht=_use_srht(config),
                use_cross_polytope=_use_cross_polytope(config),
            )
        )
        projected, partition_indices, sketch_matrix = _project_and_partition(
            embedding_matrix, params, use_id, projection_dim
        )
        rep_offset = rep * num_partitions * projection_dim
        rep_slice = out[rep_offset : rep_offset + num_partitions * projection_dim].reshape(
            num_partitions, projection_dim
        )
        np.add.at(rep_slice, partition_indices, projected)
        _normalize_and_fill_rep(
            rep_slice,
            partition_indices,
            projected,
            sketch_matrix,
            config,
            num_partitions,
            config.seed + rep,
        )

    return _maybe_count_sketch(out, config)
