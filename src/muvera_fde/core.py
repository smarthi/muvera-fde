"""
Core FDE generation for muvera-fde.

``generate_query_fde`` and ``generate_document_fde`` are the primary
computational workhorses.  The :class:`~muvera_fde.encoder.MUVERAEncoder`
high-level class wraps them with pre-built parameter caches.

This module is part of the public API.
"""

from __future__ import annotations

import numpy as np

from muvera_fde._internal.params import RepParams, build_rep_params
from muvera_fde._internal.sketch import apply_srht, count_sketch, simhash_partition_indices
from muvera_fde._internal.validation import (
    checked_intermediate_fde_length,
    prepare_embeddings,
    validate_config,
)
from muvera_fde.config import FDEConfig, ProjectionType

# ---------------------------------------------------------------------------
# Shared projection + partition helper
# ---------------------------------------------------------------------------


def _project_and_partition(
    embedding_matrix: np.ndarray,
    rep_params: RepParams,
    use_identity: bool,
    projection_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Apply optional Count Sketch and SimHash partitioning for one repetition.

    Four SimHash paths are supported:

    * **Full-rank Gaussian** (``simhash_mat`` set):
      ``sketch = projected @ W``
      Cost: O(N x projection_dim x k).

    * **Low-rank Gaussian** (``simhash_a`` / ``simhash_b`` set):
      ``sketch = (projected @ A) @ B.T``
      Cost: O(N x d x r + N x r x k).
      Convergence: O(r^-1) to full-rank (EGGROLL Theorem 4).

    * **SRHT** (``srht_d_signs`` / ``srht_sample_indices`` set):
      ``sketch = S H D projected``  (pad -> D -> FWHT -> subsample)
      Cost: O(N x d x log(d)).
      Guarantee: full JL, no rank approximation.

    * **No SimHash** (all None): single partition, trivial aggregation.

    Returns
    -------
    projected : np.ndarray, shape (N, projection_dim)
    partition_indices : np.ndarray, shape (N,), dtype int32
    sketch_matrix : np.ndarray of shape (N, k) or None
        Retained for nearest-neighbour empty-partition fill.
    """
    num_points = embedding_matrix.shape[0]
    if use_identity:
        projected = embedding_matrix
    else:
        assert rep_params.cs_indices is not None and rep_params.cs_signs is not None
        projected = np.zeros((num_points, projection_dim), dtype=np.float32)
        np.add.at(projected.T, rep_params.cs_indices, (embedding_matrix * rep_params.cs_signs).T)

    # --- SimHash: four paths ---
    sketch_matrix: np.ndarray | None
    if rep_params.simhash_mat is not None:
        # Full-rank Gaussian: O(N x d x k)
        sketch_matrix = projected @ rep_params.simhash_mat
    elif rep_params.simhash_a is not None:
        # Low-rank decomposed: O(N x d x r + N x r x k)
        assert rep_params.simhash_b is not None
        sketch_matrix = (projected @ rep_params.simhash_a) @ rep_params.simhash_b.T
    elif rep_params.srht_d_signs is not None:
        # SRHT: pad -> D -> FWHT -> subsample -- O(N x d x log(d))
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
# Empty-partition fill (document side only)
# ---------------------------------------------------------------------------


def _fill_empty_partitions(
    rep_slice: np.ndarray,
    projected: np.ndarray,
    empty_pidxs: np.ndarray,
    signs_rev: np.ndarray,
    k: int,
) -> None:
    """Fill empty partition slots with the nearest token by SimHash Hamming distance."""
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
) -> None:
    """Normalize partition sums to centroids; optionally fill empty slots."""
    partition_sizes = np.bincount(partition_indices, minlength=num_partitions).astype(
        np.float32, copy=False
    )
    filled_mask = partition_sizes > 0
    rep_slice[filled_mask] /= partition_sizes[filled_mask, np.newaxis]

    if config.fill_empty_partitions and sketch_matrix is not None:
        empty_pidxs = np.nonzero(~filled_mask)[0]
        if len(empty_pidxs) > 0:
            signs_rev = (sketch_matrix[:, ::-1] > 0).astype(np.int32)
            _fill_empty_partitions(
                rep_slice, projected, empty_pidxs, signs_rev, config.num_simhash_projections
            )


def _maybe_count_sketch(out: np.ndarray, config: FDEConfig) -> np.ndarray:
    if config.final_projection_dimension is not None:
        return count_sketch(out, config.final_projection_dimension, config.seed)
    return out


# ---------------------------------------------------------------------------
# Config-derived helpers (shared with encoder.py)
# ---------------------------------------------------------------------------


def _projection_dim_for(config: FDEConfig) -> int:
    """Per-partition slot width for the given config."""
    if config.projection_type == ProjectionType.AMS_SKETCH:
        assert config.projection_dimension is not None
        return config.projection_dimension
    return config.dimension


def _use_identity(config: FDEConfig) -> bool:
    """True when token embeddings bypass Count Sketch."""
    return config.projection_type != ProjectionType.AMS_SKETCH


def _use_low_rank_simhash(config: FDEConfig) -> bool:
    return config.projection_type == ProjectionType.LOW_RANK_GAUSSIAN


def _use_srht(config: FDEConfig) -> bool:
    return config.projection_type == ProjectionType.SRHT


# ---------------------------------------------------------------------------
# Public generation functions
# ---------------------------------------------------------------------------


def generate_query_fde(
    point_cloud: np.ndarray,
    config: FDEConfig,
    rep_params_list: list[RepParams] | None = None,
) -> np.ndarray:
    """Generate a query-side Fixed Dimensional Encoding (SUM aggregation).

    Parameters
    ----------
    point_cloud:
        Query token embeddings, shape ``(num_tokens, dimension)`` or flat 1-D.
    config:
        Encoding configuration.  ``fill_empty_partitions`` must be ``False``.
    rep_params_list:
        Precomputed per-repetition parameters.  Built on the fly when ``None``.

    Returns
    -------
    np.ndarray, shape ``(fde_dimension,)``, dtype float32
    """
    validate_config(config)
    if config.fill_empty_partitions:
        raise ValueError("Query FDE does not support fill_empty_partitions.")

    embedding_matrix = prepare_embeddings(point_cloud, config)
    use_id = _use_identity(config)
    use_lr = _use_low_rank_simhash(config)
    use_sh = _use_srht(config)
    projection_dim = _projection_dim_for(config)

    num_partitions = 1 << config.num_simhash_projections
    out = np.zeros(checked_intermediate_fde_length(config, projection_dim), dtype=np.float32)

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
                use_low_rank_simhash=use_lr,
                simhash_rank=config.simhash_rank,
                use_srht=use_sh,
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
    """Generate a document-side Fixed Dimensional Encoding (AVERAGE aggregation).

    Parameters
    ----------
    point_cloud:
        Document token embeddings, shape ``(num_tokens, dimension)`` or flat 1-D.
    config:
        Encoding configuration.
    rep_params_list:
        Precomputed per-repetition parameters.  Built on the fly when ``None``.

    Returns
    -------
    np.ndarray, shape ``(fde_dimension,)``, dtype float32
    """
    validate_config(config)
    embedding_matrix = prepare_embeddings(point_cloud, config)
    use_id = _use_identity(config)
    use_lr = _use_low_rank_simhash(config)
    use_sh = _use_srht(config)
    projection_dim = _projection_dim_for(config)

    num_partitions = 1 << config.num_simhash_projections
    out = np.zeros(checked_intermediate_fde_length(config, projection_dim), dtype=np.float32)

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
                use_low_rank_simhash=use_lr,
                simhash_rank=config.simhash_rank,
                use_srht=use_sh,
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
            rep_slice, partition_indices, projected, sketch_matrix, config, num_partitions
        )

    return _maybe_count_sketch(out, config)
