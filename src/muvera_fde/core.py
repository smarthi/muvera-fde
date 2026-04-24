"""
Core FDE generation for muvera-fde.

``generate_query_fde`` and ``generate_document_fde`` are the primary
computational workhorses.  The :class:`~muvera_fde.encoder.MUVERAEncoder`
high-level class wraps them with pre-built parameter caches.

This module is part of the public API.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from muvera_fde.config import FDEConfig, ProjectionType
from muvera_fde._internal.params import RepParams, build_rep_params
from muvera_fde._internal.sketch import simhash_partition_indices, count_sketch
from muvera_fde._internal.validation import (
    validate_config,
    prepare_embeddings,
    checked_intermediate_fde_length,
)


# ---------------------------------------------------------------------------
# Shared projection + partition helper
# ---------------------------------------------------------------------------

def _project_and_partition(
    embedding_matrix: np.ndarray,
    rep_params: RepParams,
    use_identity: bool,
    projection_dim: int,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Apply optional Count Sketch and SimHash partitioning for one repetition.

    Returns
    -------
    projected : np.ndarray, shape (num_points, projection_dim)
    partition_indices : np.ndarray, shape (num_points,), dtype int32
    sketch_matrix : np.ndarray of shape (num_points, k) or None
        Retained so callers can reuse it for nearest-neighbour fill.
    """
    num_points = embedding_matrix.shape[0]
    if use_identity:
        projected = embedding_matrix
    else:
        projected = np.zeros((num_points, projection_dim), dtype=np.float32)
        np.add.at(projected.T, rep_params.cs_indices, (embedding_matrix * rep_params.cs_signs).T)

    sketch_matrix = (
        projected @ rep_params.simhash_mat
        if rep_params.simhash_mat is not None
        else None
    )
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
    """Fill empty partition slots with the nearest token by SimHash Hamming distance.

    Operates in batches to keep peak memory at O(batch × num_points × k).
    When the point cloud is empty, ``rep_slice`` is left unchanged (all-zero).
    """
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
    sketch_matrix: Optional[np.ndarray],
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
# Public generation functions
# ---------------------------------------------------------------------------

def generate_query_fde(
    point_cloud: np.ndarray,
    config: FDEConfig,
    rep_params_list: Optional[list[RepParams]] = None,
) -> np.ndarray:
    """Generate a query-side Fixed Dimensional Encoding (SUM aggregation).

    Each token's (optionally projected) embedding is **summed** into its
    SimHash partition.  The resulting flat vector approximates Chamfer
    Similarity when paired with a document FDE from :func:`generate_document_fde`
    using the same ``config``.

    Parameters
    ----------
    point_cloud:
        Query token embeddings, shape ``(num_tokens, dimension)`` or a flat
        1-D array of length ``num_tokens * dimension``.
    config:
        Encoding configuration.  ``fill_empty_partitions`` must be ``False``.
    rep_params_list:
        Precomputed per-repetition parameters (e.g. from
        :attr:`~muvera_fde.encoder.MUVERAEncoder._rep_params`).  Built on the
        fly when ``None``.

    Returns
    -------
    np.ndarray, shape ``(fde_dimension,)``, dtype float32
        ``fde_dimension = num_repetitions × 2**num_simhash_projections × projection_dim``
        (or ``final_projection_dimension`` if Count-Sketch compression is set).

    Raises
    ------
    ValueError
        If ``config`` is invalid or ``config.fill_empty_partitions`` is ``True``.
    """
    validate_config(config)
    if config.fill_empty_partitions:
        raise ValueError("Query FDE does not support fill_empty_partitions.")

    embedding_matrix = prepare_embeddings(point_cloud, config)
    use_identity = config.projection_type == ProjectionType.DEFAULT_IDENTITY
    projection_dim = config.dimension if use_identity else config.projection_dimension
    assert projection_dim is not None

    num_partitions = 1 << config.num_simhash_projections
    out = np.zeros(checked_intermediate_fde_length(config, projection_dim), dtype=np.float32)

    for rep in range(config.num_repetitions):
        params = (
            rep_params_list[rep]
            if rep_params_list is not None
            else build_rep_params(
                config.seed + rep, config.dimension, projection_dim,
                config.num_simhash_projections, use_identity,
            )
        )
        projected, partition_indices, _ = _project_and_partition(
            embedding_matrix, params, use_identity, projection_dim
        )
        rep_offset = rep * num_partitions * projection_dim
        rep_slice = out[rep_offset: rep_offset + num_partitions * projection_dim].reshape(
            num_partitions, projection_dim
        )
        np.add.at(rep_slice, partition_indices, projected)

    return _maybe_count_sketch(out, config)


def generate_document_fde(
    point_cloud: np.ndarray,
    config: FDEConfig,
    rep_params_list: Optional[list[RepParams]] = None,
) -> np.ndarray:
    """Generate a document-side Fixed Dimensional Encoding (AVERAGE aggregation).

    Each SimHash partition slot is set to the **centroid** of all tokens that
    fall into it.  When ``config.fill_empty_partitions`` is ``True``, empty
    slots are filled with the nearest token's projection by SimHash Hamming
    distance.

    Parameters
    ----------
    point_cloud:
        Document token embeddings, shape ``(num_tokens, dimension)`` or a flat
        1-D array of length ``num_tokens * dimension``.
    config:
        Encoding configuration.
    rep_params_list:
        Precomputed per-repetition parameters.  Built on the fly when ``None``.

    Returns
    -------
    np.ndarray, shape ``(fde_dimension,)``, dtype float32

    Raises
    ------
    ValueError
        If ``config`` is invalid.
    """
    validate_config(config)
    embedding_matrix = prepare_embeddings(point_cloud, config)
    use_identity = config.projection_type == ProjectionType.DEFAULT_IDENTITY
    projection_dim = config.dimension if use_identity else config.projection_dimension
    assert projection_dim is not None

    num_partitions = 1 << config.num_simhash_projections
    out = np.zeros(checked_intermediate_fde_length(config, projection_dim), dtype=np.float32)

    for rep in range(config.num_repetitions):
        params = (
            rep_params_list[rep]
            if rep_params_list is not None
            else build_rep_params(
                config.seed + rep, config.dimension, projection_dim,
                config.num_simhash_projections, use_identity,
            )
        )
        projected, partition_indices, sketch_matrix = _project_and_partition(
            embedding_matrix, params, use_identity, projection_dim
        )
        rep_offset = rep * num_partitions * projection_dim
        rep_slice = out[rep_offset: rep_offset + num_partitions * projection_dim].reshape(
            num_partitions, projection_dim
        )
        np.add.at(rep_slice, partition_indices, projected)
        _normalize_and_fill_rep(
            rep_slice, partition_indices, projected, sketch_matrix, config, num_partitions
        )

    return _maybe_count_sketch(out, config)
