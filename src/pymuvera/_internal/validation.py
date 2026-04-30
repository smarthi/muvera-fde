"""
Validation helpers for FDEConfig.

This module is not part of the public API.
"""

from __future__ import annotations

import numpy as np

from pymuvera._internal.sketch import (
    MAX_SIMHASH_PROJECTIONS,
    MAX_SIMHASH_PROJECTIONS_WITH_FILL,
    _next_power_of_2,
)
from pymuvera.config import FDEConfig, ProjectionType

_MAX_INTERMEDIATE_FDE_BYTES: int = 1 << 30  # 1 GiB


# ---------------------------------------------------------------------------
# num_partitions helper — single source of truth
# ---------------------------------------------------------------------------


def num_partitions_for_config(config: FDEConfig) -> int:
    """Return the number of partitions per repetition for the given config.

    For all sign-based modes:  2 ** num_simhash_projections.
    For CROSS_POLYTOPE:        2 * next_power_of_2(projection_dim).

    This is the single source of truth used by core.py, encoder.py, and
    validation; it must be called with a validated config.
    """
    if config.projection_type == ProjectionType.CROSS_POLYTOPE:
        proj_dim = _projection_dim_for_config(config)
        return 2 * _next_power_of_2(proj_dim)
    return 1 << config.num_simhash_projections


def _projection_dim_for_config(config: FDEConfig) -> int:
    """Per-partition slot width (AMS_SKETCH uses projection_dimension, others use dimension)."""
    if config.projection_type == ProjectionType.AMS_SKETCH:
        assert config.projection_dimension is not None
        return config.projection_dimension
    return config.dimension


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def _check_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _check_simhash_projections(config: FDEConfig) -> None:
    # Cross-Polytope ignores num_simhash_projections entirely -- skip check
    if config.projection_type == ProjectionType.CROSS_POLYTOPE:
        return
    limit = (
        MAX_SIMHASH_PROJECTIONS_WITH_FILL
        if config.fill_empty_partitions
        else MAX_SIMHASH_PROJECTIONS
    )
    if 0 <= config.num_simhash_projections <= limit:
        return
    suffix = " when fill_empty_partitions=True" if config.fill_empty_partitions else ""
    raise ValueError(
        f"num_simhash_projections must be in [0, {limit}]{suffix}, "
        f"got {config.num_simhash_projections}"
    )


def _check_projection_dimension(config: FDEConfig) -> None:
    if config.projection_type == ProjectionType.AMS_SKETCH:
        if config.projection_dimension is None or config.projection_dimension <= 0:
            raise ValueError("A positive projection_dimension must be set when using AMS_SKETCH.")


def _check_simhash_rank(config: FDEConfig) -> None:
    if config.projection_type != ProjectionType.LOW_RANK_GAUSSIAN:
        return
    if config.simhash_rank <= 0:
        raise ValueError(
            f"simhash_rank must be positive when using LOW_RANK_GAUSSIAN, got {config.simhash_rank}"
        )
    if config.num_simhash_projections > 0 and config.simhash_rank >= config.num_simhash_projections:
        raise ValueError(
            f"simhash_rank ({config.simhash_rank}) must be strictly less than "
            f"num_simhash_projections ({config.num_simhash_projections}) "
            "to form a proper low-rank factorisation."
        )


def _check_srht(config: FDEConfig) -> None:
    if config.projection_type != ProjectionType.SRHT:
        return
    if config.num_simhash_projections == 0:
        return
    proj_dim = _projection_dim_for_config(config)
    padded_dim = _next_power_of_2(proj_dim)
    if config.num_simhash_projections > padded_dim:
        raise ValueError(
            f"SRHT requires num_simhash_projections ({config.num_simhash_projections}) "
            f"<= next_power_of_2(dimension) = {padded_dim}. "
            "Reduce num_simhash_projections or increase dimension."
        )


def validate_config(config: FDEConfig) -> None:
    """Validate all fields of an FDEConfig.

    Raises
    ------
    ValueError on any invalid field value.
    """
    _check_positive(config.dimension, "dimension")
    _check_positive(config.num_repetitions, "num_repetitions")
    if config.final_projection_dimension is not None:
        _check_positive(config.final_projection_dimension, "final_projection_dimension")
    _check_simhash_projections(config)
    _check_projection_dimension(config)
    _check_simhash_rank(config)
    _check_srht(config)


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------


def prepare_embeddings(point_cloud: np.ndarray, config: FDEConfig) -> np.ndarray:
    """Coerce a point cloud to float32 matrix of shape (num_points, dimension)."""
    point_cloud = np.asarray(point_cloud, dtype=np.float32)
    if point_cloud.ndim == 2:
        if point_cloud.shape[1] != config.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: got {point_cloud.shape[1]}, "
                f"expected {config.dimension}"
            )
        return point_cloud
    if point_cloud.ndim == 1:
        if len(point_cloud) % config.dimension != 0:
            raise ValueError(
                f"Flat point-cloud length {len(point_cloud)} not divisible by "
                f"dimension {config.dimension}"
            )
        return point_cloud.reshape(-1, config.dimension)
    raise ValueError(f"point_cloud must be 1-D or 2-D, got {point_cloud.ndim}-D")


def checked_intermediate_fde_length(
    config: FDEConfig,
    projection_dim: int,
    num_partitions: int,
) -> int:
    """Return flat intermediate FDE length and guard against huge allocations."""
    fde_length = config.num_repetitions * num_partitions * projection_dim
    required_bytes = fde_length * np.dtype(np.float32).itemsize
    if required_bytes > _MAX_INTERMEDIATE_FDE_BYTES:
        raise ValueError(
            f"Configuration would allocate an intermediate FDE of {required_bytes} bytes "
            f"({config.num_repetitions} repetitions x {num_partitions} partitions x "
            f"{projection_dim} dimensions), which exceeds the "
            f"{_MAX_INTERMEDIATE_FDE_BYTES}-byte limit."
        )
    return fde_length
