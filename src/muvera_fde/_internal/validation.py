"""
Validation helpers for FDEConfig.

This module is not part of the public API.
"""

from __future__ import annotations

import numpy as np

from muvera_fde._internal.sketch import (
    MAX_SIMHASH_PROJECTIONS,
    MAX_SIMHASH_PROJECTIONS_WITH_FILL,
    _next_power_of_2,
)
from muvera_fde.config import FDEConfig, ProjectionType

_MAX_INTERMEDIATE_FDE_BYTES: int = 1 << 30  # 1 GiB practical upper bound


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def _check_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _check_simhash_projections(config: FDEConfig) -> None:
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
    """Validate simhash_rank for LOW_RANK_GAUSSIAN."""
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
    """Validate SRHT constraints.

    SRHT subsamples k rows from the padded_dim-dimensional Hadamard output.
    This requires k <= padded_dim = next_power_of_2(projection_dim).

    For ColQwen2 (d=128): padded_dim=128, so k can be at most 128.
    For d=100: padded_dim=128, same limit.
    For d=200: padded_dim=256, k can be at most 256.

    SRHT with num_simhash_projections=0 is a no-op (no partitioning), which
    is valid -- the check only fires when k > 0.
    """
    if config.projection_type != ProjectionType.SRHT:
        return
    if config.num_simhash_projections == 0:
        return
    proj_dim = (
        config.projection_dimension
        if config.projection_type == ProjectionType.AMS_SKETCH
        else config.dimension
    )
    padded_dim = _next_power_of_2(proj_dim)
    if config.num_simhash_projections > padded_dim:
        raise ValueError(
            f"SRHT requires num_simhash_projections ({config.num_simhash_projections}) "
            f"<= next_power_of_2(dimension) = {padded_dim}. "
            f"Reduce num_simhash_projections or increase dimension."
        )


def validate_config(config: FDEConfig) -> None:
    """Validate all fields of an :class:`~muvera_fde.config.FDEConfig`.

    Raises
    ------
    ValueError
        On any invalid field value.
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
    """Coerce a point cloud to a float32 matrix of shape (num_points, dimension)."""
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


def checked_intermediate_fde_length(config: FDEConfig, projection_dim: int) -> int:
    """Return the flat intermediate FDE length and guard against huge allocations."""
    num_partitions = 1 << config.num_simhash_projections
    fde_length = config.num_repetitions * num_partitions * projection_dim
    required_bytes = fde_length * np.dtype(np.float32).itemsize
    if required_bytes > _MAX_INTERMEDIATE_FDE_BYTES:
        raise ValueError(
            f"Configuration would allocate an intermediate FDE of {required_bytes} bytes "
            f"({config.num_repetitions} repetitions x {num_partitions} partitions x "
            f"{projection_dim} dimensions), which exceeds the "
            f"{_MAX_INTERMEDIATE_FDE_BYTES}-byte limit. Reduce num_simhash_projections, "
            "num_repetitions, or projection_dimension."
        )
    return fde_length
