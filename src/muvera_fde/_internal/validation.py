"""
Validation helpers for FDEConfig.

This module is not part of the public API.
"""

from __future__ import annotations

import numpy as np

from muvera_fde.config import FDEConfig, ProjectionType
from muvera_fde._internal.sketch import MAX_SIMHASH_PROJECTIONS, MAX_SIMHASH_PROJECTIONS_WITH_FILL

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
    if config.projection_type == ProjectionType.DEFAULT_IDENTITY:
        return
    if config.projection_dimension is None or config.projection_dimension <= 0:
        raise ValueError(
            "A positive projection_dimension must be set when using a "
            "non-identity projection_type."
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


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------

def prepare_embeddings(point_cloud: np.ndarray, config: FDEConfig) -> np.ndarray:
    """Coerce a point cloud to a float32 matrix of shape (num_points, dimension).

    Accepts either a 2-D array ``(num_points, dimension)`` or a flat 1-D array
    of length ``num_points * dimension``, which is reshaped automatically.

    Raises
    ------
    ValueError
        On dimension mismatch or unsupported array rank.
    """
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
    """Return the flat intermediate FDE length and guard against huge allocations.

    Raises
    ------
    ValueError
        If the allocation would exceed 1 GiB.
    """
    num_partitions = 1 << config.num_simhash_projections
    fde_length = config.num_repetitions * num_partitions * projection_dim
    required_bytes = fde_length * np.dtype(np.float32).itemsize
    if required_bytes > _MAX_INTERMEDIATE_FDE_BYTES:
        raise ValueError(
            f"Configuration would allocate an intermediate FDE of {required_bytes} bytes "
            f"({config.num_repetitions} repetitions × {num_partitions} partitions × "
            f"{projection_dim} dimensions), which exceeds the "
            f"{_MAX_INTERMEDIATE_FDE_BYTES}-byte limit. Reduce num_simhash_projections, "
            "num_repetitions, or projection_dimension."
        )
    return fde_length
