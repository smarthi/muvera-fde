"""
Internal random-projection primitives for muvera-fde.

This module is not part of the public API.  All symbols here are considered
implementation details and may change across minor versions.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_COUNT_SKETCH_CHUNK_SIZE: int = 1_000_000
_UINT64_MASK: np.uint64 = np.uint64(0xFFFFFFFFFFFFFFFF)

MAX_SIMHASH_PROJECTIONS: int = 30
MAX_SIMHASH_PROJECTIONS_WITH_FILL: int = 20


# ---------------------------------------------------------------------------
# SimHash utilities
# ---------------------------------------------------------------------------


def simhash_matrix(seed: int, dimension: int, num_projections: int) -> np.ndarray:
    """Return a random Gaussian projection matrix for SimHash partitioning.

    Parameters
    ----------
    seed:
        RNG seed.  Must match between query and document encoders.
    dimension:
        Input embedding dimension (rows).
    num_projections:
        Number of SimHash bits *k* (columns).

    Returns
    -------
    np.ndarray, shape (dimension, num_projections), dtype float32
        Matrix *W* such that ``sign(x @ W)`` gives the *k*-bit partition key
        for embedding *x*.
    """
    return (
        np.random.default_rng(seed).standard_normal((dimension, num_projections)).astype(np.float32)
    )


def simhash_partition_indices(sketch_matrix: np.ndarray) -> np.ndarray:
    """Assign each point a Gray-coded SimHash partition index.

    Adjacent partition indices (differing by one Gray-code bit) correspond to
    geometrically neighboring regions of embedding space.

    Parameters
    ----------
    sketch_matrix:
        Raw projection values, shape (num_points, k).

    Returns
    -------
    np.ndarray, shape (num_points,), dtype int32
        Partition index in [0, 2**k) for each point.
    """
    signs = (sketch_matrix > 0).astype(np.int32)
    binary_indices = np.zeros(signs.shape[0], dtype=np.int32)
    for j in range(signs.shape[1]):
        binary_indices = (binary_indices << 1) | signs[:, j]
    return binary_indices ^ (binary_indices >> 1)


# ---------------------------------------------------------------------------
# Count Sketch
# ---------------------------------------------------------------------------


def _splitmix64(values: np.ndarray) -> np.ndarray:
    """Return a deterministic 64-bit mixed hash for each uint64 input value.

    Used internally by :func:`count_sketch` for position-derived bucket and
    sign hashing without a stored hash table.
    """
    values = (values + np.uint64(0x9E3779B97F4A7C15)) & _UINT64_MASK
    values = ((values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & _UINT64_MASK
    values = ((values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & _UINT64_MASK
    return values ^ (values >> np.uint64(31))


def count_sketch(input_vector: np.ndarray, final_dimension: int, seed: int) -> np.ndarray:
    """Compress a vector to *final_dimension* dimensions via Count Sketch.

    The result is an unbiased estimator of dot products:
    ``E[⟨sketch(x), sketch(y)⟩] = ⟨x, y⟩``.

    Parameters
    ----------
    input_vector:
        Flat vector to compress, shape (N,).
    final_dimension:
        Target output dimension.
    seed:
        RNG seed.  Must match between query and document sides.

    Returns
    -------
    np.ndarray, shape (final_dimension,), dtype float32
    """
    out = np.zeros(final_dimension, dtype=np.float32)
    seed_u64 = np.uint64(seed)
    sign_seed_u64 = seed_u64 ^ np.uint64(0xD6E8FEB86659FD93)
    final_dimension_u64 = np.uint64(final_dimension)

    for start in range(0, len(input_vector), _COUNT_SKETCH_CHUNK_SIZE):
        stop = min(start + _COUNT_SKETCH_CHUNK_SIZE, len(input_vector))
        positions = np.arange(start, stop, dtype=np.uint64)

        index_hashes = _splitmix64(positions ^ seed_u64)
        sign_hashes = _splitmix64(positions ^ sign_seed_u64)

        indices = (index_hashes % final_dimension_u64).astype(np.intp, copy=False)
        signs = np.where(
            (sign_hashes & np.uint64(1)) == 0,
            np.float32(1.0),
            np.float32(-1.0),
        ).astype(np.float32, copy=False)

        np.add.at(out, indices, signs * input_vector[start:stop])

    return out
