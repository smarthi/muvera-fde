"""
Internal random-projection primitives for muvera-fde.

This module is not part of the public API.
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
# Shared utility
# ---------------------------------------------------------------------------


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


# ---------------------------------------------------------------------------
# SimHash — full-rank Gaussian
# ---------------------------------------------------------------------------


def simhash_matrix(seed: int, dimension: int, num_projections: int) -> np.ndarray:
    """Return a random Gaussian projection matrix for SimHash partitioning.

    Parameters
    ----------
    seed: RNG seed.
    dimension: Input embedding dimension (rows).
    num_projections: Number of SimHash bits k (columns).

    Returns
    -------
    np.ndarray, shape (dimension, num_projections), dtype float32
    """
    return (
        np.random.default_rng(seed).standard_normal((dimension, num_projections)).astype(np.float32)
    )


# ---------------------------------------------------------------------------
# SimHash — low-rank Gaussian (EGGROLL-inspired)
# ---------------------------------------------------------------------------


def low_rank_simhash_factors(
    seed: int,
    dimension: int,
    num_projections: int,
    rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return low-rank factors (A, B) for a rank-r SimHash approximation.

    W ~ AB^T where A in R^{d x r}, B in R^{k x r}.
    Convergence O(r^-1) to full-rank (EGGROLL, Sarkar et al. 2025, Theorem 4).
    1/sqrt(r) normalisation omitted -- sign is scale-invariant.

    Returns
    -------
    A : np.ndarray, shape (dimension, rank), dtype float32
    B : np.ndarray, shape (num_projections, rank), dtype float32
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((dimension, rank)).astype(np.float32)
    b = rng.standard_normal((num_projections, rank)).astype(np.float32)
    return a, b


# ---------------------------------------------------------------------------
# SRHT (Subsampled Randomized Hadamard Transform)
# ---------------------------------------------------------------------------


def _fwht_batch(x: np.ndarray) -> np.ndarray:
    """Unnormalised Walsh-Hadamard transform applied row-wise (in-place copy).

    Requires x.shape[-1] to be a power of 2.  O(N x d x log d) time.
    Not normalised by 1/sqrt(d) -- sign is scale-invariant.
    """
    out = x.copy()
    n = out.shape[-1]
    h = 1
    while h < n:
        out = out.reshape(out.shape[0], -1, 2, h)
        u = out[:, :, 0, :].copy()
        v = out[:, :, 1, :].copy()
        out[:, :, 0, :] = u + v
        out[:, :, 1, :] = u - v
        out = out.reshape(out.shape[0], -1)
        h <<= 1
    return out


def srht_params(
    seed: int,
    dimension: int,
    num_projections: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return SRHT parameters: Rademacher signs, subsample indices, padded dim.

    References: Woolfe, Liberty, Rokhlin & Tygert (2008); Ailon & Chazelle (2006);
    improved analysis: Tropp (2011) arXiv:1011.1595.

    Returns
    -------
    d_signs : np.ndarray, shape (padded_dim,), dtype float32
    sample_indices : np.ndarray, shape (num_projections,), dtype int64
    padded_dim : int
    """
    padded_dim = _next_power_of_2(dimension)
    rng = np.random.default_rng(seed)
    d_signs = (2 * rng.integers(0, 2, size=padded_dim) - 1).astype(np.float32)
    sample_indices = rng.choice(padded_dim, size=num_projections, replace=False)
    sample_indices = np.sort(sample_indices).astype(np.int64)
    return d_signs, sample_indices, padded_dim


def apply_srht(
    projected: np.ndarray,
    d_signs: np.ndarray,
    sample_indices: np.ndarray,
    padded_dim: int,
) -> np.ndarray:
    """Apply SRHT to a batch: S H D projected.

    Returns shape (N, k), dtype float32.
    """
    n, d = projected.shape
    if padded_dim > d:
        padded = np.zeros((n, padded_dim), dtype=np.float32)
        padded[:, :d] = projected
    else:
        padded = projected.astype(np.float32, copy=True)
    padded *= d_signs[np.newaxis, :]
    padded = _fwht_batch(padded)
    return padded[:, sample_indices]


# ---------------------------------------------------------------------------
# Cross-Polytope LSH (Andoni & Razenshteyn, 2015)
# ---------------------------------------------------------------------------


def cross_polytope_params(
    seed: int,
    dimension: int,
) -> tuple[np.ndarray, int]:
    """Return Cross-Polytope LSH parameters: Rademacher signs and padded dim.

    Cross-Polytope LSH applies a full SRHT rotation (no subsampling) then
    assigns each token to its dominant coordinate::

        y = H D x_padded                 -- full rotation (no subsample)
        j = argmax_i |y_i|               -- dominant coordinate index
        s = int(y_j > 0)                 -- sign of dominant coord
        partition = 2*j + s              -- in [0, 2*padded_dim)

    This is theoretically optimal for cosine similarity: Cross-Polytope Voronoi
    cells tile high-dimensional space more efficiently than random hyperplanes
    (Andoni & Razenshteyn, 2015).

    Note: num_partitions = 2 * padded_dim, NOT 2^k.
    num_simhash_projections is ignored for this projection type.

    For d=128:  padded_dim=128, num_partitions=256.
    For d=320:  padded_dim=512, num_partitions=1024.

    Parameters
    ----------
    seed: RNG seed.
    dimension: Input embedding dimension d.

    Returns
    -------
    d_signs : np.ndarray, shape (padded_dim,), dtype float32
        Rademacher +-1 signs for the D matrix.
    padded_dim : int
        next_power_of_2(dimension).
    """
    padded_dim = _next_power_of_2(dimension)
    rng = np.random.default_rng(seed)
    d_signs = (2 * rng.integers(0, 2, size=padded_dim) - 1).astype(np.float32)
    return d_signs, padded_dim


def apply_cross_polytope(
    projected: np.ndarray,
    d_signs: np.ndarray,
    padded_dim: int,
) -> np.ndarray:
    """Assign tokens to Cross-Polytope partitions via argmax after SRHT rotation.

    Applies H D to the padded embedding (full rotation, no subsampling), then
    returns partition indices in [0, 2*padded_dim) via::

        j = argmax_i |y_i|
        partition = 2*j + int(y_j > 0)

    Parameters
    ----------
    projected : np.ndarray, shape (N, d)
    d_signs   : np.ndarray, shape (padded_dim,), dtype float32
    padded_dim : int

    Returns
    -------
    np.ndarray, shape (N,), dtype int32
        Partition indices in [0, 2*padded_dim).
    """
    n, d = projected.shape
    if padded_dim > d:
        padded = np.zeros((n, padded_dim), dtype=np.float32)
        padded[:, :d] = projected
    else:
        padded = projected.astype(np.float32, copy=True)

    padded *= d_signs[np.newaxis, :]
    y = _fwht_batch(padded)  # (N, padded_dim)

    j = np.argmax(np.abs(y), axis=1)  # dominant coord index (N,)
    s = (y[np.arange(n), j] > 0).astype(np.int32)  # sign bit (N,)
    return (2 * j + s).astype(np.int32)


# ---------------------------------------------------------------------------
# Gray-coded SimHash partition assignment (for sign-based modes)
# ---------------------------------------------------------------------------


def simhash_partition_indices(sketch_matrix: np.ndarray) -> np.ndarray:
    """Assign each point a Gray-coded SimHash partition index.

    Parameters
    ----------
    sketch_matrix: shape (num_points, k)

    Returns
    -------
    np.ndarray, shape (num_points,), dtype int32
        Partition index in [0, 2**k).
    """
    signs = (sketch_matrix > 0).astype(np.int32)
    binary_indices = np.zeros(signs.shape[0], dtype=np.int32)
    for j in range(signs.shape[1]):
        binary_indices = (binary_indices << 1) | signs[:, j]
    return binary_indices ^ (binary_indices >> 1)


# ---------------------------------------------------------------------------
# Fill strategies
# ---------------------------------------------------------------------------


def densifying_fill(
    rep_slice: np.ndarray,
    projected: np.ndarray,
    empty_pidxs: np.ndarray,
    rep_seed: int,
) -> None:
    """Fill empty partition slots via Densifying LSH (Shrivastava, 2014).

    For each empty slot p, deterministically derive a source token index via
    a splitmix64 hash of the partition index and repetition seed.  This is
    O(num_empty) -- dramatically faster than the O(num_tokens x k) Hamming
    nearest-neighbour fill -- at the cost of less geometrically precise fills.

    Designed for CROSS_POLYTOPE (where no sketch matrix exists for Hamming
    distances) and as a fast alternative for all other projection types.

    Parameters
    ----------
    rep_slice  : np.ndarray, shape (num_partitions, projection_dim)  -- modified in-place
    projected  : np.ndarray, shape (num_tokens, projection_dim)
    empty_pidxs: np.ndarray, shape (num_empty,), dtype int
    rep_seed   : int  -- per-repetition seed for deterministic hashing
    """
    num_tokens = projected.shape[0]
    if num_tokens == 0 or len(empty_pidxs) == 0:
        return
    positions = empty_pidxs.astype(np.uint64)
    seed_u64 = np.uint64(rep_seed) ^ np.uint64(0xA3C59AC3B4D1E6F2)
    hashed = _splitmix64(positions ^ seed_u64)
    token_indices = (hashed % np.uint64(num_tokens)).astype(np.intp)
    rep_slice[empty_pidxs] = projected[token_indices]


# ---------------------------------------------------------------------------
# Count Sketch
# ---------------------------------------------------------------------------


def _splitmix64(values: np.ndarray) -> np.ndarray:
    """Deterministic 64-bit mixed hash for each uint64 input."""
    values = (values + np.uint64(0x9E3779B97F4A7C15)) & _UINT64_MASK
    values = ((values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & _UINT64_MASK
    values = ((values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & _UINT64_MASK
    return values ^ (values >> np.uint64(31))


def count_sketch(input_vector: np.ndarray, final_dimension: int, seed: int) -> np.ndarray:
    """Compress a vector to final_dimension via Count Sketch.

    E[<sketch(x), sketch(y)>] = <x, y>.
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
