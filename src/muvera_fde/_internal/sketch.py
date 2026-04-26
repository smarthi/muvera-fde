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
# SimHash — full-rank Gaussian
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

    The implied SimHash matrix is W ~ AB^T where A in R^{d x r},
    B in R^{k x r}.  Partition assignment uses::

        sign((projected @ A) @ B.T)

    Two smaller matmuls instead of one large one::

        Full-rank cost:  O(N x d x k)
        Low-rank cost:   O(N x d x r  +  N x r x k)

    The 1/sqrt(r) normalization is omitted because SimHash sign assignments
    are scale-invariant: sign(alpha * x) = sign(x) for any alpha > 0.

    Convergence guarantee
    ---------------------
    By EGGROLL Theorem 4 (Sarkar et al., 2025), the low-rank sign pattern
    converges to the full-rank Gaussian sign pattern at O(r^-1) -- faster
    than the standard CLT rate of O(r^{-1/2}) because symmetry cancels all
    odd cumulants in the Edgeworth expansion of the marginal distribution.

    Parameters
    ----------
    seed:
        RNG seed.  Must match between query and document encoders.
    dimension:
        Input embedding dimension *d*.
    num_projections:
        Number of SimHash bits *k*.
    rank:
        Factorisation rank *r*.  Must satisfy 1 <= r < num_projections.

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
# SimHash — SRHT (Subsampled Randomized Hadamard Transform)
# ---------------------------------------------------------------------------


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


def _fwht_batch(x: np.ndarray) -> np.ndarray:
    """Unnormalized Walsh-Hadamard transform applied row-wise.

    Uses the iterative butterfly algorithm: O(N x d x log(d)) time,
    O(N x d) space (in-place on a copy).  Requires x.shape[-1] to be
    a power of 2.

    Parameters
    ----------
    x:
        Input matrix, shape (N, d), where d must be a power of 2.

    Returns
    -------
    np.ndarray, shape (N, d), dtype float32
        Walsh-Hadamard transform of each row.  NOT normalized by 1/sqrt(d);
        normalization is skipped because SimHash uses only the sign of the
        output, which is scale-invariant.
    """
    out = x.copy()
    n = out.shape[-1]
    h = 1
    while h < n:
        # Reshape to (N, n//(2h), 2, h): each block of 2h elements
        # is split into two halves for the butterfly operation.
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

    The SRHT (Woolfe, Liberty, Rokhlin & Tygert, 2008; building on
    Ailon & Chazelle, 2006 Fast Johnson-Lindenstrauss Transform) computes::

        S H D x

    where:

    * D: diagonal +-1 Rademacher matrix (stored as a flat vector of signs)
    * H: Walsh-Hadamard transform (applied implicitly via _fwht_batch)
    * S: row subsampling (stored as a sorted array of k indices)

    The input *x* is zero-padded to ``padded_dim = next_power_of_2(dimension)``
    before applying H, making the transform valid for any embedding dimension.

    The 1/sqrt(k) normalization is omitted: sign assignments are scale-invariant.

    Parameters
    ----------
    seed:
        RNG seed.  Must match between query and document encoders.
    dimension:
        Input embedding dimension *d* (before padding).
    num_projections:
        Number of SimHash bits *k* to subsample.

    Returns
    -------
    d_signs : np.ndarray, shape (padded_dim,), dtype float32
        Rademacher +-1 diagonal signs.
    sample_indices : np.ndarray, shape (num_projections,), dtype int64
        Sorted indices of the *k* subsampled Hadamard rows.
    padded_dim : int
        Padded dimension (next power of 2 >= dimension).
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
    """Apply the SRHT to a batch of embeddings.

    Computes ``S H D projected`` row-wise::

        1. Zero-pad each row from d to padded_dim (next power of 2).
        2. Element-wise multiply by Rademacher signs D.
        3. Apply unnormalised Walsh-Hadamard transform H.
        4. Subsample k columns according to sample_indices.

    Normalisation (1/sqrt(k)) is omitted -- SimHash uses only the sign
    of the output, which is scale-invariant.

    Parameters
    ----------
    projected:
        Token embeddings after optional Count Sketch, shape (N, d).
    d_signs:
        Rademacher +-1 signs, shape (padded_dim,), dtype float32.
    sample_indices:
        Indices of the k subsampled Hadamard rows, shape (k,).
    padded_dim:
        Padded dimension (next power of 2 >= d).

    Returns
    -------
    np.ndarray, shape (N, k), dtype float32
        SRHT sketch values.  sign() of these gives the k-bit partition key.
    """
    n, d = projected.shape
    # 1. Zero-pad if needed
    if padded_dim > d:
        padded = np.zeros((n, padded_dim), dtype=np.float32)
        padded[:, :d] = projected
    else:
        padded = projected.astype(np.float32, copy=True)

    # 2. Apply Rademacher diagonal D
    padded *= d_signs[np.newaxis, :]

    # 3. Walsh-Hadamard transform
    padded = _fwht_batch(padded)

    # 4. Subsample k dimensions
    return padded[:, sample_indices]


# ---------------------------------------------------------------------------
# Gray-coded SimHash partition assignment
# ---------------------------------------------------------------------------


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
# Count Sketch (for token embedding compression and final FDE compression)
# ---------------------------------------------------------------------------


def _splitmix64(values: np.ndarray) -> np.ndarray:
    """Return a deterministic 64-bit mixed hash for each uint64 input value."""
    values = (values + np.uint64(0x9E3779B97F4A7C15)) & _UINT64_MASK
    values = ((values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & _UINT64_MASK
    values = ((values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & _UINT64_MASK
    return values ^ (values >> np.uint64(31))


def count_sketch(input_vector: np.ndarray, final_dimension: int, seed: int) -> np.ndarray:
    """Compress a vector to *final_dimension* dimensions via Count Sketch.

    The result is an unbiased estimator of dot products:
    ``E[<sketch(x), sketch(y)>] = <x, y>``.

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