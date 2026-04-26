"""
Public configuration types for muvera-fde.

These are the only types callers need to import to configure an encoder.
"""

from __future__ import annotations

import enum

from pydantic import BaseModel, ConfigDict


class ProjectionType(enum.Enum):
    """Projection strategy applied to token embeddings before SimHash partitioning.

    Naming note
    -----------
    ``AMS_SKETCH`` is a misnomer inherited from the Google graph-mining C++ source.
    The actual construction is a **Count Sketch** (Charikar, Chen & Farach-Colton, 2002),
    not an AMS sketch (Alon-Matias-Szegedy, 1996).

    AMS sketch is a streaming frequency-moment estimator; Count Sketch is a sparse
    dimensionality-reduction map::

        for each input dimension i:
            hash to one output bucket: j  ~ Uniform{0, ..., projection_dim - 1}
            draw a random sign:        s  ~ Uniform{-1, +1}
            y[j] += s * x[i]

    Each input dimension touches exactly one output bucket (O(d) time), and the
    +-1 signs ensure E[<sketch(x), sketch(y)>] = <x, y>.  Both Count Sketch and
    the dense +-1/sqrt(d) projection in the MUVERA paper satisfy a Johnson-Lindenstrauss
    guarantee; they are distinct constructions.
    """

    DEFAULT_IDENTITY = 0
    """No projection; raw embeddings are used directly.

    Full-rank Gaussian SimHash matrix W in R^{d x k} is sampled once per
    repetition.  Partition assignment uses sign(projected @ W).

    Cost: O(N x d x k) per repetition.
    Quality: full-rank Gaussian baseline.
    Constraint: none.
    """

    AMS_SKETCH = 1
    """Sparse Count Sketch projection on token embeddings before SimHash.

    Reduces per-partition slot width from d to projection_dimension via a
    sparse +-1 random map before applying full-rank Gaussian SimHash.  Used
    when FDE output size is the primary concern.

    Cost (token projection): O(N x d) -- one non-zero per input dimension.
    Cost (SimHash): O(N x projection_dimension x k).
    Quality: unbiased dot-product estimator; JL guarantee inherited.
    Constraint: projection_dimension required.

    See class docstring for the Count Sketch vs AMS Sketch naming note.
    """

    LOW_RANK_GAUSSIAN = 2
    """Low-rank Gaussian SimHash: W ~ AB^T, A in R^{d x r}, B in R^{k x r}.

    Replaces the full (d x k) SimHash matrix with two smaller Gaussian factors,
    cutting the per-repetition SimHash cost from O(N x d x k) to
    O(N x d x r + N x r x k) via two smaller matmuls::

        sketch = (projected @ A) @ B.T   # shape: (N, k)

    The 1/sqrt(r) normalisation is omitted: sign assignments are scale-invariant
    (sign(alpha*x) = sign(x) for any alpha > 0), so it has no effect.

    Convergence guarantee (EGGROLL, Sarkar et al. 2025, Theorem 4): the
    low-rank sign pattern converges to the full-rank Gaussian sign pattern at
    O(r^-1) -- faster than the standard CLT rate O(r^{-1/2}) -- because
    symmetry cancels all odd cumulants in the Edgeworth expansion.

    Practical targets for ColQwen2 (d=128, k=8):

    * r = 4  ->  ~25% variance increase, 544N vs 1024N ops (~1.9x faster)
    * r = 8  ->  ~12% variance increase, 1088N vs 1024N ops (~breakeven)

    Cost: O(N x d x r + N x r x k).
    Quality: O(r^-1) convergence to full-rank.
    Constraint: simhash_rank required; 1 <= r < k.
    """

    SRHT = 3
    """Subsampled Randomized Hadamard Transform (SRHT) for SimHash.

    Replaces the dense Gaussian SimHash matrix with a structured transform:

        S H D x

    where:

    * D: random diagonal +-1 (Rademacher) matrix -- element-wise sign flip
    * H: Walsh-Hadamard transform -- O(d log d) butterfly operations
    * S: random row subsampling -- selects k of the d transformed dimensions

    The input is zero-padded to the next power of 2 >= d before applying H,
    so the transform is valid for any embedding dimension.

    The 1/sqrt(k) normalisation is omitted for the same reason as LOW_RANK_GAUSSIAN:
    SimHash sign assignments are scale-invariant.

    Theoretical guarantees (Woolfe, Liberty, Rokhlin & Tygert, 2008;
    building on Ailon & Chazelle, 2006 Fast Johnson-Lindenstrauss Transform):

    * Johnson-Lindenstrauss guarantee: pairwise distances preserved to eps with high probability
    * Cost: O(d log d) per token -- sub-quadratic in d
    * No approximation vs full-rank Gaussian when k <= d (exact JL)

    Cost comparison for ColQwen2 (d=128, k=8, log2(128)=7):

    * DEFAULT_IDENTITY: O(N x 128 x 8) = 1024N ops
    * LOW_RANK_GAUSSIAN r=4: O(N x 128 x 4 + N x 4 x 8) = 544N ops
    * SRHT: O(N x 128 x 7 + N x 8) = 904N ops

    SRHT has stronger theoretical guarantees than LOW_RANK_GAUSSIAN (no rank
    approximation error -- it IS a full JL projection, just structured) but
    is slower than LOW_RANK_GAUSSIAN for small r.  Choose SRHT when you need
    full JL quality with sub-quadratic cost; choose LOW_RANK_GAUSSIAN when
    speed dominates and mild approximation error is acceptable.

    Cost: O(N x d x log(d)) -- independent of k.
    Quality: full JL guarantee, no rank approximation.
    Constraint: num_simhash_projections must satisfy k <= next_power_of_2(d).
    """


class FDEConfig(BaseModel):
    """Immutable configuration for Fixed Dimensional Encoding.

    Parameters
    ----------
    dimension:
        Dimension of each input token embedding (e.g. 128 for ColBERT/ColQwen2).
    num_repetitions:
        Independent repetitions; more -> larger FDE output, better approximation.
    num_simhash_projections:
        Number of SimHash bits *k*; partitions = 2 ** k.  Paper default: 4 -> 16 partitions.
    seed:
        Shared RNG seed -- **must match** between query and document encoders.
    projection_type:
        ``DEFAULT_IDENTITY``, ``AMS_SKETCH``, ``LOW_RANK_GAUSSIAN``, or ``SRHT``.
        See :class:`ProjectionType` for full documentation of each.
    projection_dimension:
        Target dimension after Count Sketch projection.  Required (and must be
        positive) when *projection_type* is ``AMS_SKETCH``; ignored otherwise.
    simhash_rank:
        Rank *r* of the low-rank SimHash factorisation.  Only used when
        *projection_type* is ``LOW_RANK_GAUSSIAN``.  Must satisfy
        ``1 <= simhash_rank < num_simhash_projections``.  r=4 is a practical
        sweet spot for ColQwen2 (d=128) with ``num_simhash_projections`` >= 8.
        Ignored for all other projection types.
    fill_empty_partitions:
        Document-side only.  When ``True``, partition slots with no assigned
        tokens are filled with the projection of the nearest token by SimHash
        Hamming distance.  Must be ``False`` for query-side encoding.
    final_projection_dimension:
        If set, the full intermediate FDE is compressed to this size via
        Count Sketch after all repetitions are accumulated.  Reduces memory and
        index storage at the cost of approximation quality.

    Notes
    -----
    The output FDE dimension (before final compression) is::

        num_repetitions x 2**num_simhash_projections x projection_dim

    where *projection_dim* is ``dimension`` for DEFAULT_IDENTITY / LOW_RANK_GAUSSIAN /
    SRHT, or ``projection_dimension`` for AMS_SKETCH.
    """

    model_config = ConfigDict(frozen=True)

    dimension: int = 128
    num_repetitions: int = 1
    num_simhash_projections: int = 4
    seed: int = 1
    projection_type: ProjectionType = ProjectionType.DEFAULT_IDENTITY
    projection_dimension: int | None = None
    simhash_rank: int = 1
    fill_empty_partitions: bool = False
    final_projection_dimension: int | None = None