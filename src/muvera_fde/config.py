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
    """No projection; raw embeddings are used directly."""

    AMS_SKETCH = 1
    """Sparse Count Sketch projection (see class docstring for the naming note)."""

    LOW_RANK_GAUSSIAN = 2
    """Low-rank Gaussian SimHash: the (d x k) SimHash matrix W is factored as AB^T
    where A in R^{d x r} and B in R^{k x r}, r = ``simhash_rank``.

    Partition assignment uses sign((projected @ A) @ B.T) -- two smaller
    matmuls instead of one large one::

        Full-rank cost:  O(N x d x k)
        Low-rank cost:   O(N x d x r  +  N x r x k)

    For ColQwen2 (d=128, k=8, r=4): 544N vs 1024N -- roughly 2x faster.

    The 1/sqrt(r) normalisation is omitted because SimHash sign assignments are
    scale-invariant: sign(alpha*x) = sign(x) for any alpha > 0.

    Convergence guarantee (EGGROLL, Sarkar et al. 2025, Theorem 4): the
    low-rank sign pattern converges to the full-rank Gaussian sign pattern at
    rate O(r^-1) -- faster than the standard CLT rate O(r^{-1/2}) because
    symmetry cancels all odd cumulants in the Edgeworth expansion.

    Practical targets for ColQwen2 (d=128):

    * r = 4  ->  ~25% variance increase over full-rank
    * r = 8  ->  ~12% variance increase
    * r = 16 ->  visually indistinguishable from full-rank

    This projection type controls the SimHash matrix only and is orthogonal
    to token-embedding projection (``AMS_SKETCH``).  Set ``simhash_rank`` in
    :class:`FDEConfig` to choose r.
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
        ``DEFAULT_IDENTITY`` (no projection), ``AMS_SKETCH`` (Count Sketch on token
        embeddings), or ``LOW_RANK_GAUSSIAN`` (low-rank factored SimHash matrix).
    projection_dimension:
        Target dimension after Count Sketch projection.  Required (and must be
        positive) when *projection_type* is ``AMS_SKETCH``; ignored otherwise.
    simhash_rank:
        Rank *r* of the low-rank SimHash factorisation.  Only used when
        *projection_type* is ``LOW_RANK_GAUSSIAN``.  Must satisfy
        ``1 <= simhash_rank < num_simhash_projections``.  Higher rank -> better
        approximation at higher cost.  r=4 is a practical sweet spot for
        ColQwen2 (128-dim) with ``num_simhash_projections`` >= 8.
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

    where *projection_dim* is ``dimension`` for DEFAULT_IDENTITY / LOW_RANK_GAUSSIAN
    or ``projection_dimension`` for AMS_SKETCH.
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
