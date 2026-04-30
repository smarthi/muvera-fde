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
    """

    DEFAULT_IDENTITY = 0
    """Full-rank Gaussian SimHash (baseline).

    Samples W ~ N(0,1)^{d x k} per repetition.
    Partition cost: O(N x d x k).
    Quality: full-rank Gaussian baseline.
    Constraint: none.
    """

    AMS_SKETCH = 1
    """Count Sketch token projection + full-rank Gaussian SimHash.

    Reduces per-partition slot width from d to projection_dimension before
    SimHash via a sparse +-1 random map.
    Partition cost: O(N x projection_dimension x k).
    Quality: unbiased dot-product estimator; JL guarantee.
    Constraint: projection_dimension required.
    """

    LOW_RANK_GAUSSIAN = 2
    """Low-rank factored SimHash: W ~ AB^T where A in R^{d x r}, B in R^{k x r}.

    Two smaller matmuls instead of one large one::

        sketch = (projected @ A) @ B.T   # O(N x d x r + N x r x k)

    Convergence guarantee (EGGROLL, Sarkar et al. 2025, Theorem 4):
    O(r^-1) convergence to full-rank -- faster than CLT rate O(r^{-1/2})
    because symmetry cancels odd cumulants in the Edgeworth expansion.

    Practical targets (d=128, k=8): r=4 -> ~1.9x faster, ~25% variance increase.

    Cost: O(N x d x r + N x r x k).
    Quality: O(r^-1) convergence to full-rank.
    Constraint: simhash_rank required; 1 <= r < k.
    """

    SRHT = 3
    """Subsampled Randomized Hadamard Transform SimHash.

    Structured transform S H D x (sign flip, WHT, subsample) at O(d log d),
    independent of k.  Full JL guarantee, no rank approximation error.

    References: Woolfe, Liberty, Rokhlin & Tygert, 2008; Ailon & Chazelle, 2006;
    improved analysis: Tropp, 2011 arXiv:1011.1595.

    Cost: O(N x d x log d) -- independent of k.
    Quality: full JL, no rank error.
    Constraint: k <= next_power_of_2(d).
    """

    CROSS_POLYTOPE = 4
    """Cross-Polytope LSH using structured SRHT rotation + argmax.

    Applies a full SRHT rotation (D then H on the padded embedding), then
    assigns each token to the partition determined by its dominant coordinate::

        y = H D x_padded                  # full SRHT rotation
        j = argmax_i |y_i|                # dominant coordinate index
        s = int(y_j > 0)                  # sign of dominant coordinate
        partition = 2*j + s               # in [0, 2 * padded_dim)

    This is theoretically optimal for cosine similarity -- Cross-Polytope LSH
    partitions align with the Voronoi cells of the cross-polytope rather than
    random hyperplanes, giving provably better partition efficiency in high
    dimensions (Andoni & Razenshteyn, 2015).

    Key difference from SRHT: SRHT uses sign(S H D x) across k subsampled
    dimensions.  CROSS_POLYTOPE uses argmax(|H D x|) across ALL padded_dim
    dimensions, producing 2*padded_dim partitions per repetition.

    num_simhash_projections is IGNORED for CROSS_POLYTOPE.
    num_partitions = 2 * next_power_of_2(dimension).

    For ColQwen2 (d=128):  padded_dim=128,  num_partitions=256.
    For ColQwen3.5 (d=320): padded_dim=512,  num_partitions=1024.

    Because partitions are plentiful, fill_empty_partitions is recommended.
    CROSS_POLYTOPE always uses densifying fill (O(num_empty), hash-based)
    rather than Hamming NN fill, since no sketch matrix is produced.

    Cost: O(N x d x log d) -- same as SRHT.
    Quality: theoretically optimal cosine-similarity partition efficiency.
    Constraint: dimension must be >= 1; num_simhash_projections is ignored.
    """


class FDEConfig(BaseModel):
    """Immutable configuration for Fixed Dimensional Encoding.

    Parameters
    ----------
    dimension:
        Dimension of each input token embedding.
        ColQwen2=128, ColQwen3.5 v3=320.
    num_repetitions:
        Independent repetitions; more -> larger FDE, better approximation.
    num_simhash_projections:
        SimHash bits *k*; partitions = 2^k.  Ignored for CROSS_POLYTOPE.
    seed:
        Shared RNG seed -- must match between query and document encoders.
    projection_type:
        DEFAULT_IDENTITY, AMS_SKETCH, LOW_RANK_GAUSSIAN, SRHT, or CROSS_POLYTOPE.
    projection_dimension:
        Target dimension after Count Sketch.  Required for AMS_SKETCH.
    simhash_rank:
        Rank *r* for LOW_RANK_GAUSSIAN.  Must satisfy 1 <= r < k.
    fill_empty_partitions:
        Document-side only.  Fill empty partition slots.  Must be False for queries.
    densifying_fill:
        When True (and fill_empty_partitions=True), use O(num_empty) hash-based
        Densifying LSH fill (Shrivastava, 2014) instead of the default O(N*k)
        Hamming nearest-neighbour fill.  Automatically forced True for CROSS_POLYTOPE
        since no sketch matrix is available for Hamming distances.
        Densifying fill is faster but less geometrically precise than Hamming fill.
    final_projection_dimension:
        Post-accumulation Count Sketch compression to this size.

    Notes
    -----
    Output FDE dimension (before final compression)::

        num_repetitions x num_partitions x projection_dim

    where num_partitions = 2^k for DEFAULT_IDENTITY / AMS_SKETCH / LOW_RANK_GAUSSIAN /
    SRHT, or 2*next_power_of_2(dimension) for CROSS_POLYTOPE.
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
    densifying_fill: bool = False
    final_projection_dimension: int | None = None
