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
    not an AMS sketch (Alon–Matias–Szegedy, 1996).

    AMS sketch is a streaming frequency-moment estimator; Count Sketch is a sparse
    dimensionality-reduction map::

        for each input dimension i:
            hash to one output bucket: j  ~ Uniform{0, …, projection_dim - 1}
            draw a random sign:        s  ~ Uniform{-1, +1}
            y[j] += s * x[i]

    Each input dimension touches exactly one output bucket (O(d) time), and the
    ±1 signs ensure E[⟨sketch(x), sketch(y)⟩] = ⟨x, y⟩.  Both Count Sketch and
    the dense ±1/√d projection in the MUVERA paper satisfy a Johnson–Lindenstrauss
    guarantee; they are distinct constructions.
    """

    DEFAULT_IDENTITY = 0
    """No projection; raw embeddings are used directly."""

    AMS_SKETCH = 1
    """Sparse Count Sketch projection (see class docstring for the naming note)."""


class FDEConfig(BaseModel):
    """Immutable configuration for Fixed Dimensional Encoding.

    Parameters
    ----------
    dimension:
        Dimension of each input token embedding (e.g. 128 for ColBERT/ColQwen2).
    num_repetitions:
        Independent repetitions; more → larger FDE output, better approximation.
    num_simhash_projections:
        Number of SimHash bits *k*; partitions = 2 ** k.  Paper default: 4 → 16 partitions.
    seed:
        Shared RNG seed — **must match** between query and document encoders.
    projection_type:
        ``DEFAULT_IDENTITY`` (no projection) or ``AMS_SKETCH`` (Count Sketch).
    projection_dimension:
        Target dimension after Count Sketch projection.  Required (and must be
        positive) when *projection_type* is ``AMS_SKETCH``; ignored otherwise.
    fill_empty_partitions:
        Document-side only.  When ``True``, partition slots with no assigned
        tokens are filled with the projection of the nearest token by SimHash
        Hamming distance.  **Must be ``False`` for query-side encoding.**
    final_projection_dimension:
        If set, the full intermediate FDE is compressed to this size via
        Count Sketch after all repetitions are accumulated.  Reduces memory and
        index storage at the cost of approximation quality.

    Notes
    -----
    The output FDE dimension (before final compression) is::

        num_repetitions × 2**num_simhash_projections × projection_dim

    where *projection_dim* is ``dimension`` for identity projection or
    ``projection_dimension`` for Count Sketch.
    """

    model_config = ConfigDict(frozen=True)

    dimension: int = 128
    num_repetitions: int = 1
    num_simhash_projections: int = 4
    seed: int = 1
    projection_type: ProjectionType = ProjectionType.DEFAULT_IDENTITY
    projection_dimension: int | None = None
    fill_empty_partitions: bool = False
    final_projection_dimension: int | None = None
