"""
High-level encoder for muvera-fde.

:class:`MUVERAEncoder` is the primary entry point for most users.  It wraps
:func:`~muvera_fde.core.generate_query_fde` and
:func:`~muvera_fde.core.generate_document_fde` with a pre-built parameter
cache so repeated encode calls pay no RNG or allocation overhead.
"""

from __future__ import annotations

import numpy as np

from muvera_fde._internal.params import RepParams, build_rep_params
from muvera_fde._internal.validation import checked_intermediate_fde_length, validate_config
from muvera_fde.config import FDEConfig, ProjectionType
from muvera_fde.core import (
    _projection_dim_for,
    _use_identity,
    _use_low_rank_simhash,
    generate_document_fde,
    generate_query_fde,
)


class MUVERAEncoder:
    """Encodes multi-vector token embeddings into fixed-dimensional FDE vectors.

    Use the **same encoder instance** for both queries and documents so that
    the random partition structure (seed, SimHash matrices, Count Sketch
    parameters) is guaranteed to be consistent.

    Parameters
    ----------
    dimension:
        Token embedding dimension (128 for ColBERT / ColQwen2).
    num_simhash_projections:
        Number of SimHash bits *k*; partitions = 2 ** k.  Paper default: 4.
    num_repetitions:
        Independent repetitions for better approximation quality.
    seed:
        Shared RNG seed -- must be identical across query and document encoders.
    projection_type:
        :attr:`~muvera_fde.config.ProjectionType.DEFAULT_IDENTITY` (no projection),
        :attr:`~muvera_fde.config.ProjectionType.AMS_SKETCH` (Count Sketch on token
        embeddings), or
        :attr:`~muvera_fde.config.ProjectionType.LOW_RANK_GAUSSIAN` (low-rank
        factored SimHash matrix).
    projection_dimension:
        Target dimension after Count Sketch projection.  Required when
        *projection_type* is ``AMS_SKETCH``; ignored otherwise.
    simhash_rank:
        Rank *r* for ``LOW_RANK_GAUSSIAN`` SimHash factorisation.
        Must satisfy ``1 <= simhash_rank < num_simhash_projections``.
        Higher rank -> better approximation at higher cost.
        r=4 is a practical sweet spot for ColQwen2 (128-dim, k>=8).
        Ignored when *projection_type* is not ``LOW_RANK_GAUSSIAN``.
    fill_empty_partitions:
        Document side only.  When ``True``, empty partition slots are filled
        with the projection of the nearest token by SimHash Hamming distance.
    final_projection_dimension:
        If set, the output FDE is compressed to this size via Count Sketch,
        reducing memory and index storage.

    Examples
    --------
    Basic usage (ColQwen2 / ColBERT embeddings, 128-dim)::

        import numpy as np
        from muvera_fde import MUVERAEncoder

        enc = MUVERAEncoder(dimension=128, num_simhash_projections=4, num_repetitions=2)

        query_tokens = np.random.randn(32,  128).astype(np.float32)
        doc_tokens   = np.random.randn(512, 128).astype(np.float32)

        q_fde = enc.encode_query(query_tokens)   # shape: (2 * 16 * 128,) = (4096,)
        d_fde = enc.encode_document(doc_tokens)
        score = float(q_fde @ d_fde)             # approx Chamfer Similarity

    With low-rank SimHash (EGGROLL-style, faster partition assignment)::

        from muvera_fde import ProjectionType

        enc = MUVERAEncoder(
            dimension=128,
            num_simhash_projections=8,   # 2^8 = 256 partitions
            num_repetitions=4,
            projection_type=ProjectionType.LOW_RANK_GAUSSIAN,
            simhash_rank=4,              # r=4 -> ~25% variance vs full-rank
        )
        # SimHash cost: O(N*128*4 + N*4*8) = O(544N) vs O(1024N) full-rank
        # Convergence to full-rank: O(r^-1) = O(0.25) (EGGROLL Theorem 4)

    With Count Sketch token compression::

        enc = MUVERAEncoder(
            dimension=128,
            num_simhash_projections=4,
            num_repetitions=4,
            projection_type=ProjectionType.AMS_SKETCH,
            projection_dimension=64,
            fill_empty_partitions=True,
            final_projection_dimension=512,
        )
    """

    def __init__(
        self,
        dimension: int = 128,
        num_simhash_projections: int = 4,
        num_repetitions: int = 1,
        seed: int = 1,
        projection_type: ProjectionType = ProjectionType.DEFAULT_IDENTITY,
        projection_dimension: int | None = None,
        simhash_rank: int = 1,
        fill_empty_partitions: bool = False,
        final_projection_dimension: int | None = None,
    ) -> None:
        self._base_config: dict = dict(
            dimension=dimension,
            num_repetitions=num_repetitions,
            num_simhash_projections=num_simhash_projections,
            seed=seed,
            projection_type=projection_type,
            projection_dimension=projection_dimension,
            simhash_rank=simhash_rank,
            final_projection_dimension=final_projection_dimension,
        )
        self.fill_empty_partitions = fill_empty_partitions

        # Validate eagerly -- fde_dimension and encode_* are always safe after __init__.
        config = FDEConfig(**self._base_config, fill_empty_partitions=fill_empty_partitions)
        validate_config(config)

        _proj_dim = _projection_dim_for(config)
        _use_id = _use_identity(config)
        _use_lr = _use_low_rank_simhash(config)

        checked_intermediate_fde_length(config, _proj_dim)

        # Precompute per-repetition parameters once.
        self._rep_params: list[RepParams] = [
            build_rep_params(
                seed + rep,
                dimension,
                _proj_dim,
                num_simhash_projections,
                _use_id,
                use_low_rank_simhash=_use_lr,
                simhash_rank=simhash_rank,
            )
            for rep in range(num_repetitions)
        ]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fde_dimension(self) -> int:
        """Output dimension of every FDE vector produced by this encoder.

        Returns ``final_projection_dimension`` when Count-Sketch compression is
        configured; otherwise returns
        ``num_repetitions x 2**num_simhash_projections x projection_dim``
        where projection_dim is ``dimension`` for DEFAULT_IDENTITY / LOW_RANK_GAUSSIAN,
        or ``projection_dimension`` for AMS_SKETCH.
        """
        if self._base_config["final_projection_dimension"] is not None:
            return self._base_config["final_projection_dimension"]
        config = FDEConfig(**self._base_config, fill_empty_partitions=self.fill_empty_partitions)
        proj_dim = _projection_dim_for(config)
        num_partitions = 1 << self._base_config["num_simhash_projections"]
        return self._base_config["num_repetitions"] * num_partitions * proj_dim

    # ------------------------------------------------------------------
    # Single-item encode
    # ------------------------------------------------------------------

    def encode_query(self, token_embeddings: np.ndarray) -> np.ndarray:
        """Encode query token embeddings into a single FDE vector (SUM aggregation).

        Parameters
        ----------
        token_embeddings:
            Shape ``(num_tokens, dimension)`` or flat 1-D array of length
            ``num_tokens * dimension``.

        Returns
        -------
        np.ndarray, shape ``(fde_dimension,)``, dtype float32
        """
        config = FDEConfig(**self._base_config, fill_empty_partitions=False)
        return generate_query_fde(token_embeddings, config, self._rep_params)

    def encode_document(self, token_embeddings: np.ndarray) -> np.ndarray:
        """Encode document token embeddings into a single FDE vector (AVERAGE aggregation).

        Parameters
        ----------
        token_embeddings:
            Shape ``(num_tokens, dimension)`` or flat 1-D array of length
            ``num_tokens * dimension``.

        Returns
        -------
        np.ndarray, shape ``(fde_dimension,)``, dtype float32
        """
        config = FDEConfig(**self._base_config, fill_empty_partitions=self.fill_empty_partitions)
        return generate_document_fde(token_embeddings, config, self._rep_params)

    # ------------------------------------------------------------------
    # Batch encode
    # ------------------------------------------------------------------

    def encode_queries_batch(self, batch: list[np.ndarray]) -> np.ndarray:
        """Encode a batch of query point clouds.

        Parameters
        ----------
        batch:
            List of query point clouds; each element is accepted by
            :meth:`encode_query`.

        Returns
        -------
        np.ndarray, shape ``(N, fde_dimension)``, dtype float32
        """
        return np.stack([self.encode_query(q) for q in batch])

    def encode_documents_batch(self, batch: list[np.ndarray]) -> np.ndarray:
        """Encode a batch of document point clouds.

        Parameters
        ----------
        batch:
            List of document point clouds; each element is accepted by
            :meth:`encode_document`.

        Returns
        -------
        np.ndarray, shape ``(N, fde_dimension)``, dtype float32
        """
        return np.stack([self.encode_document(d) for d in batch])

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        cfg = self._base_config
        pt = cfg["projection_type"]
        rank_str = (
            f", simhash_rank={cfg['simhash_rank']}"
            if pt == ProjectionType.LOW_RANK_GAUSSIAN
            else ""
        )
        return (
            f"MUVERAEncoder("
            f"dimension={cfg['dimension']}, "
            f"num_simhash_projections={cfg['num_simhash_projections']}, "
            f"num_repetitions={cfg['num_repetitions']}, "
            f"projection_type={pt.name}"
            f"{rank_str}, "
            f"fde_dimension={self.fde_dimension}"
            f")"
        )
