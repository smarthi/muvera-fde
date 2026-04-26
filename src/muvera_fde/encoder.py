"""
High-level encoder for muvera-fde.
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
    _use_srht,
    generate_document_fde,
    generate_query_fde,
)


class MUVERAEncoder:
    """Encodes multi-vector token embeddings into fixed-dimensional FDE vectors.

    Use the **same encoder instance** for both queries and documents so that
    the random partition structure is guaranteed to be consistent.

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
        One of:

        * ``DEFAULT_IDENTITY`` -- full-rank Gaussian SimHash (baseline).
        * ``AMS_SKETCH`` -- Count Sketch token projection + full-rank SimHash.
        * ``LOW_RANK_GAUSSIAN`` -- low-rank factored SimHash (EGGROLL-inspired).
        * ``SRHT`` -- Subsampled Randomized Hadamard Transform SimHash.

        See :class:`~muvera_fde.config.ProjectionType` for full details and
        cost/quality tradeoffs.
    projection_dimension:
        Target dimension after Count Sketch.  Required for ``AMS_SKETCH``.
    simhash_rank:
        Rank *r* for ``LOW_RANK_GAUSSIAN``.
        Must satisfy ``1 <= simhash_rank < num_simhash_projections``.
        Ignored for other projection types.
    fill_empty_partitions:
        Document side only.  When ``True``, empty partition slots are filled
        with the projection of the nearest token by SimHash Hamming distance.
    final_projection_dimension:
        If set, compress the output FDE via Count Sketch to this size.

    Examples
    --------
    Baseline (full-rank Gaussian SimHash)::

        enc = MUVERAEncoder(dimension=128, num_simhash_projections=4, num_repetitions=2)

    Low-rank SimHash (EGGROLL, ~1.9x faster for ColQwen2 k=8)::

        enc = MUVERAEncoder(
            dimension=128, num_simhash_projections=8, num_repetitions=4,
            projection_type=ProjectionType.LOW_RANK_GAUSSIAN, simhash_rank=4,
        )

    SRHT (full JL guarantee, O(d log d) cost)::

        enc = MUVERAEncoder(
            dimension=128, num_simhash_projections=8, num_repetitions=4,
            projection_type=ProjectionType.SRHT,
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

        config = FDEConfig(**self._base_config, fill_empty_partitions=fill_empty_partitions)
        validate_config(config)

        _proj_dim = _projection_dim_for(config)
        _use_id = _use_identity(config)
        _use_lr = _use_low_rank_simhash(config)
        _use_sh = _use_srht(config)

        checked_intermediate_fde_length(config, _proj_dim)

        self._rep_params: list[RepParams] = [
            build_rep_params(
                seed + rep,
                dimension,
                _proj_dim,
                num_simhash_projections,
                _use_id,
                use_low_rank_simhash=_use_lr,
                simhash_rank=simhash_rank,
                use_srht=_use_sh,
            )
            for rep in range(num_repetitions)
        ]

    @property
    def fde_dimension(self) -> int:
        """Output dimension of every FDE vector produced by this encoder."""
        if self._base_config["final_projection_dimension"] is not None:
            return self._base_config["final_projection_dimension"]
        config = FDEConfig(**self._base_config, fill_empty_partitions=self.fill_empty_partitions)
        proj_dim = _projection_dim_for(config)
        num_partitions = 1 << self._base_config["num_simhash_projections"]
        return self._base_config["num_repetitions"] * num_partitions * proj_dim

    def encode_query(self, token_embeddings: np.ndarray) -> np.ndarray:
        """Encode query token embeddings into a single FDE vector (SUM)."""
        config = FDEConfig(**self._base_config, fill_empty_partitions=False)
        return generate_query_fde(token_embeddings, config, self._rep_params)

    def encode_document(self, token_embeddings: np.ndarray) -> np.ndarray:
        """Encode document token embeddings into a single FDE vector (AVERAGE)."""
        config = FDEConfig(**self._base_config, fill_empty_partitions=self.fill_empty_partitions)
        return generate_document_fde(token_embeddings, config, self._rep_params)

    def encode_queries_batch(self, batch: list[np.ndarray]) -> np.ndarray:
        """Encode a batch of query point clouds -> (N, fde_dimension)."""
        return np.stack([self.encode_query(q) for q in batch])

    def encode_documents_batch(self, batch: list[np.ndarray]) -> np.ndarray:
        """Encode a batch of document point clouds -> (N, fde_dimension)."""
        return np.stack([self.encode_document(d) for d in batch])

    def __repr__(self) -> str:
        cfg = self._base_config
        pt = cfg["projection_type"]
        extra = ""
        if pt == ProjectionType.LOW_RANK_GAUSSIAN:
            extra = f", simhash_rank={cfg['simhash_rank']}"
        return (
            f"MUVERAEncoder("
            f"dimension={cfg['dimension']}, "
            f"num_simhash_projections={cfg['num_simhash_projections']}, "
            f"num_repetitions={cfg['num_repetitions']}, "
            f"projection_type={pt.name}"
            f"{extra}, "
            f"fde_dimension={self.fde_dimension}"
            f")"
        )
