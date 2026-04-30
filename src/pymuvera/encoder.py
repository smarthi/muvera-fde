"""
High-level encoder for muvera-fde.
"""

from __future__ import annotations

import numpy as np

from pymuvera._internal.params import RepParams, build_rep_params
from pymuvera._internal.validation import (
    _projection_dim_for_config,
    checked_intermediate_fde_length,
    num_partitions_for_config,
    validate_config,
)
from pymuvera.config import FDEConfig, ProjectionType
from pymuvera.core import (
    _use_cross_polytope,
    _use_identity,
    _use_low_rank_simhash,
    _use_srht,
    generate_document_fde,
    generate_query_fde,
)


class MUVERAEncoder:
    """Encodes multi-vector token embeddings into fixed-dimensional FDE vectors.

    Use the **same encoder instance** for both queries and documents.

    Parameters
    ----------
    dimension:
        Token embedding dimension. ColQwen2=128, ColQwen3.5 v3=320.
    num_simhash_projections:
        SimHash bits k; partitions = 2^k.  Ignored for CROSS_POLYTOPE.
    num_repetitions:
        Independent repetitions.
    seed:
        Shared RNG seed -- must match query and document sides.
    projection_type:
        DEFAULT_IDENTITY, AMS_SKETCH, LOW_RANK_GAUSSIAN, SRHT, or CROSS_POLYTOPE.
    projection_dimension:
        Required for AMS_SKETCH.
    simhash_rank:
        r for LOW_RANK_GAUSSIAN; 1 <= r < k.
    fill_empty_partitions:
        Document side only -- fill empty partition slots.
    densifying_fill:
        When fill_empty_partitions=True, use O(num_empty) Densifying LSH fill
        (Shrivastava, 2014) instead of O(N*k) Hamming NN fill.
        Automatically forced True for CROSS_POLYTOPE.
    final_projection_dimension:
        Post-accumulation Count Sketch compression.

    Examples
    --------
    Cross-Polytope LSH (theoretically optimal cosine partitioning)::

        enc = MUVERAEncoder(
            dimension=128,
            num_repetitions=4,
            projection_type=ProjectionType.CROSS_POLYTOPE,
            fill_empty_partitions=True,   # densifying fill used automatically
        )
        # num_partitions = 2 * 128 = 256 per repetition
        # fde_dimension = 4 * 256 * 128 = 131,072

    Cross-Polytope + Densifying fill for ColQwen3.5 (d=320)::

        enc = MUVERAEncoder(
            dimension=320,
            num_repetitions=8,
            projection_type=ProjectionType.CROSS_POLYTOPE,
            fill_empty_partitions=True,
            final_projection_dimension=81920,
        )

    Densifying fill with DEFAULT_IDENTITY (faster fill for large k)::

        enc = MUVERAEncoder(
            dimension=128,
            num_simhash_projections=10,
            num_repetitions=4,
            fill_empty_partitions=True,
            densifying_fill=True,   # O(num_empty) instead of O(N*k)
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
        densifying_fill: bool = False,
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
            densifying_fill=densifying_fill,
            final_projection_dimension=final_projection_dimension,
        )
        self.fill_empty_partitions = fill_empty_partitions

        config = FDEConfig(**self._base_config, fill_empty_partitions=fill_empty_partitions)
        validate_config(config)

        _proj_dim = _projection_dim_for_config(config)
        _num_parts = num_partitions_for_config(config)
        checked_intermediate_fde_length(config, _proj_dim, _num_parts)

        self._rep_params: list[RepParams] = [
            build_rep_params(
                seed + rep,
                dimension,
                _proj_dim,
                num_simhash_projections,
                _use_identity(config),
                use_low_rank_simhash=_use_low_rank_simhash(config),
                simhash_rank=simhash_rank,
                use_srht=_use_srht(config),
                use_cross_polytope=_use_cross_polytope(config),
            )
            for rep in range(num_repetitions)
        ]

    @property
    def fde_dimension(self) -> int:
        """Output dimension of every FDE vector.

        For sign-based modes:  num_repetitions * 2^k * projection_dim.
        For CROSS_POLYTOPE:    num_repetitions * 2*padded_dim * projection_dim.
        With final compression: final_projection_dimension.
        """
        if self._base_config["final_projection_dimension"] is not None:
            return self._base_config["final_projection_dimension"]
        config = FDEConfig(**self._base_config, fill_empty_partitions=self.fill_empty_partitions)
        proj_dim = _projection_dim_for_config(config)
        num_parts = num_partitions_for_config(config)
        return self._base_config["num_repetitions"] * num_parts * proj_dim

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
        extras = []
        if pt == ProjectionType.LOW_RANK_GAUSSIAN:
            extras.append(f"simhash_rank={cfg['simhash_rank']}")
        if cfg.get("densifying_fill"):
            extras.append("densifying_fill=True")
        extra_str = (", " + ", ".join(extras)) if extras else ""
        return (
            f"MUVERAEncoder("
            f"dimension={cfg['dimension']}, "
            f"num_simhash_projections={cfg['num_simhash_projections']}, "
            f"num_repetitions={cfg['num_repetitions']}, "
            f"projection_type={pt.name}"
            f"{extra_str}, "
            f"fde_dimension={self.fde_dimension}"
            f")"
        )
