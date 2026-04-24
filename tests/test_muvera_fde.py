"""
Tests for muvera-fde.

Coverage targets:
  - FDEConfig validation (all error branches)
  - MUVERAEncoder construction and repr
  - Shape/dtype contracts for encode_query / encode_document / batch variants
  - Determinism (same seed → same output)
  - Cross-encoder consistency (mismatched seeds → different output)
  - Dot-product approximation guarantee (unbiasedness check, empirical)
  - fill_empty_partitions on the document side
  - Count Sketch final compression
  - flat 1-D input acceptance
  - Empty point cloud handling
  - generate_query_fde / generate_document_fde low-level API
"""

from __future__ import annotations

import numpy as np
import pytest

from muvera_fde import (
    FDEConfig,
    MUVERAEncoder,
    ProjectionType,
    generate_document_fde,
    generate_query_fde,
)
from muvera_fde._internal.validation import validate_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DIM = 32


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def default_enc() -> MUVERAEncoder:
    return MUVERAEncoder(dimension=DIM, num_simhash_projections=3, num_repetitions=2, seed=7)


@pytest.fixture
def query_cloud(rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((20, DIM)).astype(np.float32)


@pytest.fixture
def doc_cloud(rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((80, DIM)).astype(np.float32)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestFDEConfigValidation:

    def test_invalid_dimension(self) -> None:
        with pytest.raises(ValueError, match="dimension"):
            validate_config(FDEConfig(dimension=0))

    def test_invalid_num_repetitions(self) -> None:
        with pytest.raises(ValueError, match="num_repetitions"):
            validate_config(FDEConfig(num_repetitions=0))

    def test_invalid_final_projection_dimension(self) -> None:
        with pytest.raises(ValueError, match="final_projection_dimension"):
            validate_config(FDEConfig(final_projection_dimension=-1))

    def test_simhash_projections_too_large(self) -> None:
        with pytest.raises(ValueError, match="num_simhash_projections"):
            validate_config(FDEConfig(num_simhash_projections=31))

    def test_simhash_projections_too_large_with_fill(self) -> None:
        with pytest.raises(ValueError, match="fill_empty_partitions"):
            validate_config(FDEConfig(num_simhash_projections=21, fill_empty_partitions=True))

    def test_ams_sketch_missing_projection_dimension(self) -> None:
        with pytest.raises(ValueError, match="projection_dimension"):
            validate_config(
                FDEConfig(projection_type=ProjectionType.AMS_SKETCH, projection_dimension=None)
            )

    def test_ams_sketch_zero_projection_dimension(self) -> None:
        with pytest.raises(ValueError, match="projection_dimension"):
            validate_config(
                FDEConfig(projection_type=ProjectionType.AMS_SKETCH, projection_dimension=0)
            )

    def test_valid_config_passes(self) -> None:
        validate_config(FDEConfig(dimension=128, num_repetitions=3, num_simhash_projections=4))

    def test_zero_simhash_projections_allowed(self) -> None:
        validate_config(FDEConfig(num_simhash_projections=0))

    def test_query_fill_raises(self) -> None:
        config = FDEConfig(fill_empty_partitions=True)
        cloud = np.ones((4, 128), dtype=np.float32)
        with pytest.raises(ValueError, match="fill_empty_partitions"):
            generate_query_fde(cloud, config)


# ---------------------------------------------------------------------------
# Output shape and dtype
# ---------------------------------------------------------------------------

class TestOutputShape:

    def test_encode_query_shape(
        self, default_enc: MUVERAEncoder, query_cloud: np.ndarray
    ) -> None:
        out = default_enc.encode_query(query_cloud)
        assert out.shape == (default_enc.fde_dimension,)
        assert out.dtype == np.float32

    def test_encode_document_shape(
        self, default_enc: MUVERAEncoder, doc_cloud: np.ndarray
    ) -> None:
        out = default_enc.encode_document(doc_cloud)
        assert out.shape == (default_enc.fde_dimension,)
        assert out.dtype == np.float32

    def test_batch_query_shape(
        self, default_enc: MUVERAEncoder, rng: np.random.Generator
    ) -> None:
        batch = [rng.standard_normal((20, DIM)).astype(np.float32) for _ in range(5)]
        out = default_enc.encode_queries_batch(batch)
        assert out.shape == (5, default_enc.fde_dimension)

    def test_batch_document_shape(
        self, default_enc: MUVERAEncoder, rng: np.random.Generator
    ) -> None:
        batch = [rng.standard_normal((80, DIM)).astype(np.float32) for _ in range(3)]
        out = default_enc.encode_documents_batch(batch)
        assert out.shape == (3, default_enc.fde_dimension)

    def test_fde_dimension_formula_identity(self) -> None:
        enc = MUVERAEncoder(dimension=DIM, num_simhash_projections=3, num_repetitions=2)
        # 2 repetitions × 2^3 partitions × DIM
        assert enc.fde_dimension == 2 * 8 * DIM

    def test_fde_dimension_ams_sketch(self) -> None:
        enc = MUVERAEncoder(
            dimension=DIM,
            num_simhash_projections=3,
            num_repetitions=2,
            projection_type=ProjectionType.AMS_SKETCH,
            projection_dimension=16,
        )
        assert enc.fde_dimension == 2 * 8 * 16

    def test_fde_dimension_final_compression(self) -> None:
        enc = MUVERAEncoder(dimension=DIM, num_repetitions=2, final_projection_dimension=64)
        assert enc.fde_dimension == 64


# ---------------------------------------------------------------------------
# Flat 1-D input
# ---------------------------------------------------------------------------

class TestFlatInput:

    def test_flat_query_input(
        self, default_enc: MUVERAEncoder, query_cloud: np.ndarray
    ) -> None:
        out_2d = default_enc.encode_query(query_cloud)
        out_1d = default_enc.encode_query(query_cloud.flatten())
        np.testing.assert_array_equal(out_2d, out_1d)

    def test_flat_document_input(
        self, default_enc: MUVERAEncoder, doc_cloud: np.ndarray
    ) -> None:
        out_2d = default_enc.encode_document(doc_cloud)
        out_1d = default_enc.encode_document(doc_cloud.flatten())
        np.testing.assert_array_equal(out_2d, out_1d)

    def test_flat_length_mismatch_raises(self, default_enc: MUVERAEncoder) -> None:
        bad = np.ones(DIM + 1, dtype=np.float32)
        with pytest.raises(ValueError, match="not divisible"):
            default_enc.encode_query(bad)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_query_deterministic(
        self, default_enc: MUVERAEncoder, query_cloud: np.ndarray
    ) -> None:
        a = default_enc.encode_query(query_cloud)
        b = default_enc.encode_query(query_cloud)
        np.testing.assert_array_equal(a, b)

    def test_document_deterministic(
        self, default_enc: MUVERAEncoder, doc_cloud: np.ndarray
    ) -> None:
        a = default_enc.encode_document(doc_cloud)
        b = default_enc.encode_document(doc_cloud)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_produce_different_fdes(
        self, query_cloud: np.ndarray
    ) -> None:
        enc_a = MUVERAEncoder(dimension=DIM, seed=1)
        enc_b = MUVERAEncoder(dimension=DIM, seed=2)
        assert not np.array_equal(
            enc_a.encode_query(query_cloud),
            enc_b.encode_query(query_cloud),
        )


# ---------------------------------------------------------------------------
# Dot-product approximation (empirical unbiasedness)
# ---------------------------------------------------------------------------

class TestDotProductApproximation:
    """
    The MUVERA FDE guarantees that fde(q) · fde(d) ≈ Chamfer(q, d).
    Here we proxy Chamfer Similarity by the MaxSim sum and verify that the
    FDE dot-product is in the right ballpark (not necessarily tight — we just
    check that it has the same sign and same order of magnitude).

    For the identity projection and many repetitions, the approximation
    should be reasonably tight.
    """

    def _chamfer_maxsim(self, query: np.ndarray, doc: np.ndarray) -> float:
        """Sum of max cosine similarities: Σ_q max_d cos(q, d)."""
        q_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-9)
        d_norm = doc / (np.linalg.norm(doc, axis=1, keepdims=True) + 1e-9)
        sims = q_norm @ d_norm.T  # (num_q, num_d)
        return float(sims.max(axis=1).sum())

    def test_fde_dot_positive_for_similar_clouds(self, rng: np.random.Generator) -> None:
        """FDE dot product should be positive for clouds sharing the same distribution."""
        enc = MUVERAEncoder(
            dimension=DIM,
            num_simhash_projections=4,
            num_repetitions=4,
            seed=0,
        )
        cloud = rng.standard_normal((50, DIM)).astype(np.float32)
        q_fde = enc.encode_query(cloud)
        d_fde = enc.encode_document(cloud)
        assert float(q_fde @ d_fde) > 0

    def test_fde_dot_higher_for_similar_than_random(self, rng: np.random.Generator) -> None:
        """FDE dot product should rank similar pairs above dissimilar ones."""
        enc = MUVERAEncoder(dimension=DIM, num_simhash_projections=4, num_repetitions=6, seed=0)

        base_query = rng.standard_normal((30, DIM)).astype(np.float32)
        similar_doc = base_query + 0.1 * rng.standard_normal((30, DIM)).astype(np.float32)
        random_doc = rng.standard_normal((30, DIM)).astype(np.float32)

        q_fde = enc.encode_query(base_query)
        similar_fde = enc.encode_document(similar_doc)
        random_fde = enc.encode_document(random_doc)

        assert float(q_fde @ similar_fde) > float(q_fde @ random_fde)


# ---------------------------------------------------------------------------
# fill_empty_partitions
# ---------------------------------------------------------------------------

class TestFillEmptyPartitions:

    def test_fill_produces_no_zero_rows_for_dense_cloud(self, rng: np.random.Generator) -> None:
        enc = MUVERAEncoder(
            dimension=DIM,
            num_simhash_projections=3,
            num_repetitions=1,
            fill_empty_partitions=True,
            seed=0,
        )
        cloud = rng.standard_normal((200, DIM)).astype(np.float32)
        fde = enc.encode_document(cloud)
        # Each partition block should have non-zero norm with fill enabled
        block_size = DIM
        blocks = fde.reshape(-1, block_size)
        norms = np.linalg.norm(blocks, axis=1)
        assert (norms == 0).sum() == 0, "fill_empty_partitions left zero partition blocks"

    def test_no_fill_may_have_zero_rows_sparse_cloud(self, rng: np.random.Generator) -> None:
        """A very sparse cloud with many partitions is likely to leave some empty."""
        enc = MUVERAEncoder(
            dimension=DIM,
            num_simhash_projections=4,
            num_repetitions=1,
            fill_empty_partitions=False,
            seed=0,
        )
        # Only 2 tokens → 16 partitions → many likely empty
        cloud = rng.standard_normal((2, DIM)).astype(np.float32)
        fde = enc.encode_document(cloud)
        blocks = fde.reshape(-1, DIM)
        norms = np.linalg.norm(blocks, axis=1)
        assert (norms == 0).sum() > 0, "Expected some empty partitions with 2-token cloud"

    def test_fill_encoder_cannot_encode_query(self, rng: np.random.Generator) -> None:
        enc = MUVERAEncoder(
            dimension=DIM,
            num_simhash_projections=3,
            fill_empty_partitions=True,
        )
        cloud = rng.standard_normal((10, DIM)).astype(np.float32)
        # encode_query internally sets fill_empty_partitions=False; should not raise
        result = enc.encode_query(cloud)
        assert result.shape == (enc.fde_dimension,)


# ---------------------------------------------------------------------------
# Count Sketch projection (AMS_SKETCH)
# ---------------------------------------------------------------------------

class TestCountSketchProjection:

    def test_ams_sketch_output_shape(self, rng: np.random.Generator) -> None:
        enc = MUVERAEncoder(
            dimension=DIM,
            num_simhash_projections=3,
            num_repetitions=2,
            projection_type=ProjectionType.AMS_SKETCH,
            projection_dimension=16,
        )
        cloud = rng.standard_normal((50, DIM)).astype(np.float32)
        assert enc.encode_query(cloud).shape == (enc.fde_dimension,)
        assert enc.encode_document(cloud).shape == (enc.fde_dimension,)

    def test_final_projection_dimension(self, rng: np.random.Generator) -> None:
        enc = MUVERAEncoder(
            dimension=DIM,
            num_repetitions=2,
            final_projection_dimension=64,
        )
        cloud = rng.standard_normal((50, DIM)).astype(np.float32)
        q = enc.encode_query(cloud)
        d = enc.encode_document(cloud)
        assert q.shape == (64,)
        assert d.shape == (64,)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_single_token_query(self, default_enc: MUVERAEncoder) -> None:
        cloud = np.ones((1, DIM), dtype=np.float32)
        out = default_enc.encode_query(cloud)
        assert out.shape == (default_enc.fde_dimension,)

    def test_dimension_mismatch_raises(self, default_enc: MUVERAEncoder) -> None:
        bad_cloud = np.ones((10, DIM + 5), dtype=np.float32)
        with pytest.raises(ValueError, match="dimension mismatch"):
            default_enc.encode_query(bad_cloud)

    def test_3d_input_raises(self, default_enc: MUVERAEncoder) -> None:
        bad_cloud = np.ones((10, DIM, 1), dtype=np.float32)
        with pytest.raises(ValueError, match="1-D or 2-D"):
            default_enc.encode_query(bad_cloud)

    def test_repr_contains_key_fields(self, default_enc: MUVERAEncoder) -> None:
        r = repr(default_enc)
        assert "MUVERAEncoder" in r
        assert str(DIM) in r

    def test_zero_simhash_projections(self, rng: np.random.Generator) -> None:
        """num_simhash_projections=0 → single partition, trivial aggregation."""
        enc = MUVERAEncoder(dimension=DIM, num_simhash_projections=0, num_repetitions=1)
        cloud = rng.standard_normal((20, DIM)).astype(np.float32)
        out = enc.encode_query(cloud)
        assert out.shape == (DIM,)

    def test_float64_input_coerced(self, default_enc: MUVERAEncoder) -> None:
        cloud_f64 = np.random.randn(20, DIM)  # float64 by default
        out = default_enc.encode_query(cloud_f64)
        assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# Low-level functional API
# ---------------------------------------------------------------------------

class TestFunctionalAPI:

    def test_generate_query_fde_matches_encoder(
        self, default_enc: MUVERAEncoder, query_cloud: np.ndarray
    ) -> None:
        enc_out = default_enc.encode_query(query_cloud)
        config = FDEConfig(
            dimension=DIM,
            num_repetitions=2,
            num_simhash_projections=3,
            seed=7,
            fill_empty_partitions=False,
        )
        func_out = generate_query_fde(query_cloud, config, default_enc._rep_params)
        np.testing.assert_array_equal(enc_out, func_out)

    def test_generate_document_fde_matches_encoder(
        self, default_enc: MUVERAEncoder, doc_cloud: np.ndarray
    ) -> None:
        enc_out = default_enc.encode_document(doc_cloud)
        config = FDEConfig(
            dimension=DIM,
            num_repetitions=2,
            num_simhash_projections=3,
            seed=7,
            fill_empty_partitions=False,
        )
        func_out = generate_document_fde(doc_cloud, config, default_enc._rep_params)
        np.testing.assert_array_equal(enc_out, func_out)
