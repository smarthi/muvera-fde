"""
Tests for muvera-fde.

Coverage targets:
  - FDEConfig validation (all error branches, including LOW_RANK_GAUSSIAN)
  - MUVERAEncoder construction and repr
  - Shape/dtype contracts for encode_query / encode_document / batch variants
  - Determinism (same seed -> same output)
  - Cross-encoder consistency (mismatched seeds -> different output)
  - Dot-product approximation guarantee (unbiasedness check, empirical)
  - fill_empty_partitions on the document side
  - Count Sketch final compression
  - flat 1-D input acceptance
  - Empty point cloud handling
  - generate_query_fde / generate_document_fde low-level API
  - LOW_RANK_GAUSSIAN SimHash: shape, determinism, convergence, validation
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
    def test_encode_query_shape(self, default_enc: MUVERAEncoder, query_cloud: np.ndarray) -> None:
        out = default_enc.encode_query(query_cloud)
        assert out.shape == (default_enc.fde_dimension,)
        assert out.dtype == np.float32

    def test_encode_document_shape(self, default_enc: MUVERAEncoder, doc_cloud: np.ndarray) -> None:
        out = default_enc.encode_document(doc_cloud)
        assert out.shape == (default_enc.fde_dimension,)
        assert out.dtype == np.float32

    def test_batch_query_shape(self, default_enc: MUVERAEncoder, rng: np.random.Generator) -> None:
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
    def test_flat_query_input(self, default_enc: MUVERAEncoder, query_cloud: np.ndarray) -> None:
        out_2d = default_enc.encode_query(query_cloud)
        out_1d = default_enc.encode_query(query_cloud.flatten())
        np.testing.assert_array_equal(out_2d, out_1d)

    def test_flat_document_input(self, default_enc: MUVERAEncoder, doc_cloud: np.ndarray) -> None:
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
    def test_query_deterministic(self, default_enc: MUVERAEncoder, query_cloud: np.ndarray) -> None:
        a = default_enc.encode_query(query_cloud)
        b = default_enc.encode_query(query_cloud)
        np.testing.assert_array_equal(a, b)

    def test_document_deterministic(
        self, default_enc: MUVERAEncoder, doc_cloud: np.ndarray
    ) -> None:
        a = default_enc.encode_document(doc_cloud)
        b = default_enc.encode_document(doc_cloud)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_produce_different_fdes(self, query_cloud: np.ndarray) -> None:
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
    def test_fde_dot_positive_for_similar_clouds(self, rng: np.random.Generator) -> None:
        enc = MUVERAEncoder(dimension=DIM, num_simhash_projections=4, num_repetitions=4, seed=0)
        cloud = rng.standard_normal((50, DIM)).astype(np.float32)
        q_fde = enc.encode_query(cloud)
        d_fde = enc.encode_document(cloud)
        assert float(q_fde @ d_fde) > 0

    def test_fde_dot_higher_for_similar_than_random(self, rng: np.random.Generator) -> None:
        enc = MUVERAEncoder(dimension=DIM, num_simhash_projections=4, num_repetitions=6, seed=0)
        base_query = rng.standard_normal((30, DIM)).astype(np.float32)
        similar_doc = base_query + 0.1 * rng.standard_normal((30, DIM)).astype(np.float32)
        random_doc = rng.standard_normal((30, DIM)).astype(np.float32)
        q_fde = enc.encode_query(base_query)
        assert float(q_fde @ enc.encode_document(similar_doc)) > float(
            q_fde @ enc.encode_document(random_doc)
        )


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
        norms = np.linalg.norm(fde.reshape(-1, DIM), axis=1)
        assert (norms == 0).sum() == 0

    def test_no_fill_may_have_zero_rows_sparse_cloud(self, rng: np.random.Generator) -> None:
        enc = MUVERAEncoder(
            dimension=DIM,
            num_simhash_projections=4,
            num_repetitions=1,
            fill_empty_partitions=False,
            seed=0,
        )
        cloud = rng.standard_normal((2, DIM)).astype(np.float32)
        fde = enc.encode_document(cloud)
        norms = np.linalg.norm(fde.reshape(-1, DIM), axis=1)
        assert (norms == 0).sum() > 0


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
        enc = MUVERAEncoder(dimension=DIM, num_repetitions=2, final_projection_dimension=64)
        cloud = rng.standard_normal((50, DIM)).astype(np.float32)
        assert enc.encode_query(cloud).shape == (64,)
        assert enc.encode_document(cloud).shape == (64,)


# ---------------------------------------------------------------------------
# LOW_RANK_GAUSSIAN SimHash
# ---------------------------------------------------------------------------


class TestLowRankGaussianSimHash:
    """Tests for the EGGROLL-inspired low-rank SimHash factorisation."""

    def _lr_enc(self, rank: int = 3, k: int = 4, reps: int = 2) -> MUVERAEncoder:
        return MUVERAEncoder(
            dimension=DIM,
            num_simhash_projections=k,
            num_repetitions=reps,
            projection_type=ProjectionType.LOW_RANK_GAUSSIAN,
            simhash_rank=rank,
            seed=99,
        )

    def test_output_shape(self, rng: np.random.Generator) -> None:
        enc = self._lr_enc()
        cloud = rng.standard_normal((50, DIM)).astype(np.float32)
        # fde_dimension = reps x 2^k x dimension (same formula as DEFAULT_IDENTITY)
        expected = 2 * (1 << 4) * DIM
        assert enc.fde_dimension == expected
        assert enc.encode_query(cloud).shape == (expected,)
        assert enc.encode_document(cloud).shape == (expected,)

    def test_dtype(self, rng: np.random.Generator) -> None:
        enc = self._lr_enc()
        cloud = rng.standard_normal((30, DIM)).astype(np.float32)
        assert enc.encode_query(cloud).dtype == np.float32
        assert enc.encode_document(cloud).dtype == np.float32

    def test_deterministic(self, rng: np.random.Generator) -> None:
        enc = self._lr_enc()
        cloud = rng.standard_normal((30, DIM)).astype(np.float32)
        np.testing.assert_array_equal(enc.encode_query(cloud), enc.encode_query(cloud))
        np.testing.assert_array_equal(enc.encode_document(cloud), enc.encode_document(cloud))

    def test_different_ranks_give_different_fdes(self, rng: np.random.Generator) -> None:
        cloud = rng.standard_normal((40, DIM)).astype(np.float32)
        enc1 = self._lr_enc(rank=1)
        enc2 = self._lr_enc(rank=3)
        # Different random factors A, B -> different FDE vectors
        assert not np.array_equal(enc1.encode_query(cloud), enc2.encode_query(cloud))

    def test_different_seeds_give_different_fdes(self, rng: np.random.Generator) -> None:
        cloud = rng.standard_normal((40, DIM)).astype(np.float32)
        enc1 = MUVERAEncoder(
            dimension=DIM,
            num_simhash_projections=4,
            projection_type=ProjectionType.LOW_RANK_GAUSSIAN,
            simhash_rank=3,
            seed=1,
        )
        enc2 = MUVERAEncoder(
            dimension=DIM,
            num_simhash_projections=4,
            projection_type=ProjectionType.LOW_RANK_GAUSSIAN,
            simhash_rank=3,
            seed=2,
        )
        assert not np.array_equal(enc1.encode_query(cloud), enc2.encode_query(cloud))

    def test_invalid_rank_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="simhash_rank must be positive"):
            MUVERAEncoder(
                dimension=DIM,
                num_simhash_projections=4,
                projection_type=ProjectionType.LOW_RANK_GAUSSIAN,
                simhash_rank=0,
            )

    def test_invalid_rank_gte_k_raises(self) -> None:
        # rank must be < num_simhash_projections
        with pytest.raises(ValueError, match="strictly less than"):
            MUVERAEncoder(
                dimension=DIM,
                num_simhash_projections=4,
                projection_type=ProjectionType.LOW_RANK_GAUSSIAN,
                simhash_rank=4,  # equal to k -- invalid
            )

    def test_lr_approximates_chamfer_similarity(self, rng: np.random.Generator) -> None:
        """LR SimHash should approximate Chamfer Similarity -- similar > random.

        EGGROLL Theorem 4 proves convergence of the sign-pattern distribution to
        the full-rank Gaussian at O(r^-1).  The practical consequence is that the
        FDE dot product should still rank similar docs above random docs.  This is
        tested independently of the full-rank encoder since the convergence is
        distributional (over many draws of A, B, W), not sample-wise between two
        different random instances.
        """
        enc = MUVERAEncoder(
            dimension=DIM,
            num_simhash_projections=5,
            num_repetitions=6,
            projection_type=ProjectionType.LOW_RANK_GAUSSIAN,
            simhash_rank=4,
            seed=0,
        )
        base_query = rng.standard_normal((40, DIM)).astype(np.float32)
        similar_doc = base_query + 0.1 * rng.standard_normal((40, DIM)).astype(np.float32)
        random_doc = rng.standard_normal((40, DIM)).astype(np.float32)

        q_fde = enc.encode_query(base_query)
        score_similar = float(q_fde @ enc.encode_document(similar_doc))
        score_random = float(q_fde @ enc.encode_document(random_doc))
        assert score_similar > score_random, (
            f"LR SimHash failed to rank similar ({score_similar:.3f}) "
            f"above random ({score_random:.3f})"
        )

    def test_lr_positive_score_for_same_cloud(self, rng: np.random.Generator) -> None:
        """FDE dot product should be positive when query and doc share the same tokens."""
        enc = MUVERAEncoder(
            dimension=DIM,
            num_simhash_projections=4,
            num_repetitions=4,
            projection_type=ProjectionType.LOW_RANK_GAUSSIAN,
            simhash_rank=3,
            seed=0,
        )
        cloud = rng.standard_normal((50, DIM)).astype(np.float32)
        assert float(enc.encode_query(cloud) @ enc.encode_document(cloud)) > 0

    def test_fill_empty_partitions_compatible(self, rng: np.random.Generator) -> None:
        enc = MUVERAEncoder(
            dimension=DIM,
            num_simhash_projections=4,
            num_repetitions=2,
            projection_type=ProjectionType.LOW_RANK_GAUSSIAN,
            simhash_rank=3,
            fill_empty_partitions=True,
            seed=0,
        )
        cloud = rng.standard_normal((200, DIM)).astype(np.float32)
        fde = enc.encode_document(cloud)
        assert fde.shape == (enc.fde_dimension,)
        assert fde.dtype == np.float32

    def test_repr_includes_simhash_rank(self) -> None:
        enc = self._lr_enc(rank=4, k=5)
        r = repr(enc)
        assert "LOW_RANK_GAUSSIAN" in r
        assert "simhash_rank=4" in r

    def test_repr_omits_simhash_rank_for_identity(self) -> None:
        enc = MUVERAEncoder(dimension=DIM)
        assert "simhash_rank" not in repr(enc)

    def test_batch_shapes(self, rng: np.random.Generator) -> None:
        enc = self._lr_enc()
        qs = [rng.standard_normal((20, DIM)).astype(np.float32) for _ in range(4)]
        ds = [rng.standard_normal((80, DIM)).astype(np.float32) for _ in range(6)]
        assert enc.encode_queries_batch(qs).shape == (4, enc.fde_dimension)
        assert enc.encode_documents_batch(ds).shape == (6, enc.fde_dimension)

    def test_zero_simhash_projections_with_low_rank(self, rng: np.random.Generator) -> None:
        # num_simhash_projections=0 means single partition; low-rank factors
        # are never built (rank constraint doesn't apply when k=0)
        enc = MUVERAEncoder(
            dimension=DIM,
            num_simhash_projections=0,
            projection_type=ProjectionType.LOW_RANK_GAUSSIAN,
            simhash_rank=1,
        )
        cloud = rng.standard_normal((20, DIM)).astype(np.float32)
        out = enc.encode_query(cloud)
        assert out.shape == (DIM,)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_token_query(self, default_enc: MUVERAEncoder) -> None:
        out = default_enc.encode_query(np.ones((1, DIM), dtype=np.float32))
        assert out.shape == (default_enc.fde_dimension,)

    def test_dimension_mismatch_raises(self, default_enc: MUVERAEncoder) -> None:
        with pytest.raises(ValueError, match="dimension mismatch"):
            default_enc.encode_query(np.ones((10, DIM + 5), dtype=np.float32))

    def test_3d_input_raises(self, default_enc: MUVERAEncoder) -> None:
        with pytest.raises(ValueError, match="1-D or 2-D"):
            default_enc.encode_query(np.ones((10, DIM, 1), dtype=np.float32))

    def test_repr_contains_key_fields(self, default_enc: MUVERAEncoder) -> None:
        assert "MUVERAEncoder" in repr(default_enc)
        assert str(DIM) in repr(default_enc)

    def test_zero_simhash_projections(self, rng: np.random.Generator) -> None:
        enc = MUVERAEncoder(dimension=DIM, num_simhash_projections=0, num_repetitions=1)
        out = enc.encode_query(rng.standard_normal((20, DIM)).astype(np.float32))
        assert out.shape == (DIM,)

    def test_float64_input_coerced(self, default_enc: MUVERAEncoder) -> None:
        out = default_enc.encode_query(np.random.randn(20, DIM))
        assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# Low-level functional API
# ---------------------------------------------------------------------------


class TestFunctionalAPI:
    def test_generate_query_fde_matches_encoder(
        self, default_enc: MUVERAEncoder, query_cloud: np.ndarray
    ) -> None:
        config = FDEConfig(
            dimension=DIM,
            num_repetitions=2,
            num_simhash_projections=3,
            seed=7,
            fill_empty_partitions=False,
        )
        np.testing.assert_array_equal(
            default_enc.encode_query(query_cloud),
            generate_query_fde(query_cloud, config, default_enc._rep_params),
        )

    def test_generate_document_fde_matches_encoder(
        self, default_enc: MUVERAEncoder, doc_cloud: np.ndarray
    ) -> None:
        config = FDEConfig(
            dimension=DIM,
            num_repetitions=2,
            num_simhash_projections=3,
            seed=7,
            fill_empty_partitions=False,
        )
        np.testing.assert_array_equal(
            default_enc.encode_document(doc_cloud),
            generate_document_fde(doc_cloud, config, default_enc._rep_params),
        )


# ---------------------------------------------------------------------------
# SRHT SimHash
# ---------------------------------------------------------------------------


class TestSRHTProjection:
    """Tests for the Subsampled Randomized Hadamard Transform SimHash."""

    def _srht_enc(self, k: int = 4, reps: int = 2) -> MUVERAEncoder:
        return MUVERAEncoder(
            dimension=DIM,
            num_simhash_projections=k,
            num_repetitions=reps,
            projection_type=ProjectionType.SRHT,
            seed=77,
        )

    def test_output_shape(self, rng: np.random.Generator) -> None:
        enc = self._srht_enc()
        cloud = rng.standard_normal((50, DIM)).astype(np.float32)
        # fde_dimension = reps x 2^k x dimension (same as DEFAULT_IDENTITY)
        expected = 2 * (1 << 4) * DIM
        assert enc.fde_dimension == expected
        assert enc.encode_query(cloud).shape == (expected,)
        assert enc.encode_document(cloud).shape == (expected,)

    def test_dtype(self, rng: np.random.Generator) -> None:
        enc = self._srht_enc()
        cloud = rng.standard_normal((30, DIM)).astype(np.float32)
        assert enc.encode_query(cloud).dtype == np.float32
        assert enc.encode_document(cloud).dtype == np.float32

    def test_deterministic(self, rng: np.random.Generator) -> None:
        enc = self._srht_enc()
        cloud = rng.standard_normal((30, DIM)).astype(np.float32)
        np.testing.assert_array_equal(enc.encode_query(cloud), enc.encode_query(cloud))
        np.testing.assert_array_equal(enc.encode_document(cloud), enc.encode_document(cloud))

    def test_different_seeds_give_different_fdes(self, rng: np.random.Generator) -> None:
        cloud = rng.standard_normal((40, DIM)).astype(np.float32)
        enc1 = MUVERAEncoder(
            dimension=DIM,
            num_simhash_projections=4,
            projection_type=ProjectionType.SRHT,
            seed=1,
        )
        enc2 = MUVERAEncoder(
            dimension=DIM,
            num_simhash_projections=4,
            projection_type=ProjectionType.SRHT,
            seed=2,
        )
        assert not np.array_equal(enc1.encode_query(cloud), enc2.encode_query(cloud))

    def test_k_exceeds_padded_dim_raises(self) -> None:
        # dimension=2 -> padded_dim=2; k=3 > 2 but k=3 <= MAX(30), so only _check_srht fires
        with pytest.raises(ValueError, match="SRHT requires"):
            MUVERAEncoder(
                dimension=2,
                num_simhash_projections=3,  # > next_power_of_2(2) = 2
                projection_type=ProjectionType.SRHT,
            )

    def test_approximates_chamfer_similarity(self, rng: np.random.Generator) -> None:
        """SRHT SimHash should rank similar docs above random docs."""
        enc = MUVERAEncoder(
            dimension=DIM,
            num_simhash_projections=4,
            num_repetitions=6,
            projection_type=ProjectionType.SRHT,
            seed=0,
        )
        base_query = rng.standard_normal((40, DIM)).astype(np.float32)
        similar_doc = base_query + 0.1 * rng.standard_normal((40, DIM)).astype(np.float32)
        random_doc = rng.standard_normal((40, DIM)).astype(np.float32)
        q_fde = enc.encode_query(base_query)
        assert float(q_fde @ enc.encode_document(similar_doc)) > float(
            q_fde @ enc.encode_document(random_doc)
        )

    def test_positive_score_same_cloud(self, rng: np.random.Generator) -> None:
        enc = self._srht_enc(k=4, reps=4)
        cloud = rng.standard_normal((50, DIM)).astype(np.float32)
        assert float(enc.encode_query(cloud) @ enc.encode_document(cloud)) > 0

    def test_fill_empty_partitions_compatible(self, rng: np.random.Generator) -> None:
        enc = MUVERAEncoder(
            dimension=DIM,
            num_simhash_projections=4,
            num_repetitions=2,
            projection_type=ProjectionType.SRHT,
            fill_empty_partitions=True,
            seed=0,
        )
        cloud = rng.standard_normal((200, DIM)).astype(np.float32)
        fde = enc.encode_document(cloud)
        assert fde.shape == (enc.fde_dimension,)
        assert fde.dtype == np.float32

    def test_batch_shapes(self, rng: np.random.Generator) -> None:
        enc = self._srht_enc()
        qs = [rng.standard_normal((20, DIM)).astype(np.float32) for _ in range(4)]
        ds = [rng.standard_normal((80, DIM)).astype(np.float32) for _ in range(6)]
        assert enc.encode_queries_batch(qs).shape == (4, enc.fde_dimension)
        assert enc.encode_documents_batch(ds).shape == (6, enc.fde_dimension)

    def test_non_power_of_2_dimension_padded(self, rng: np.random.Generator) -> None:
        """d=24 (not power of 2) is padded to 32 -- k <= 32 must hold."""
        enc = MUVERAEncoder(
            dimension=24,
            num_simhash_projections=4,
            projection_type=ProjectionType.SRHT,
            seed=0,
        )
        cloud = rng.standard_normal((30, 24)).astype(np.float32)
        out = enc.encode_query(cloud)
        # fde_dimension = 1 x 2^4 x 24 = 384
        assert out.shape == (1 * 16 * 24,)

    def test_repr_shows_srht(self) -> None:
        enc = self._srht_enc()
        assert "SRHT" in repr(enc)

    def test_zero_simhash_projections_srht(self, rng: np.random.Generator) -> None:
        """k=0 with SRHT: no SRHT params built, single partition."""
        enc = MUVERAEncoder(
            dimension=DIM,
            num_simhash_projections=0,
            projection_type=ProjectionType.SRHT,
        )
        cloud = rng.standard_normal((20, DIM)).astype(np.float32)
        assert enc.encode_query(cloud).shape == (DIM,)

    def test_fwht_correctness(self) -> None:
        """Verify the butterfly FWHT against a known 4-element transform."""
        from muvera_fde._internal.sketch import _fwht_batch

        x = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        result = _fwht_batch(x)
        # H @ e_0 = [1, 1, 1, 1] (unnormalised)
        np.testing.assert_allclose(result, [[1.0, 1.0, 1.0, 1.0]], atol=1e-5)

        x2 = np.array([[1.0, -1.0, 1.0, -1.0]], dtype=np.float32)
        result2 = _fwht_batch(x2)
        # H @ [1,-1,1,-1] = [0, 4, 0, 0] in Hadamard-ordered butterfly
        np.testing.assert_allclose(result2, [[0.0, 4.0, 0.0, 0.0]], atol=1e-5)
