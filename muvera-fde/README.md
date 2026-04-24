# muvera-fde

**Fixed Dimensional Encodings for multi-vector retrieval.**

[![PyPI](https://img.shields.io/pypi/v/muvera-fde)](https://pypi.org/project/muvera-fde/)
[![Python](https://img.shields.io/pypi/pyversions/muvera-fde)](https://pypi.org/project/muvera-fde/)
[![CI](https://github.com/smarthi/muvera-fde/actions/workflows/ci.yml/badge.svg)](https://github.com/smarthi/muvera-fde/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A Python port of Google's graph-mining MUVERA implementation.  
Paper: [MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings](https://arxiv.org/abs/2405.19504) (Rajput et al., 2024).

---

## What is MUVERA?

Late-interaction retrieval models like **ColBERT**, **ColPali**, and **ColQwen2**
represent each query and document as a *variable-length set* of token embeddings
rather than a single vector.  Scoring two sets requires the computationally
expensive **MaxSim** (or Chamfer Similarity) operation:

```
Chamfer(Q, D) = Σ_{q ∈ Q} max_{d ∈ D} cos(q, d)
```

This makes large-scale approximate nearest-neighbour (ANN) retrieval
impractical with standard indexes.

MUVERA solves this by converting each multi-vector set into a **single
fixed-dimensional vector** (an FDE) such that:

```
fde_query(Q) · fde_doc(D)  ≈  Chamfer(Q, D)
```

The FDE dot product can then be indexed with any standard ANN library
(FAISS, ScaNN, OpenSearch, etc.), bringing sub-linear retrieval back to
late-interaction models.

---

## Installation

```bash
pip install muvera-fde
```

Requires Python ≥ 3.10, NumPy ≥ 1.24, Pydantic ≥ 2.0.

---

## Quick start

```python
import numpy as np
from muvera_fde import MUVERAEncoder

# Same encoder instance for queries AND documents (shared seed / partition structure)
enc = MUVERAEncoder(
    dimension=128,              # ColBERT / ColQwen2 embedding dimension
    num_simhash_projections=4,  # 2^4 = 16 partitions per repetition
    num_repetitions=2,          # 2 independent repetitions
    seed=42,
)

print(enc)
# MUVERAEncoder(dimension=128, num_simhash_projections=4, num_repetitions=2,
#               projection_type=DEFAULT_IDENTITY, fde_dimension=4096)

# Simulated ColQwen2-style token embeddings
query_tokens = np.random.randn(32,  128).astype(np.float32)
doc_tokens   = np.random.randn(512, 128).astype(np.float32)

q_fde = enc.encode_query(query_tokens)    # shape: (4096,)
d_fde = enc.encode_document(doc_tokens)   # shape: (4096,)

# Approximate Chamfer Similarity — feed directly into FAISS / OpenSearch
score = float(q_fde @ d_fde)
```

---

## Key concepts

### Query vs. Document encoding

| Side     | Aggregation | fill_empty_partitions |
|----------|-------------|-----------------------|
| Query    | **SUM** — token embeddings summed into their SimHash partition | always `False` |
| Document | **AVERAGE** — centroid of all tokens per partition | optional `True` |

### SimHash partitioning

Each token embedding is assigned to one of `2**num_simhash_projections`
partitions based on the sign pattern of its projection onto `k` random
Gaussian vectors.  Adjacent partitions in Gray-code order correspond to
geometrically similar regions of embedding space.

### Fill empty partitions (document side)

With few document tokens relative to `2**k` partitions, many partition
slots will be empty.  Setting `fill_empty_partitions=True` copies the
projection of the nearest token (by SimHash Hamming distance) into each
empty slot, improving recall at the cost of a small amount of extra
compute.

```python
enc = MUVERAEncoder(
    dimension=128,
    num_simhash_projections=4,
    num_repetitions=4,
    fill_empty_partitions=True,  # document side only
)
```

### Count Sketch projection (smaller FDEs)

When FDE dimension is too large for your index, two compression options
are available:

**Per-repetition projection** — reduces the per-partition width before
accumulation:

```python
from muvera_fde import ProjectionType

enc = MUVERAEncoder(
    dimension=128,
    projection_type=ProjectionType.AMS_SKETCH,
    projection_dimension=32,          # 128 → 32 per partition
    num_simhash_projections=4,
    num_repetitions=4,
)
# fde_dimension = 4 × 16 × 32 = 2048 instead of 4 × 16 × 128 = 8192
```

**Post-accumulation compression** — compresses the full FDE after all
repetitions:

```python
enc = MUVERAEncoder(
    dimension=128,
    num_simhash_projections=4,
    num_repetitions=4,
    final_projection_dimension=512,   # compress 8192 → 512
)
```

Both compressions use Count Sketch, which preserves dot products in
expectation: `E[⟨sketch(x), sketch(y)⟩] = ⟨x, y⟩`.

### Batch encoding

```python
queries   = [np.random.randn(32,  128).astype(np.float32) for _ in range(100)]
documents = [np.random.randn(512, 128).astype(np.float32) for _ in range(1000)]

Q = enc.encode_queries_batch(queries)    # shape: (100, fde_dimension)
D = enc.encode_documents_batch(documents) # shape: (1000, fde_dimension)

# All-pairs approximate Chamfer Similarities
scores = Q @ D.T  # shape: (100, 1000)
```

### Low-level functional API

```python
from muvera_fde import FDEConfig, generate_query_fde, generate_document_fde

config = FDEConfig(
    dimension=128,
    num_repetitions=2,
    num_simhash_projections=4,
    seed=42,
)

q_fde = generate_query_fde(query_tokens, config)
d_fde = generate_document_fde(doc_tokens, config)
```

---

## Two-stage retrieval pipeline

The intended production pattern for ColQwen2 / ColBERT:

```
                   ┌─────────────────────────────────────┐
                   │  Offline indexing                   │
                   │                                     │
  doc token embs ──► encode_document()  ──► FDE vector ──► ANN index
                   └─────────────────────────────────────┘

                   ┌─────────────────────────────────────┐
                   │  Online retrieval                   │
  query tokens  ──► encode_query()  ──► FDE vector       │
                   │        │                            │
                   │        ▼                            │
                   │   ANN search  ──► top-K candidates  │
                   │        │                            │
                   │        ▼                            │
                   │   MaxSim re-rank  ──► final top-K   │
                   └─────────────────────────────────────┘
```

Stage 1 (ANN on FDE vectors) is sub-linear and eliminates 99%+ of candidates.  
Stage 2 (exact MaxSim on raw token embeddings) reranks the small candidate set for full accuracy.

---

## API reference

### `MUVERAEncoder`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dimension` | `int` | 128 | Token embedding dimension |
| `num_simhash_projections` | `int` | 4 | SimHash bits *k*; partitions = 2^k |
| `num_repetitions` | `int` | 1 | Independent repetitions |
| `seed` | `int` | 1 | Shared RNG seed |
| `projection_type` | `ProjectionType` | `DEFAULT_IDENTITY` | Identity or Count Sketch |
| `projection_dimension` | `int \| None` | None | Required for `AMS_SKETCH` |
| `fill_empty_partitions` | `bool` | False | Fill empty doc-side partitions |
| `final_projection_dimension` | `int \| None` | None | Post-accumulation compression |

**Properties:** `fde_dimension`  
**Methods:** `encode_query`, `encode_document`, `encode_queries_batch`, `encode_documents_batch`

---

## Attribution

This library is a faithful Python port of the C++ implementation in
[Google's graph-mining project](https://github.com/google/graph-mining/tree/main/sketching/point_cloud),
originally licensed under Apache 2.0.

See [NOTICE](NOTICE) for the full upstream attribution.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
