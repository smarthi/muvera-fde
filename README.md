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
rather than a single vector. Scoring two sets requires the computationally
expensive **MaxSim** (Chamfer Similarity) operation:

```
Chamfer(Q, D) = Σ_{q ∈ Q} max_{d ∈ D} cos(q, d)
```

This makes large-scale ANN retrieval impractical with standard indexes.

MUVERA solves this by converting each multi-vector set into a **single
fixed-dimensional vector** (FDE) such that:

```
fde_query(Q) · fde_doc(D)  ≈  Chamfer(Q, D)
```

Standard ANN libraries (FAISS, ScaNN, OpenSearch k-NN) can then index FDE
vectors directly, restoring sub-linear retrieval for late-interaction models.

---

## Installation

```bash
pip install muvera-fde
```

Requires Python ≥ 3.12, NumPy ≥ 1.24, Pydantic ≥ 2.0.

---

## Quick start

```python
import numpy as np
from muvera_fde import MUVERAEncoder

# One encoder instance for both queries and documents — seed must match
enc = MUVERAEncoder(
    dimension=128,              # ColBERT / ColQwen2 token embedding dimension
    num_simhash_projections=4,  # 2^4 = 16 partitions per repetition
    num_repetitions=2,          # 2 independent repetitions
    seed=42,
)

print(enc)
# MUVERAEncoder(dimension=128, num_simhash_projections=4, num_repetitions=2,
#               projection_type=DEFAULT_IDENTITY, fde_dimension=4096)

query_tokens = np.random.randn(32,  128).astype(np.float32)   # 32 query tokens
doc_tokens   = np.random.randn(512, 128).astype(np.float32)   # 512 document tokens

q_fde = enc.encode_query(query_tokens)    # shape: (4096,)
d_fde = enc.encode_document(doc_tokens)   # shape: (4096,)

# Approximate Chamfer Similarity — drop into any ANN index as a float32 vector
score = float(q_fde @ d_fde)
```

---

## API reference

### `MUVERAEncoder`

The primary entry point. Initialise **once** and reuse for all queries and
documents — the random partition structure (SimHash matrices, Count Sketch
parameters) must be identical on both sides.

```python
MUVERAEncoder(
    dimension: int = 128,
    num_simhash_projections: int = 4,
    num_repetitions: int = 1,
    seed: int = 1,
    projection_type: ProjectionType = ProjectionType.DEFAULT_IDENTITY,
    projection_dimension: int | None = None,
    fill_empty_partitions: bool = False,
    final_projection_dimension: int | None = None,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dimension` | 128 | Token embedding dimension |
| `num_simhash_projections` | 4 | SimHash bits *k*; partitions = 2^k |
| `num_repetitions` | 1 | Independent repetitions (more → better approximation) |
| `seed` | 1 | Shared RNG seed — **must match** query and document sides |
| `projection_type` | `DEFAULT_IDENTITY` | `DEFAULT_IDENTITY` or `AMS_SKETCH` (Count Sketch) |
| `projection_dimension` | `None` | Target dim after Count Sketch; required for `AMS_SKETCH` |
| `fill_empty_partitions` | `False` | Document side: fill empty slots via Hamming-nearest-neighbour |
| `final_projection_dimension` | `None` | Post-accumulation Count Sketch compression |

**Property:** `fde_dimension` — output vector length.

---

### Encoding single inputs

```python
enc = MUVERAEncoder(dimension=128, num_simhash_projections=4, num_repetitions=2)

# Query: SUM aggregation — token embeddings summed into their SimHash partition
q_fde = enc.encode_query(query_tokens)    # (num_tokens, 128) → (fde_dim,)

# Document: AVERAGE aggregation — centroid of tokens per partition
d_fde = enc.encode_document(doc_tokens)   # (num_tokens, 128) → (fde_dim,)

# Both also accept flat 1-D input (num_tokens * dimension,)
q_fde = enc.encode_query(query_tokens.flatten())
```

---

### Batch encoding

```python
queries   = [np.random.randn(32,  128).astype(np.float32) for _ in range(100)]
documents = [np.random.randn(512, 128).astype(np.float32) for _ in range(1000)]

Q = enc.encode_queries_batch(queries)     # shape: (100,  fde_dimension)
D = enc.encode_documents_batch(documents) # shape: (1000, fde_dimension)

# All-pairs approximate Chamfer Similarities in one matmul
scores = Q @ D.T   # shape: (100, 1000)
top_k  = np.argsort(scores, axis=1)[:, ::-1][:, :10]  # top-10 per query
```

---

### Reducing FDE size

Two orthogonal compression knobs:

**Option A — per-partition Count Sketch** (reduces width before accumulation):

```python
from muvera_fde import ProjectionType

enc = MUVERAEncoder(
    dimension=128,
    num_simhash_projections=4,
    num_repetitions=4,
    projection_type=ProjectionType.AMS_SKETCH,
    projection_dimension=32,   # 128 → 32 per partition slot
)
# fde_dimension = 4 reps × 16 partitions × 32 = 2048  (vs 8192 without)
```

**Option B — post-accumulation Count Sketch** (compresses the final vector):

```python
enc = MUVERAEncoder(
    dimension=128,
    num_simhash_projections=4,
    num_repetitions=4,
    final_projection_dimension=512,   # 8192 → 512
)
# fde_dimension = 512
```

Both preserve dot products in expectation: `E[⟨sketch(x), sketch(y)⟩] = ⟨x, y⟩`.

---

### Filling empty partition slots

With few document tokens and many partitions (large *k*), many slots will be
empty (all-zero). Enabling `fill_empty_partitions` copies the projection of
the nearest token by SimHash Hamming distance into each empty slot, improving
recall for short documents:

```python
enc = MUVERAEncoder(
    dimension=128,
    num_simhash_projections=4,
    num_repetitions=2,
    fill_empty_partitions=True,   # document side only; queries ignore this flag
)

short_doc_tokens = np.random.randn(8, 128).astype(np.float32)
d_fde = enc.encode_document(short_doc_tokens)   # no all-zero partition blocks
```

---

### Low-level functional API

Bypass the encoder class entirely when you need to manage parameters manually
(e.g. distributed indexing where workers share pre-built parameters):

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

# Pass pre-built RepParams to skip RNG sampling on every call
enc = MUVERAEncoder(dimension=128, num_repetitions=2, num_simhash_projections=4, seed=42)
q_fde = generate_query_fde(query_tokens, config, enc._rep_params)
```

---

### `FDEConfig` serialisation

`FDEConfig` is a frozen Pydantic model — save it alongside your ANN index so
the encoder configuration is always recoverable:

```python
import json
from muvera_fde import FDEConfig

config = FDEConfig(dimension=128, num_repetitions=4, num_simhash_projections=4, seed=42)

# Save
with open("fde_config.json", "w") as f:
    json.dump(config.model_dump(), f)

# Load
with open("fde_config.json") as f:
    config2 = FDEConfig(**json.load(f))

assert config == config2
```

---

## Two-stage retrieval pipeline

The intended production pattern for ColQwen2 / ColBERT:

```
Offline:
  doc token embeddings  →  encode_document()  →  FDE vector  →  ANN index

Online:
  query token embeddings  →  encode_query()  →  FDE vector
                                                     │
                                              ANN search (fast, sub-linear)
                                                     │
                                            top-K candidate docs
                                                     │
                                       MaxSim re-rank on raw token embeddings
                                                     │
                                               final top-K results
```

Stage 1 (ANN on FDE vectors) eliminates 99%+ of the corpus cheaply.
Stage 2 (exact MaxSim on raw token embeddings) reranks the small candidate
set for full accuracy.

### Minimal FAISS integration

```python
import faiss
import numpy as np
from muvera_fde import MUVERAEncoder

enc = MUVERAEncoder(dimension=128, num_simhash_projections=4, num_repetitions=2, seed=42)
dim = enc.fde_dimension  # 4096

# Build index
index = faiss.IndexFlatIP(dim)   # inner product ≈ Chamfer Similarity

# Index documents (offline)
doc_embeddings = [...]   # list of (num_tokens, 128) float32 arrays
D = enc.encode_documents_batch(doc_embeddings)   # (N, 4096)
faiss.normalize_L2(D)
index.add(D)

# Query (online)
query_tokens = np.random.randn(32, 128).astype(np.float32)
q_fde = enc.encode_query(query_tokens).reshape(1, -1)
faiss.normalize_L2(q_fde)

_, candidate_ids = index.search(q_fde, k=100)   # stage 1: fast ANN
# stage 2: MaxSim re-rank candidate_ids with raw token embeddings ...
```

---

## Attribution

Python port of the C++ implementation in
[Google's graph-mining project](https://github.com/google/graph-mining/tree/main/sketching/point_cloud),
licensed under Apache 2.0.

See [NOTICE](NOTICE) for the full upstream attribution.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
