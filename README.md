# pymuvera — MUVERA + EGGROLL: Fixed Dimensional Encodings for Multi-Vector Retrieval

**Sub-linear ANN retrieval for ColBERT, ColPali, and ColQwen2.**

[![PyPI](https://img.shields.io/pypi/v/pymuvera)](https://pypi.org/project/pymuvera/)
[![Python](https://img.shields.io/pypi/pyversions/pymuvera)](https://pypi.org/project/pymuvera/)
[![CI](https://github.com/smarthi/muvera-fde/actions/workflows/ci.yml/badge.svg)](https://github.com/smarthi/muvera-fde/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A pure-Python port of Google's graph-mining MUVERA implementation, extended with
**low-rank SimHash factorisation** inspired by the EGGROLL paper (Sarkar et al., 2025).

| | Reference |
|---|---|
| MUVERA paper | [Rajput et al., 2024](https://arxiv.org/abs/2405.19504) |
| EGGROLL paper | [Sarkar et al., 2025](https://eshyperscale.github.io/imgs/paper.pdf) |
| Original C++ implementation | [google/graph-mining](https://github.com/google/graph-mining/tree/main/sketching/point_cloud) |

---

## What this library adds beyond the original paper

The MUVERA paper uses a full-rank Gaussian matrix for SimHash partitioning. This
library adds two new SimHash projection modes, each with distinct cost/quality tradeoffs:

**`LOW_RANK_GAUSSIAN`** factors the SimHash matrix as AB⊤ (where `A ∈ ℝ^{d×r}`,
`B ∈ ℝ^{k×r}`, `r ≪ k`), cutting partition compute from `O(N·d·k)` to
`O(N·d·r + N·r·k)`. The theoretical backing is EGGROLL (Sarkar et al., 2025,
Theorem 4): O(r⁻¹) convergence to the full-rank Gaussian sign pattern. At `r=4`
with ColQwen2 (d=128, k=8): **~1.9× faster**, ~25% variance increase.

**`SRHT`** (Subsampled Randomized Hadamard Transform, Ailon & Chazelle 2009) applies
a structured `S·H·D` transform — random sign flip, Walsh-Hadamard, random row
subsample — at `O(N·d·log d)` cost, independent of k. It carries a **full JL
guarantee** with zero rank-approximation error, making it the theoretically safest
choice. For ColQwen2 (d=128, k=8): **904N ops vs 1024N** for full-rank.

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
pip install pymuvera
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
    simhash_rank: int = 1,
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
| `projection_type` | `DEFAULT_IDENTITY` | `DEFAULT_IDENTITY`, `AMS_SKETCH` (Count Sketch on token embeddings), `LOW_RANK_GAUSSIAN` (low-rank factored SimHash, EGGROLL), or `SRHT` (Subsampled Randomized Hadamard Transform) |
| `projection_dimension` | `None` | Target dim after Count Sketch; required for `AMS_SKETCH` |
| `simhash_rank` | 1 | Rank *r* for `LOW_RANK_GAUSSIAN`; must satisfy `1 ≤ r < num_simhash_projections`. r=4 is a practical sweet spot for ColQwen2 (d=128, k≥8) |
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

### SimHash projection modes

Three SimHash projection modes are available, each trading speed against quality.
All produce the **same FDE output shape** and are **drop-in replacements** for
each other — only the SimHash matrix computation changes.

#### Mode 1: `DEFAULT_IDENTITY` — full-rank Gaussian (baseline)

Samples a fresh `(d × k)` Gaussian matrix per repetition. JL guarantee,
full-rank quality. Baseline for comparison.

```python
enc = MUVERAEncoder(
    dimension=128,
    num_simhash_projections=8,
    num_repetitions=4,
)
# SimHash cost: O(N × 128 × 8) = 1024N ops/rep
```

---

#### Mode 2: `LOW_RANK_GAUSSIAN` — low-rank factored SimHash (EGGROLL)

Factors `W ≈ AB⊤` where `A ∈ ℝ^{d×r}`, `B ∈ ℝ^{k×r}`, replacing one large
matmul with two smaller ones:

```python
from muvera_fde import ProjectionType

enc = MUVERAEncoder(
    dimension=128,
    num_simhash_projections=8,
    num_repetitions=4,
    projection_type=ProjectionType.LOW_RANK_GAUSSIAN,
    simhash_rank=4,   # r=4: O(N×128×4 + N×4×8) = 544N ops — 1.9× faster
    seed=42,
)
```

**Convergence** (EGGROLL, Sarkar et al. 2025, Theorem 4): O(r⁻¹) convergence
to full-rank Gaussian — faster than the CLT rate O(r⁻¹/²) because symmetry
cancels all odd cumulants in the Edgeworth expansion.

| `simhash_rank` | Variance vs full-rank | Cost (k=8) | Speedup |
|---|---|---|---|
| 1 | ~100% baseline | 136N ops | 7.5× |
| 4 | ~25% increase | 544N ops | 1.9× |
| 8 | ~12% increase | 1088N ops | ~breakeven |

> The 1/√r normalisation is omitted — SimHash sign assignments are
> scale-invariant (`sign(αx) = sign(x)`), so it has no effect.

---

#### Mode 3: `SRHT` — Subsampled Randomized Hadamard Transform

Applies the structured transform `S·H·D` row-wise:

* **D** — random diagonal ±1 (Rademacher sign flip)
* **H** — Walsh-Hadamard transform (O(d log d) butterfly)
* **S** — random row subsampling to k dimensions

Input is zero-padded to the next power of 2 ≥ d before applying H.

```python
enc = MUVERAEncoder(
    dimension=128,
    num_simhash_projections=8,
    num_repetitions=4,
    projection_type=ProjectionType.SRHT,
    seed=42,
)
# SimHash cost: O(N × 128 × log₂(128) + N × 8) = O(N × 128 × 7 + N × 8) = 904N ops
# No rank approximation error — full JL guarantee (Ailon & Chazelle, 2009)
# Constraint: num_simhash_projections <= next_power_of_2(dimension)
```

**Theoretical guarantee**: SRHT is a full Johnson-Lindenstrauss projection —
it preserves pairwise distances to ε with high probability, with no rank
approximation error. Unlike LOW_RANK_GAUSSIAN, it converges exactly to
full-rank Gaussian quality at `k = d`.

---

#### Three-way comparison for ColQwen2 (d=128)

| Mode | SimHash cost (k=8) | vs baseline | Quality | Extra constraint |
|---|---|---|---|---|
| `DEFAULT_IDENTITY` | 1024N ops | 1× | Full-rank Gaussian baseline | None |
| `LOW_RANK_GAUSSIAN` r=4 | 544N ops | **1.9×** | O(r⁻¹) convergence, ~25% variance ↑ | `1 ≤ r < k` |
| `LOW_RANK_GAUSSIAN` r=1 | 136N ops | **7.5×** | ~100% variance baseline | `1 ≤ r < k` |
| `SRHT` | 904N ops | 1.1× | Full JL, no rank error | `k ≤ next_pow2(d)` |

**When to use each:**

* **`DEFAULT_IDENTITY`** — default choice; correctness baseline, no constraints.
* **`LOW_RANK_GAUSSIAN`** — when speed is the priority and mild quality loss is acceptable.
  Use r=4 for ColQwen2. Becomes more attractive as k grows (cost scales as O(r) not O(k)).
* **`SRHT`** — when you need full JL quality at sub-quadratic cost, or when k is large
  (SRHT cost is O(d log d) regardless of k). Preferred for precision-critical workloads
  like legal/tax document retrieval at WK where recall matters.

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

### `FDEConfig` serialization

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

Low-rank SimHash extension inspired by
[EGGROLL: Evolution Strategies at the Hyperscale](https://eshyperscale.github.io/imgs/paper.pdf)
(Sarkar et al., 2025).

See [NOTICE](NOTICE) for the full upstream attribution.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
