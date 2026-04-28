# pymuvera — MUVERA + EGGROLL: Fixed Dimensional Encodings for Multi-Vector Retrieval

**Sub-linear ANN retrieval for ColBERT, ColPali, and ColQwen2.**

[![PyPI](https://img.shields.io/pypi/v/pymuvera)](https://pypi.org/project/pymuvera/)
[![Python](https://img.shields.io/pypi/pyversions/pymuvera)](https://pypi.org/project/pymuvera/)
[![CI](https://github.com/smarthi/muvera-fde/actions/workflows/ci.yml/badge.svg)](https://github.com/smarthi/muvera-fde/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A pure-Python port of Google's graph-mining MUVERA implementation, extended with
**low-rank SimHash factorisation** (EGGROLL, Sarkar et al., 2025) and
**Subsampled Randomized Hadamard Transform** (SRHT, Woolfe, Liberty, Rokhlin & Tygert, 2008) SimHash modes.

|                                       | Reference                                                                                     |
|---------------------------------------|-----------------------------------------------------------------------------------------------|
| MUVERA paper                          | [Dhulipala et al., 2024](https://arxiv.org/abs/2405.19504)                                    |
| EGGROLL paper                         | [Sarkar et al., 2025](https://eshyperscale.github.io/imgs/paper.pdf)                          |
| Johnson-Lindenstrauss Transform paper | [Ailon et al., 2006](https://www.cs.princeton.edu/~chazelle/pubs/FJLT-sicomp09.pdf)           |
| Original C++ implementation           | [google/graph-mining](https://github.com/google/graph-mining/tree/main/sketching/point_cloud) |

---

## What this library adds beyond the original paper

The MUVERA paper uses a full-rank Gaussian matrix for SimHash partitioning. This
library adds two new SimHash projection modes, each with distinct cost/quality tradeoffs:

**`LOW_RANK_GAUSSIAN`** factors the SimHash matrix as AB⊤ (where `A ∈ ℝ^{d×r}`,
`B ∈ ℝ^{k×r}`, `r ≪ k`), cutting partition compute from `O(N·d·k)` to
`O(N·d·r + N·r·k)`. The theoretical backing is EGGROLL (Sarkar et al., 2025,
Theorem 4): O(r⁻¹) convergence to the full-rank Gaussian sign pattern. At `r=4`
with ColQwen2 (d=128, k=8): **~1.9× faster**, ~25% variance increase.

**`SRHT`** (Subsampled Randomized Hadamard Transform, Woolfe, Liberty, Rokhlin & Tygert, 2008) applies
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

The primary entry point. Initialize **once** and reuse for all queries and
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

**Convergence** (EGGROLL, Sarkar et al. 2025, Theorem 4): the low-rank sign
pattern converges to the full-rank Gaussian at **O(r⁻¹)** — faster than the
**CLT rate of O(r⁻¹/²)**.

**What is the CLT rate?** The Central Limit Theorem tells us that averaging *n*
independent random variables reduces error at O(n⁻¹/²) — the square root of the
sample size. This is the default convergence rate for most random approximations.
EGGROLL beats it because the low-rank matrix AB⊤ has a *symmetric* distribution:
the sign of each projection is equally likely to be ±1, which causes all **odd
cumulants** (1st, 3rd, 5th order terms) in the Edgeworth expansion to cancel
exactly. Since those odd terms are what normally contribute O(r⁻¹/²) error,
their cancellation pushes the leading error down to O(r⁻¹) — the same mechanism
that makes symmetric random walks converge faster than asymmetric ones.

| `simhash_rank` r | CLT rate O(r⁻¹/²) | EGGROLL rate O(r⁻¹) | Speedup vs baseline |
|---|---|---|---|
| 4 | ~50% error | **~25% error** | 1.9× |
| 9 | ~33% error | **~11% error** | — |
| 16 | ~25% error | **~6% error** | — |

Cost breakdown for ColQwen2 (d=128, k=8):

| `simhash_rank` | SimHash cost | Speedup |
|---|---|---|
| 1 | 136N ops | 7.5× |
| 4 | 544N ops | 1.9× |
| 8 | 1088N ops | ~breakeven |

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
# No rank approximation error — full JL guarantee (Woolfe, Liberty, Rokhlin & Tygert, 2008)
# Constraint: num_simhash_projections <= next_power_of_2(dimension)
```

**Theoretical guarantee**: SRHT is a full Johnson-Lindenstrauss projection —
it preserves pairwise distances to ε with high probability, with no rank
approximation error. Unlike LOW_RANK_GAUSSIAN, it converges exactly to
full-rank Gaussian quality at `k = d`.
Tropp (2011) provides the tightest known analysis, proving that
`ℓ ≥ (1+ι) · k log(k)` subsampled dimensions suffice to preserve an entire
k-dimensional subspace with optimal constants via matrix Chernoff inequalities.
For SimHash (sign-only) use, this subspace result is sufficient but not tight —
sign assignments are scale-invariant so the embedding constants do not apply directly.

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
  like legal/tax document retrieval where recall matters.

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

---

## Configuration guide

Most users hit poor results not because of a wrong projection type but because of a
misconfigured `num_simhash_projections` / `num_repetitions` / `simhash_rank` combination.
This section explains every tradeoff in plain terms, with concrete numbers for ColQwen2
(128-dim) and ColQwen3.5 (320-dim) — the two most common production models.

---

### Know your embedding dimension first

Different models produce different per-token embedding dimensions. Set `dimension` to
match your model exactly — this is the single most important parameter.

| Model | `dimension` | Notes |
|---|---|---|
| ColBERT v2 | 128 | Original late-interaction baseline |
| ColQwen2 | 128 | Most widely deployed as of 2025 |
| ColQwen3.5 v1 | 128 | Early checkpoint |
| ColQwen3.5 v3 | 320 | Current recommended checkpoint |
| Ops-ColQwen3-4B | 320 | OpenSearch variant, up to 2560 via extended head |

> **Common mistake:** Using `dimension=128` with ColQwen3.5 v3 (which is 320-dim) silently
> truncates every token embedding to 128 dims, discarding 60% of the representation before
> MUVERA even runs. Always verify with `model.config.projection_dim` or check the model card.

---

### The two knobs that matter most

#### `num_simhash_projections` (k) — partition granularity

Each repetition divides embedding space into **2^k buckets**. Tokens that land in the
same bucket get averaged together into one FDE slot.

| k | Partitions | Tokens/partition (512-token doc) | Recommendation |
|---|---|---|---|
| 4 | 16 | 32 | coarse; fast but high collision rate |
| 6 | 64 | 8 | reasonable default |
| 8 | 256 | 2 | good quality; use `fill_empty_partitions=True` |
| 10 | 1,024 | 0.5 | too sparse for most docs; many empty slots |

> **Rule of thumb:** aim for **4–10 tokens per partition** on average.
> For a 512-token ColQwen3.5 page: k=6 (8 tokens/partition) or k=8 with fill enabled.

#### `num_repetitions` — approximation quality

Each repetition is an independent random partition of the same embedding space. More
repetitions directly improves recall and is the safest quality knob to increase.

- More repetitions **always** improves recall.
- Cost scales linearly: 2× repetitions = 2× FDE size = 2× encode time.
- Diminishing returns set in around 8–16 repetitions for most corpora.

> **Rule of thumb:** start with `num_repetitions=8`. If recall is poor, double it before
> touching any other parameter.

---

### The budget equation

```
fde_dimension = num_repetitions × 2^k × dimension
```

For a fixed FDE budget, spending it on **more repetitions beats larger k** for most corpora:

| Config | fde_dimension (ColQwen3.5, d=320) | Notes |
|---|---|---|
| k=6, reps=20 | 20 × 64 × 320 = 409,600 | many repetitions, coarse partitions |
| k=8, reps=10 | 10 × 256 × 320 = 819,200 | balanced — usually better recall |
| k=8, reps=5 | 5 × 256 × 320 = 409,600 | same budget as first row; better quality |

Use `final_projection_dimension` to compress to a target index size after choosing
the right k/repetitions balance:

```python
enc = MUVERAEncoder(
    dimension=320,               # ColQwen3.5 v3
    num_simhash_projections=8,
    num_repetitions=10,
    fill_empty_partitions=True,
    final_projection_dimension=81920,  # compress to target index size
)
```

---

### When to use `fill_empty_partitions`

With k=8 (256 partitions) and a short document (< 200 tokens), many partition slots
will be empty — all zeros in the FDE. Zeros contribute nothing to the dot product and
directly hurt recall.

Enable `fill_empty_partitions=True` whenever:

```
num_doc_tokens / 2^k < 2
```

| k | Enable fill if doc tokens < |
|---|---|
| 6 | 128 |
| 8 | 512 |
| 10 | 2,048 |

For ColQwen3.5 pages at k=8: nearly always enable fill, since most document pages
produce fewer than 512 tokens.

---

### `LOW_RANK_GAUSSIAN` — when it helps and when it does not

Low-rank SimHash only makes theoretical sense when **r is much smaller than k**.
The computational benefit comes from the ratio r/k — if that ratio is close to 1,
you get all the approximation error with almost no speed gain.

| k | r | r/k ratio | Assessment |
|---|---|---|---|
| 6 | 4 | 0.67 | ❌ nearly full-rank — avoid |
| 8 | 4 | 0.50 | ⚠️ marginal benefit |
| 16 | 4 | 0.25 | ✅ good tradeoff (~1.9× faster, ~25% variance ↑) |
| 16 | 2 | 0.13 | ✅ aggressive (~4× faster, ~50% variance ↑) |

> **The k=6, rank=4 trap:** this is a near-full-rank approximation of a 6-bit matrix.
> You pay ~25% variance penalty with only a 1.4× compute saving. This combination
> produces the worst results of all modes (as seen in early ColQwen3.5 benchmarks).
> **Minimum recommended config for LOW_RANK_GAUSSIAN: k ≥ 16, rank ≤ k//4.**

---

### Recommended starting configs

#### ColQwen2 (d=128) — general purpose

```python
enc = MUVERAEncoder(
    dimension=128,
    num_simhash_projections=8,
    num_repetitions=8,
    fill_empty_partitions=True,
    seed=42,
)
# fde_dimension = 8 × 256 × 128 = 262,144
# tokens/partition at 512 tokens: 2 — fill is essential
```

#### ColQwen3.5 v3 (d=320) — general purpose

```python
enc = MUVERAEncoder(
    dimension=320,
    num_simhash_projections=8,
    num_repetitions=8,
    fill_empty_partitions=True,
    seed=42,
)
# fde_dimension = 8 × 256 × 320 = 655,360
# use final_projection_dimension if index size is a constraint
```

#### ColQwen3.5 v3 — speed-optimized (SRHT)

```python
enc = MUVERAEncoder(
    dimension=320,
    num_simhash_projections=8,
    num_repetitions=8,
    projection_type=ProjectionType.SRHT,
    fill_empty_partitions=True,
    seed=42,
)
# Full JL guarantee, ~12% faster SimHash than DEFAULT_IDENTITY at k=8
# Best quality/speed tradeoff in benchmarks
```

#### ColQwen3.5 v3 — low-rank (correctly configured)

```python
enc = MUVERAEncoder(
    dimension=320,
    num_simhash_projections=16,   # k must be large for low-rank to help
    num_repetitions=4,
    projection_type=ProjectionType.LOW_RANK_GAUSSIAN,
    simhash_rank=4,               # r/k = 4/16 = 0.25 — meaningful low-rank
    fill_empty_partitions=True,
    seed=42,
)
# fde_dimension = 4 × 65536 × 320 = 83,886,080 — use final_projection_dimension
```

---

### Quality vs. exact MaxSim — setting realistic expectations

MUVERA FDE retrieval is a **first-stage filter**, not a replacement for exact MaxSim.
Typical recall gaps on a 512-token ColQwen3.5 corpus:

| Stage | R@1 (typical) | Retrieval time |
|---|---|---|
| Exact MaxSim (multi-vector) | ~0.88 | slow, scales with corpus size |
| MUVERA FDE + ANN (first stage) | ~0.63 | fast, sub-linear |
| MUVERA FDE → MaxSim rerank top-100 | ~0.86 | fast + small rerank overhead |

The ~25 point R@1 gap between exact and FDE-only is normal and expected. Always pair
pymuvera with a MaxSim reranking step on the ANN shortlist for production use.

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