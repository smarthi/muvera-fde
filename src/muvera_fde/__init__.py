"""
muvera-fde: Fixed Dimensional Encodings for Multi-Vector Retrieval.

Python port of Google's graph-mining MUVERA implementation:
  https://github.com/google/graph-mining/tree/main/sketching/point_cloud

Paper: "MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings"
       https://arxiv.org/abs/2405.19504

Quick start
-----------
>>> import numpy as np
>>> from muvera_fde import MUVERAEncoder
>>>
>>> enc = MUVERAEncoder(dimension=128, num_simhash_projections=4, num_repetitions=2)
>>> q_fde = enc.encode_query(np.random.randn(32, 128).astype("float32"))
>>> d_fde = enc.encode_document(np.random.randn(512, 128).astype("float32"))
>>> score = float(q_fde @ d_fde)   # approximate Chamfer Similarity
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("muvera-fde")
except PackageNotFoundError:  # editable / source installs before build
    __version__ = "0.0.0.dev0"

from muvera_fde.config import FDEConfig, ProjectionType
from muvera_fde.core import generate_document_fde, generate_query_fde
from muvera_fde.encoder import MUVERAEncoder

__all__ = [
    # High-level
    "MUVERAEncoder",
    # Config
    "FDEConfig",
    "ProjectionType",
    # Low-level functional API
    "generate_query_fde",
    "generate_document_fde",
    # Version
    "__version__",
]
