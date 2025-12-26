"""Types used in the standard featurization module."""
from typing import Literal, NewType

import torch
from jaxtyping import Float

# Scalar node feature types
ScalarNodeFeature = Literal[
    "amino_acid_one_hot",
    "alpha",
    "kappa",
    "dihedrals",
    "sidechain_torsions",
    "sequence_positional_encoding",
]

# Vector node feature types
VectorNodeFeature = Literal["orientation", "virtual_cb_vector"]

# Scalar edge feature types
ScalarEdgeFeature = Literal["edge_distance", "sequence_distance"]

# Vector edge feature types
VectorEdgeFeature = Literal["edge_vectors", "pos_emb"]

# Tensor types
OrientationTensor = NewType(
    "OrientationTensor", Float[torch.Tensor, "n_nodes 2 3"]
)

