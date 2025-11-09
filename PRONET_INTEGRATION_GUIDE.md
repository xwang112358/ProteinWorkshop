# ProNet Integration Guide (Backbone Level)

## Executive Summary

**Goal:** Integrate ProNet backbone encoder into ProteinWorkshop, prioritizing existing framework features over custom implementations.

**Scope:** Backbone representation only (requires CA, N, C atom coordinates)

**Key Integration Principles:**
1. ✅ **Reuse ProteinWorkshop features:** `amino_acid_one_hot`, `dihedrals`, `sidechain_torsions`
2. ✅ **Port ProNet-specific components:** Spherical harmonics, Bessel functions, Euler angle computation
3. ✅ **Minimal framework changes:** Extend `ProteinFeaturiser` to add backbone coordinates
4. ✅ **Standard interfaces:** Return `EncoderOutput`, use `Batch`/`ProteinBatch`

**Main Files to Create:**
- `proteinworkshop/models/graph_encoders/layers/pronet_features.py` - Geometric encodings
- `proteinworkshop/models/graph_encoders/layers/pronet.py` - Interaction layers
- `proteinworkshop/models/graph_encoders/pronet.py` - Main encoder
- `proteinworkshop/config/encoder/pronet.yaml` - Model config
- `proteinworkshop/config/features/pronet_backbone.yaml` - Feature config

---

## Overview

ProNet is an invariant graph neural network that computes geometric features (angles, distances, orientations) from 3D coordinates. This guide focuses on integrating the **backbone representation level** into ProteinWorkshop.

**Key Characteristics:**
- Uses spherical harmonics and Bessel functions for geometric encoding
- Computes theta/phi angles and Euler angles from CA, N, C atom coordinates
- Uses ProteinWorkshop's pre-computed edges (e.g., `knn_30`)
- Invariant to rotation and translation

**Integration Priority:**
- Leverage ProteinWorkshop's existing feature system wherever possible
- Use pre-computed edges from feature config instead of dynamic radius graph
- Only implement ProNet-specific geometric encodings (spherical harmonics)
- Use existing `dihedrals` features instead of custom embeddings

---

## Integration Steps

### 1. Port ProNet-Specific Feature Encodings

#### 1.1 Create spherical harmonics feature modules
**Location:** `proteinworkshop/models/graph_encoders/layers/pronet_features.py`

**Purpose:** Only port the geometric encoding modules that are unique to ProNet (Bessel functions × Spherical harmonics). Use ProteinWorkshop's existing node features.

```python
import torch
import torch.nn as nn
import sympy as sym
import numpy as np
from scipy.optimize import brentq
from scipy import special as sp

# Port utility functions from features.py
def Jn(r, n):
    """Numerical spherical Bessel functions of order n"""
    return sp.spherical_jn(n, r)

def Jn_zeros(n, k):
    """Compute first k zeros of spherical Bessel functions up to order n"""
    # Port implementation from features.py lines 15-30
    pass

def spherical_bessel_formulas(n):
    """Compute sympy formulas for spherical Bessel functions"""
    # Port implementation from features.py lines 33-48
    pass

def bessel_basis(n, k):
    """Compute normalized and rescaled spherical Bessel basis functions"""
    # Port implementation from features.py lines 51-82
    pass

def real_sph_harm(L, spherical_coordinates=True, zero_m_only=False):
    """Compute real spherical harmonics up to degree L"""
    # Port implementation from features.py lines 150-241
    pass

class AngleEmbedding(nn.Module):
    """Embed (distance, angle) pairs using Bessel × Spherical harmonics
    
    Used for encoding Euler angles in backbone representation.
    """
    def __init__(self, num_radial: int, num_spherical: int, cutoff: float = 10.0):
        super().__init__()
        # Port d_angle_emb implementation from features.py lines 244-278
        # Pre-compute Bessel and spherical harmonic basis functions
        # Store as lambdified functions for efficient evaluation
        pass
    
    def forward(self, dist: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dist: Edge distances (num_edges,)
            angle: Angles in radians (num_edges,)
        Returns:
            Embedded features (num_edges, num_spherical * num_radial)
        """
        pass

class ThetaPhiEmbedding(nn.Module):
    """Embed (distance, theta, phi) triplets using Bessel × Spherical harmonics
    
    Used for encoding angular geometry in complete 3D graph.
    """
    def __init__(self, num_radial: int, num_spherical: int, cutoff: float = 10.0):
        super().__init__()
        # Port d_theta_phi_emb implementation from features.py lines 281-341
        pass
    
    def forward(self, dist: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dist: Edge distances (num_edges,)
            theta: Polar angles (num_edges,)
            phi: Azimuthal angles (num_edges,)
        Returns:
            Embedded features (num_edges, num_spherical^2 * num_radial)
        """
        pass
```

#### 1.2 Create ProNet interaction layers
**Location:** `proteinworkshop/models/graph_encoders/layers/pronet.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, inits
from torch_scatter import scatter
from torch_sparse import matmul

class Linear(nn.Module):
    """Linear layer with configurable weight initialization (glorot or zeros)"""
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True, 
                 weight_initializer: str = 'glorot'):
        # Port from pronet.py lines 28-68
        pass

class TwoLinear(nn.Module):
    """Two stacked linear layers with optional activation"""
    def __init__(self, in_channels: int, middle_channels: int, out_channels: int,
                 bias: bool = False, act: bool = False):
        # Port from pronet.py lines 71-105
        pass

class EdgeGraphConv(MessagePassing):
    """Graph convolution with Hadamard product between node and edge features
    
    Key difference from standard GraphConv: edge features modulate messages via element-wise multiplication.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin_l = Linear(in_channels, out_channels)
        self.lin_r = Linear(in_channels, out_channels, bias=False)
        # Port from pronet.py lines 108-145
    
    def forward(self, x, edge_index, edge_weight):
        """
        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Edge indices (2, num_edges)
            edge_weight: Edge features (num_edges, in_channels)
        Returns:
            Updated node features (num_nodes, out_channels)
        """
        pass

class InteractionBlock(nn.Module):
    """ProNet interaction block with three parallel message passing streams
    
    Stream 0: (dist, theta, phi) geometric embeddings
    Stream 1: Euler angle embeddings (3 angles for backbone)
    Stream 2: Sequence positional embeddings
    """
    def __init__(
        self,
        hidden_channels: int,
        output_channels: int,
        num_radial: int,
        num_spherical: int,
        num_layers: int,
        mid_emb: int,
        act=F.silu,
        num_pos_emb: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.act = act
        self.dropout = nn.Dropout(dropout)
        
        # Three parallel EdgeGraphConv layers
        self.conv0 = EdgeGraphConv(hidden_channels, hidden_channels)
        self.conv1 = EdgeGraphConv(hidden_channels, hidden_channels)
        self.conv2 = EdgeGraphConv(hidden_channels, hidden_channels)
        
        # Feature projection layers
        self.lin_feature0 = TwoLinear(num_radial * num_spherical ** 2, mid_emb, hidden_channels)
        self.lin_feature1 = TwoLinear(3 * num_radial * num_spherical, mid_emb, hidden_channels)  # 3 Euler angles
        self.lin_feature2 = TwoLinear(num_pos_emb, mid_emb, hidden_channels)
        
        # Node feature processing
        self.lin_1 = Linear(hidden_channels, hidden_channels)
        self.lin_2 = Linear(hidden_channels, hidden_channels)
        
        # Output processing per stream
        self.lin0 = Linear(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        
        # Concatenation and refinement layers
        self.lins_cat = nn.ModuleList([Linear(3 * hidden_channels, hidden_channels)])
        self.lins_cat.extend([Linear(hidden_channels, hidden_channels) for _ in range(num_layers - 1)])
        
        self.lins = nn.ModuleList([Linear(hidden_channels, hidden_channels) for _ in range(num_layers - 1)])
        self.final = Linear(hidden_channels, output_channels)
        # Port from pronet.py lines 148-243
    
    def forward(self, x, feature0, feature1, pos_emb, edge_index, batch):
        """
        Args:
            x: Node features (num_nodes, hidden_channels)
            feature0: (dist, theta, phi) embeddings (num_edges, num_radial * num_spherical^2)
            feature1: Euler angle embeddings (num_edges, 3 * num_radial * num_spherical)
            pos_emb: Positional embeddings (num_edges, num_pos_emb)
            edge_index: Edge indices (2, num_edges)
            batch: Batch indices (num_nodes,)
        Returns:
            Updated node features (num_nodes, hidden_channels)
        """
        # Process node features
        # Apply three parallel convolutions with different edge features
        # Concatenate outputs and refine
        # Residual connection
        pass
```

---

### 2. Create Main ProNet Encoder (Backbone Level)

**Location:** `proteinworkshop/models/graph_encoders/pronet.py`

```python
from typing import Set, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import radius_graph
from graphein.protein.tensor.data import ProteinBatch

from proteinworkshop.custom_types import EncoderOutput
from proteinworkshop.models.utils import get_aggregation
from proteinworkshop.models.graph_encoders.layers.pronet import InteractionBlock, Linear
from proteinworkshop.models.graph_encoders.layers.pronet_features import (
    AngleEmbedding, ThetaPhiEmbedding
)

class ProNetBackbone(nn.Module):
    """ProNet encoder for backbone-level protein representations
    
    Uses CA, N, C atom coordinates to compute:
    - Geometric features (theta, phi angles)
    - Euler angles between local frames
    - Spherical harmonic encodings
    
    Leverages ProteinWorkshop's existing node features (amino acid type, dihedrals).
    """
    def __init__(
        self,
        num_blocks: int = 4,
        hidden_channels: int = 128,
        mid_emb: int = 64,
        num_radial: int = 6,
        num_spherical: int = 2,
        int_emb_layers: int = 3,
        num_pos_emb: int = 16,
        dropout: float = 0.0,
        pool: str = "sum",
    ):
        super().__init__()
        self.num_pos_emb = num_pos_emb
        self.hidden_channels = hidden_channels
        
        # Geometric feature encoders (ProNet-specific)
        self.theta_phi_emb = ThetaPhiEmbedding(num_radial, num_spherical, cutoff)
        self.angle_emb = AngleEmbedding(num_radial, num_spherical, cutoff)
        
        # Node feature embedding
        # Input: amino_acid_one_hot (20) + dihedrals (6) + sidechain_torsions (8)
        # Total: 34 node features (20 + 6 + 8)
        # Use LazyLinear to automatically infer input dimension from batch.x
        self.embedding = nn.LazyLinear(hidden_channels)
        
        # Interaction blocks
        self.interaction_blocks = nn.ModuleList([
            InteractionBlock(
                hidden_channels=hidden_channels,
                output_channels=hidden_channels,
                num_radial=num_radial,
                num_spherical=num_spherical,
                num_layers=int_emb_layers,
                mid_emb=mid_emb,
                act=F.silu,
                num_pos_emb=num_pos_emb,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])
        
        # Readout/pooling
        self.pool = get_aggregation(pool)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        if hasattr(self.embedding, 'reset_parameters'):
            self.embedding.reset_parameters()
        for block in self.interaction_blocks:
            block.reset_parameters()
    
    @property
    def required_batch_attributes(self) -> Set[str]:
        """Required batch attributes for ProNet backbone encoder
        
        - x: Node features (amino acid one-hot + dihedrals from ProteinWorkshop)
        - pos: CA coordinates
        - coords_n: N atom coordinates
        - coords_c: C atom coordinates
        - edge_index: Pre-computed edges (from ProteinWorkshop feature config)
        - batch: Batch assignment vector
        """
        return {"x", "pos", "coords_n", "coords_c", "edge_index", "batch"}
    
    def _compute_positional_embedding(self, edge_index, num_pos_emb: int = 16):
        """Compute sinusoidal positional embeddings from sequence distance"""
        # Port pos_emb from pronet.py
        # Returns: (num_edges, num_pos_emb)
        pass
    
    def _compute_theta_phi(self, pos, edge_index, num_nodes):
        """Compute theta and phi angles for edges"""
        # theta: angle between edge and previous node
        # phi: dihedral angle involving edge
        # Returns: theta, phi
        pass
    
    def _compute_euler_angles(self, pos, pos_n, pos_c, edge_index):
        """Compute Euler angles between local frames"""
        # Build local coordinate frames from (CA, N, C)
        # Compute three Euler angles between frames
        # Returns: angle1, angle2, angle3
        pass
    
    def _compute_dihedral_angle(self, pos, edge_index, num_nodes):
        """Compute dihedral angle (tau) for aminoacid level"""
        # Returns: tau
        pass
    
    def forward(self, batch: Union[Batch, ProteinBatch]) -> EncoderOutput:
        """Forward pass for backbone-level ProNet
        
        Workflow:
        1. Use ProteinWorkshop's pre-computed node features (batch.x)
        2. Extract CA, N, C coordinates
        3. Use pre-computed edges from batch.edge_index
        4. Compute ProNet-specific geometric features
        5. Apply interaction blocks
        6. Pool to graph level
        """
        # Extract coordinates
        pos = batch.pos           # CA coordinates (num_nodes, 3)
        pos_n = batch.coords_n    # N coordinates (num_nodes, 3)
        pos_c = batch.coords_c    # C coordinates (num_nodes, 3)
        batch_idx = batch.batch   # Batch assignment (num_nodes,)
        
        # Use ProteinWorkshop's node features directly
        # batch.x already contains: amino_acid_one_hot + dihedrals (from feature config)
        x = self.embedding(batch.x)  # (num_nodes, hidden_channels)
        
        # Use pre-computed edges from ProteinWorkshop (e.g., knn_30)
        edge_index = batch.edge_index  # (2, num_edges)
        j, i = edge_index  # Source and target nodes
        
        # Compute distances
        dist = (pos[i] - pos[j]).norm(dim=1)  # (num_edges,)
        
        # Compute geometric features (ProNet-specific)
        pos_emb = self._compute_positional_embedding(edge_index)
        theta, phi = self._compute_theta_phi(pos, edge_index, len(pos))
        
        # Encode (dist, theta, phi) with spherical harmonics
        feature0 = self.theta_phi_emb(dist, theta, phi)
        
        # Compute Euler angles between local frames
        angle1, angle2, angle3 = self._compute_euler_angles(pos, pos_n, pos_c, edge_index)
        
        # Encode three Euler angles
        feature1 = torch.cat([
            self.angle_emb(dist, angle1),
            self.angle_emb(dist, angle2),
            self.angle_emb(dist, angle3)
        ], dim=1)
        
        # Apply interaction blocks
        for block in self.interaction_blocks:
            x = block(x, feature0, feature1, pos_emb, edge_index, batch_idx)
        
        # Pool to graph level
        graph_embedding = self.pool(x, batch_idx)
        
        return EncoderOutput({
            "node_embedding": x,
            "graph_embedding": graph_embedding,
        })
```

---

### 3. Create Configuration File

**Location:** `proteinworkshop/config/encoder/pronet.yaml`

```yaml
_target_: proteinworkshop.models.graph_encoders.pronet.ProNetBackbone
num_blocks: 4
hidden_channels: 128
mid_emb: 64
num_radial: 6
num_spherical: 2
int_emb_layers: 3
num_pos_emb: 16
dropout: 0.0
pool: sum
```

---

### 4. Leverage ProteinWorkshop's Feature System

**Key Insight:** Use existing ProteinWorkshop features instead of custom embeddings. ProNet only needs to add N and C coordinates.

#### 4.1 Create ProNet backbone feature config

**Location:** `proteinworkshop/config/features/pronet_backbone.yaml`

```yaml
_target_: proteinworkshop.features.factory.ProteinFeaturiser
representation: CA  # CA is primary coordinate
scalar_node_features:
  - amino_acid_one_hot  # ProteinWorkshop's feature (20D)
  - dihedrals           # ProteinWorkshop's feature (6D: phi, psi, omega in sin/cos form)
  - sidechain_torsions  # ProteinWorkshop's feature (8D: chi1-chi4 in sin/cos form)
vector_node_features: []
edge_types:
  - knn_30              # Pre-compute KNN edges (can use knn_16, knn_30, eps_8, etc.)
scalar_edge_features: []
vector_edge_features: []
add_backbone_coords: true  # Enable N, C coordinate extraction (see 4.2)
```

**Why this works:**
- `amino_acid_one_hot`: Replaces DIVE's custom `num_aa_type` embedding (20D)
- `dihedrals`: Provides backbone dihedral angles φ, ψ, ω (6D: each angle as [sin, cos])
- `sidechain_torsions`: Provides sidechain chi angles χ1-χ4 (8D: each angle as [sin, cos])
- `knn_30`: Pre-computes spatial edges (similar to radius graph but more efficient)
- ProNet's spherical harmonics encode additional geometric information on these edges
- No need to compute edges dynamically in forward pass

#### 4.2 Add backbone atom coordinates to batch

**Option A: Extend ProteinFeaturiser (Recommended)**

Modify `proteinworkshop/features/factory.py` to optionally add backbone coordinates:

```python
class ProteinFeaturiser(nn.Module):
    def __init__(
        self,
        representation: StructureRepresentation,
        scalar_node_features: List[ScalarNodeFeature],
        vector_node_features: List[VectorNodeFeature],
        edge_types: List[str],
        scalar_edge_features: List[ScalarEdgeFeature],
        vector_edge_features: List[VectorEdgeFeature],
        add_backbone_coords: bool = False,  # NEW parameter
    ):
        # ... existing code ...
        self.add_backbone_coords = add_backbone_coords
    
    def forward(self, batch: Union[Batch, ProteinBatch]) -> Union[Batch, ProteinBatch]:
        # ... existing feature computation ...
        
        # Add backbone atom coordinates if requested
        if self.add_backbone_coords:
            batch = self._add_backbone_coordinates(batch)
        
        return batch
    
    def _add_backbone_coordinates(self, batch: Union[Batch, ProteinBatch]):
        """Extract N and C atom coordinates from batch
        
        ProteinBatch already contains coords for all atoms.
        Need to extract N and C specifically.
        """
        # Check if coordinates already exist
        if hasattr(batch, 'coords_n') and hasattr(batch, 'coords_c'):
            return batch
        
        # Extract from batch.coords (shape: num_residues x num_atoms_per_res x 3)
        # Standard PDB atom ordering: N, CA, C, O, ...
        if hasattr(batch, 'coords') and batch.coords.ndim == 3:
            batch.coords_n = batch.coords[:, 0, :]  # N atom (index 0)
            batch.coords_c = batch.coords[:, 2, :]  # C atom (index 2)
        else:
            # Fallback: parse from raw structure
            batch = self._parse_backbone_coords_from_structure(batch)
        
        return batch
    
    def _parse_backbone_coords_from_structure(self, batch):
        """Parse N, C coordinates from graphein structure"""
        # Implementation depends on how ProteinBatch stores raw data
        # Typically available in batch metadata or graph
        pass
```

**Update feature config:**

```yaml
# proteinworkshop/config/features/pronet_backbone.yaml
_target_: proteinworkshop.features.factory.ProteinFeaturiser
representation: CA
scalar_node_features:
  - amino_acid_one_hot
  - dihedrals
vector_node_features: []
edge_types:
  - knn_30  # Pre-compute KNN edges
scalar_edge_features: []
vector_edge_features: []
add_backbone_coords: true  # Enable N, C coordinate extraction
```

**Option B: Custom Transform (Alternative)**

If modifying ProteinFeaturiser is not desired, create standalone transform:

```python
# proteinworkshop/features/transforms.py

class AddBackboneCoordinates:
    """Transform to add N and C atom coordinates to batch"""
    
    def __call__(self, batch: Union[Batch, ProteinBatch]) -> Union[Batch, ProteinBatch]:
        if hasattr(batch, 'coords') and batch.coords.ndim == 3:
            batch.coords_n = batch.coords[:, 0, :]  # N atom
            batch.coords_c = batch.coords[:, 2, :]  # C atom
        return batch
```

Then compose in config:

```yaml
features:
  _target_: proteinworkshop.features.factory.ProteinFeaturiser
  # ... other params ...

transforms:
  - _target_: proteinworkshop.features.transforms.AddBackboneCoordinates
```

---

### 5. Testing Strategy

#### 5.1 Unit tests
**Location:** `tests/models/test_pronet.py`

```python
import pytest
import torch
from graphein.protein.tensor.data import get_random_batch
from proteinworkshop.models.graph_encoders.pronet import ProNetBackbone

def test_pronet_backbone_forward():
    """Test ProNet backbone forward pass"""
    model = ProNetBackbone(hidden_channels=64, num_blocks=2)
    
    # Create mock batch with required attributes
    batch = get_random_batch(batch_size=4)
    batch.x = torch.randn(batch.num_nodes, 34)  # 20 (amino acid) + 6 (dihedrals) + 8 (sidechain)
    batch.coords_n = torch.randn(batch.num_nodes, 3)
    batch.coords_c = torch.randn(batch.num_nodes, 3)
    
    output = model(batch)
    
    assert "node_embedding" in output
    assert "graph_embedding" in output
    assert output["node_embedding"].shape == (batch.num_nodes, 64)
    assert output["graph_embedding"].shape == (4, 64)

def test_pronet_geometric_features():
    """Test ProNet-specific geometric feature computation"""
    model = ProNetBackbone(hidden_channels=32)
    
    # Test with simple linear chain
    pos = torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]], dtype=torch.float)
    pos_n = pos + torch.tensor([0., 1., 0.])
    pos_c = pos + torch.tensor([0., -1., 0.])
    edge_index = torch.tensor([[0, 1], [1, 2]]).t()
    
    # Test theta/phi computation
    theta, phi = model._compute_theta_phi(pos, edge_index, len(pos))
    assert theta.shape == (2,)
    assert phi.shape == (2,)
    
    # Test Euler angle computation  
    angle1, angle2, angle3 = model._compute_euler_angles(pos, pos_n, pos_c, edge_index)
    assert angle1.shape == (2,)
    assert angle2.shape == (2,)
    assert angle3.shape == (2,)

def test_pronet_with_feature_config():
    """Test ProNet with ProteinWorkshop feature pipeline"""
    import hydra
    from proteinworkshop.features.factory import ProteinFeaturiser
    
    # Load feature config
    cfg = hydra.compose(
        config_name="train",
        overrides=["features=pronet_backbone", "encoder=pronet"]
    )
    
    featuriser = hydra.utils.instantiate(cfg.features)
    model = hydra.utils.instantiate(cfg.encoder)
    
    # Create and process batch
    batch = get_random_batch(batch_size=2)
    batch = featuriser(batch)
    
    # Verify required attributes exist
    assert hasattr(batch, 'x')
    assert hasattr(batch, 'pos')
    assert hasattr(batch, 'coords_n')
    assert hasattr(batch, 'coords_c')
    
    output = model(batch)
    assert output["node_embedding"].shape[0] == batch.num_nodes

#### 5.2 Integration tests

```python
def test_pronet_on_dataset():
    """Test ProNet training loop on small dataset"""
    cfg = compose_config(
        encoder="pronet",
        dataset="cath",
        task="fold_classification",
        features="pronet_ca",
        trainer={"max_epochs": 1}
    )
    
    # Run short training
    train_model(cfg)
    pass
```

---

### 6. Documentation Updates

#### 6.1 Add to model registry
**Location:** `docs/source/models.rst`

```markdown
### ProNet

Invariant graph neural network using spherical harmonics and Bessel functions.

**Key Features:**
- Three representation levels: aminoacid, backbone, allatom
- Computes geometric features from 3D coordinates
- Rotation and translation invariant

**Reference:** "Learning Protein Representations via Complete 3D Graph Networks"

**Config:** `encoder=pronet`
**Features:** `features=pronet_ca` (or `pronet_backbone`, `pronet_allatom`)
```

#### 6.2 Update adding_new_model_tutorial.ipynb
Add ProNet as an example of an invariant encoder with custom geometric features.

---

### 7. Key Differences from DIVE Implementation

| Aspect | DIVE Implementation | ProteinWorkshop Adaptation |
|--------|---------------------|----------------------------|
| **Scope** | 3 levels (aminoacid, backbone, allatom) | Focus on **backbone only** |
| **Node features** | Custom `bb_embs` (6D), `side_chain_embs` (8D) | Use `dihedrals` (6D) ✓ + `sidechain_torsions` (8D) ✓ |
| **Amino acid encoding** | `Embedding(26, hidden_dim)` or one-hot concat | Use `amino_acid_one_hot` (20D) feature |
| **Edge construction** | Dynamic `radius_graph(cutoff=10.0)` | Pre-computed `knn_30` (or other edge types) |
| **Batch structure** | Custom DIVE batch class | `ProteinBatch` / `Batch` |
| **Output** | Scalar prediction `y` | `EncoderOutput` dict with embeddings |
| **Feature computation** | Inline in forward | Extract to helper methods |
| **Coordinates** | Direct access `batch_data.coords_n` | Add via `add_backbone_coords=true` |
| **Activation** | Custom `swish` function | Use `F.silu` (equivalent) |

**Priority Decisions:**
1. ✅ Use ProteinWorkshop's `amino_acid_one_hot` instead of custom embedding
2. ✅ Use ProteinWorkshop's `dihedrals` + `sidechain_torsions` (14D total, matching DIVE's bb_embs + side_chain_embs)
3. ✅ Keep ProNet's spherical harmonic encodings (core contribution)
4. ✅ Adapt to ProteinWorkshop's batch structure
5. ✅ Return `EncoderOutput` for compatibility with tasks

---

### 8. Implementation Checklist

#### Phase 1: Core Implementation
- [ ] Port spherical harmonics/Bessel functions to `layers/pronet_features.py`
  - [ ] Utility functions (Jn, Jn_zeros, spherical_bessel_formulas, bessel_basis, real_sph_harm)
  - [ ] `AngleEmbedding` class
  - [ ] `ThetaPhiEmbedding` class
- [ ] Port ProNet layers to `layers/pronet.py`
  - [ ] `Linear` class
  - [ ] `TwoLinear` class
  - [ ] `EdgeGraphConv` class
  - [ ] `InteractionBlock` class
- [ ] Create main `pronet.py` encoder
  - [ ] `ProNetBackbone` class
  - [ ] Helper methods: `_compute_positional_embedding`, `_compute_theta_phi`, `_compute_euler_angles`
  - [ ] `required_batch_attributes` property
  - [ ] `forward` method

#### Phase 2: Configuration
- [ ] Create `config/encoder/pronet.yaml`
- [ ] Create `config/features/pronet_backbone.yaml`
- [ ] Modify `features/factory.py` to add `add_backbone_coords` parameter
  - [ ] Implement `_add_backbone_coordinates` method
  - [ ] Implement `_parse_backbone_coords_from_structure` method

#### Phase 3: Testing
- [ ] Unit tests (`tests/models/test_pronet.py`)
  - [ ] Test forward pass
  - [ ] Test geometric feature computation
  - [ ] Test with feature config
- [ ] Integration tests
  - [ ] Test on CATH dataset
  - [ ] Test on fold classification task
  - [ ] Verify training loop runs

#### Phase 4: Validation & Documentation
- [ ] Verify compatibility with existing datasets (CATH, Fold, EC)
- [ ] Benchmark against GearNet baseline (same task, same features)
- [ ] Add model documentation to `docs/source/models.rst`
- [ ] Update tutorial notebook with ProNet example
- [ ] Check memory usage and training time

---

## Notes

### Coordinate Handling
- ProNet backbone requires CA, N, C atom coordinates
- ProteinWorkshop standardizes CA coordinates as `batch.pos`
- N and C coordinates added via `add_backbone_coords=true` in feature config
- Standard PDB atom ordering: N (index 0), CA (index 1), C (index 2)

### Edge Construction
- **DIVE:** Uses dynamic `radius_graph(cutoff=10.0, max_num_neighbors=32)` in forward pass
- **ProteinWorkshop:** Pre-computes edges via feature config (e.g., `knn_30`, `eps_8`)
- **Benefit:** Pre-computation is faster and more consistent with ProteinWorkshop patterns
- **Recommendation:** Use `knn_30` as default (similar spatial coverage to radius graph with cutoff=10Å)

### Feature Mapping Strategy

| DIVE Feature | Dimension | ProteinWorkshop Equivalent | Dimension |
|--------------|-----------|---------------------------|-----------|
| `num_aa_type` one-hot | 26 | `amino_acid_one_hot` | 20 |
| `bb_embs` | 6 | `dihedrals` | 6 ✓ |
| `side_chain_embs` | 8 | `sidechain_torsions` | 8 ✓ |
| Custom spherical harmonics | ✓ | **Keep as-is** (core ProNet contribution) | ✓ |

**Rationale:**
- ProteinWorkshop uses standard 20 amino acids (excludes rare/modified residues)
- `dihedrals` provides backbone dihedral angles φ, ψ, ω in [sin, cos] form (6D total)
- `sidechain_torsions` provides chi angles χ1-χ4 in [sin, cos] form (8D total)
- ProNet's spherical harmonics encode additional 3D geometry not in ProteinWorkshop

### Performance Considerations
- **Spherical harmonics:** Pre-compute basis functions during `__init__` (lambdify once)
- **Edge pre-computation:** Using `knn_30` instead of dynamic radius graph improves speed
- **Memory:** Store edge features (3 types × num_edges); monitor with large proteins
- **Speed:** Expect slower than GearNet due to more complex geometric encoding, but faster than DIVE's dynamic edges

### Validation Plan
1. **Correctness:** Test geometric feature computation on known structures
2. **Invariance:** Verify rotation/translation invariance with random transforms
3. **Performance:** Benchmark on CATH fold classification vs GearNet
4. **Ablation:** Test with/without each feature stream (theta-phi, Euler, positional)

### Known Limitations
- Backbone-only implementation (aminoacid and allatom levels deferred)
- Requires all three backbone atoms (CA, N, C) - may fail on incomplete structures
- Higher computational cost than simpler invariant models
- More hyperparameters than GearNet (num_radial, num_spherical)

---

## Usage Examples

### Training from Command Line

```bash
# Train ProNet on fold classification
python proteinworkshop/train.py \
    encoder=pronet \
    features=pronet_backbone \
    dataset=cath \
    task=fold_classification \
    trainer.max_epochs=100

# Train with custom hyperparameters
python proteinworkshop/train.py \
    encoder=pronet \
    encoder.num_blocks=6 \
    encoder.hidden_channels=256 \
    features=pronet_backbone \
    features.edge_types=[knn_50] \
    dataset=cath \
    task=fold_classification
```



---

## Comparison with GearNet

| Feature | GearNet | ProNet |
|---------|---------|--------|
| **Edge construction** | Pre-computed (KNN/sequence) | Pre-computed (KNN) |
| **Geometric features** | Relative positions + orientations | Spherical harmonics + Euler angles |
| **Edge types** | Multiple types supported | Single type (spatial KNN) |
| **Complexity** | Moderate | High (spherical harmonics) |
| **Parameters** | ~2M (default) | ~2M (default) |
| **Speed** | Faster | Slower (more geometric encoding) |
| **Use case** | General purpose | High geometric fidelity |

**When to use ProNet over GearNet:**
- Tasks requiring fine-grained geometric understanding
- Datasets with complete backbone structures
- When computational cost is acceptable
- Research exploring geometric encodings
