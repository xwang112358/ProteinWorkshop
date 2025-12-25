# ProteinWorkshop Featurization Module - Workflow Guide

## Overview

The featurization module in ProteinWorkshop provides a flexible and modular framework for extracting features from protein structures. It supports multiple structural representations, various node and edge features, and different edge construction strategies. This guide explains the workflow, provides reproduction steps, and offers optimization suggestions.

---

## Table of Contents

1. [Featurization Workflow](#featurization-workflow)
2. [Architecture Components](#architecture-components)
3. [Step-by-Step Workflow](#step-by-step-workflow)
4. [How to Reproduce](#how-to-reproduce)
5. [Configuration Examples](#configuration-examples)
6. [Optimization Strategies](#optimization-strategies)
7. [Memory Usage Improvements](#memory-usage-improvements)
8. [Training Efficiency Tips](#training-efficiency-tips)

---

## Featurization Workflow

The featurization pipeline follows this high-level workflow:

```
Raw Protein Data → Scalar Node Features → Representation Transform → 
Vector Node Features → Edge Construction → Scalar Edge Features → 
Vector Edge Features → Featurized Graph
```

### Pipeline Stages

1. **Scalar Node Features**: Compute invariant node properties (amino acid type, angles, etc.)
2. **Representation Transform**: Convert between CA, backbone (BB), or full-atom (FA) representations
3. **Vector Node Features**: Compute equivariant node properties (orientations)
4. **Edge Construction**: Build graph connectivity (KNN, epsilon-radius, sequential)
5. **Scalar Edge Features**: Compute invariant edge properties (distances, types)
6. **Vector Edge Features**: Compute equivariant edge properties (edge vectors)

---

## Architecture Components

### Core Classes

#### 1. **ProteinFeaturiser** (`proteinworkshop/features/factory.py`)

The main orchestrator class that coordinates all featurization steps.

```python
class ProteinFeaturiser(nn.Module):
    def __init__(
        self,
        representation: StructureRepresentation,  # "ca", "ca_bb", or "full_atom"
        scalar_node_features: List[ScalarNodeFeature],
        vector_node_features: List[VectorNodeFeature],
        edge_types: List[str],
        scalar_edge_features: List[ScalarEdgeFeature],
        vector_edge_features: List[VectorEdgeFeature],
    )
```

**Key Features:**
- Modular design allowing mix-and-match of features
- Support for both invariant and equivariant features
- Automatic handling of batched data
- NaN handling for numerical stability

---

## Step-by-Step Workflow

### Step 1: Scalar Node Features

**Location**: `proteinworkshop/features/node_features.py`

**Available Features:**

| Feature | Description | Dimensionality | Type |
|---------|-------------|----------------|------|
| `amino_acid_one_hot` | One-hot encoding of amino acid type | 21 or 23 | Invariant |
| `sequence_positional_encoding` | Transformer-like positional encoding | 16 | Invariant |
| `alpha` | Virtual torsion angle (4 Cα atoms) | 2 (sin/cos) | Invariant |
| `kappa` | Virtual bond angle (3 Cα atoms) | 2 (sin/cos) | Invariant |
| `dihedrals` | Backbone dihedral angles (φ, ψ, ω) | 6 (sin/cos × 3) | Invariant |
| `sidechain_torsions` | Sidechain torsion angles (χ₁-χ₄) | 8 (sin/cos × 4) | Invariant |

**Code Flow:**
```python
def compute_scalar_node_features(x, node_features):
    feats = []
    for feature in node_features:
        if feature == "amino_acid_one_hot":
            feats.append(amino_acid_one_hot(x, num_classes=23))
        elif feature == "alpha":
            feats.append(alpha(x.coords, x.batch, rad=True, embed=True))
        # ... more features
    return torch.cat(feats, dim=1)
```

**Note**: All angular features are provided in [sin, cos] form for periodicity handling.

---

### Step 2: Representation Transform

**Location**: `proteinworkshop/features/representation.py`

**Available Representations:**

1. **CA (C-alpha only)**: One node per residue
   - Simplest representation
   - Uses only Cα coordinates
   - Memory efficient

2. **BB (Backbone)**: Four nodes per residue (N, Cα, C, O)
   - More detailed than CA
   - Captures backbone geometry
   - Node features are tiled over the 4 atoms

3. **FA (Full Atom)**: All atoms as nodes
   - Most detailed representation
   - Highest memory requirement
   - Each atom becomes a node

4. **BB_SC (Backbone + Sidechain centroid)**: 5 nodes per residue

5. **CA_SC (C-alpha + Sidechain centroid)**: 2 nodes per residue

**Code Flow:**
```python
def transform_representation(x, representation_type):
    if representation_type == "CA":
        x.pos = x.coords[:, 1, :]  # Select Cα
    elif representation_type == "BB":
        return ca_to_bb_repr(x)  # Expand to 4 nodes/residue
    elif representation_type == "FA":
        return ca_to_fa_repr(x)  # Expand to all atoms
    # ...
    return x
```

**Memory Impact:**
- CA: Base memory (N nodes)
- BB: 4× CA memory
- FA: ~15-30× CA memory (depends on amino acid composition)

---

### Step 3: Vector Node Features

**Location**: `proteinworkshop/features/node_features.py`

**Available Features:**

| Feature | Description | Dimensionality | Type |
|---------|-------------|----------------|------|
| `orientation` | Forward and backward unit vectors | 2 vectors (6D) | Equivariant |

**Code Flow:**
```python
def compute_vector_node_features(x, vector_features):
    vector_node_features = []
    for feature in vector_features:
        if feature == "orientation":
            # Compute forward/backward orientation vectors
            vector_node_features.append(orientations(x.coords, x._slice_dict["coords"]))
    x.x_vector_attr = torch.cat(vector_node_features, dim=0)
    return x
```

**Orientation Computation:**
- Forward vector: `normalize(pos[i+1] - pos[i])`
- Backward vector: `normalize(pos[i-1] - pos[i])`
- Handles batch boundaries correctly
- Zero-padded for terminal residues

---

### Step 4: Edge Construction

**Location**: `proteinworkshop/features/edges.py`

**Available Edge Types:**

1. **KNN Edges** (`knn_k`): K-nearest neighbors
   - Example: `knn_10`, `knn_16`, `knn_30`
   - Spatial proximity based
   - Requires `pos` attribute

2. **Epsilon Edges** (`eps_r`): Radius-based
   - Example: `eps_8`, `eps_16`
   - All nodes within distance threshold
   - Variable degree

3. **Sequential Edges**:
   - `seq_forward`: i → i+1 edges
   - `seq_backward`: i+1 → i edges
   - Respects chain boundaries

**Code Flow:**
```python
def compute_edges(x, edge_types):
    edges = []
    for edge_type in edge_types:
        if edge_type.startswith("knn"):
            k = int(edge_type.split("_")[1])
            edges.append(knn_graph(x.pos, k=k))
        elif edge_type.startswith("eps"):
            r = float(edge_type.split("_")[1])
            edges.append(radius_graph(x.pos, r=r))
        # ... sequential edges
    
    # Concatenate and track edge types
    edge_index = torch.cat(edges, dim=1)
    edge_type = torch.cat([torch.full((e.shape[1],), i) for i, e in enumerate(edges)])
    return edge_index, edge_type
```

**Trade-offs:**
- KNN: Fixed degree, good for GNNs, moderate memory
- Epsilon: Variable degree, captures local geometry, potentially higher memory
- Sequential: Minimal edges, captures sequence information

---

### Step 5: Scalar Edge Features

**Location**: `proteinworkshop/features/edge_features.py`

**Available Features:**

| Feature | Description | Dimensionality |
|---------|-------------|----------------|
| `edge_distance` | Euclidean distance between nodes | 1 |
| `node_features` | Concatenated source + target node features | 2 × node_dim |
| `edge_type` | Edge type indicator | 1 |
| `sequence_distance` | Sequence separation (j - i) | 1 |
| `pos_emb` | Structured positional embedding | 16 |

**Code Flow:**
```python
def compute_scalar_edge_features(x, features):
    feats = []
    for feature in features:
        if feature == "edge_distance":
            feats.append(compute_edge_distance(x.pos, x.edge_index))
        elif feature == "node_features":
            n1, n2 = x.x[x.edge_index[0]], x.x[x.edge_index[1]]
            feats.append(torch.cat([n1, n2], dim=1))
        # ... more features
    return torch.cat(feats, dim=1)
```

---

### Step 6: Vector Edge Features

**Location**: `proteinworkshop/features/edge_features.py`

**Available Features:**

| Feature | Description | Dimensionality |
|---------|-------------|----------------|
| `edge_vectors` | Unit-normalized directional vectors | 1 vector (3D) |

**Code Flow:**
```python
def compute_vector_edge_features(x, features):
    vector_edge_features = []
    for feature in features:
        if feature == "edge_vectors":
            E_vectors = x.pos[x.edge_index[0]] - x.pos[x.edge_index[1]]
            vector_edge_features.append(_normalize(E_vectors).unsqueeze(-2))
    x.edge_vector_attr = torch.cat(vector_edge_features, dim=0)
    return x
```

---

## How to Reproduce

### Method 1: Using the CLI

```bash
# Basic usage with predefined feature config
workshop train dataset=cath encoder=egnn task=inverse_folding features=ca_bb trainer=cpu

# Custom features via command line
workshop train \
    dataset=cath \
    encoder=egnn \
    task=inverse_folding \
    features.representation=CA \
    features.scalar_node_features=[amino_acid_one_hot,dihedrals] \
    features.edge_types=[knn_16] \
    features.scalar_edge_features=[edge_distance] \
    trainer=gpu
```

### Method 2: Using Python API

```python
from proteinworkshop.features.factory import ProteinFeaturiser
from proteinworkshop.datasets.cath import CATHDataModule
from torch_geometric.loader import DataLoader

# Initialize featuriser
featuriser = ProteinFeaturiser(
    representation="CA",
    scalar_node_features=["amino_acid_one_hot", "dihedrals"],
    vector_node_features=["orientation"],
    edge_types=["knn_16"],
    scalar_edge_features=["edge_distance"],
    vector_edge_features=["edge_vectors"],
)

# Load dataset
datamodule = CATHDataModule(
    path="data/cath/",
    pdb_dir="data/pdb/",
    format="mmtf",
    batch_size=32
)
datamodule.setup()

# Apply featurisation
for batch in datamodule.train_dataloader():
    # Add sequence position for positional encoding
    batch.seq_pos = torch.arange(batch.coords.shape[0], dtype=torch.long)
    
    # Featurize
    featurized_batch = featuriser(batch)
    
    # Now use with your model
    # output = model(featurized_batch)
```

### Method 3: Creating Custom Feature Configs

Create a YAML config file at `proteinworkshop/config/features/my_features.yaml`:

```yaml
_target_: proteinworkshop.features.factory.ProteinFeaturiser
representation: CA
scalar_node_features:
  - amino_acid_one_hot
  - dihedrals
  - alpha
  - kappa
vector_node_features: []
edge_types:
  - knn_16
  - knn_32  # Can use multiple edge types
scalar_edge_features:
  - edge_distance
  - edge_type
vector_edge_features: []
```

Then use it:

```bash
workshop train dataset=cath encoder=egnn task=inverse_folding features=my_features trainer=gpu
```

### Method 4: Testing Individual Features

```python
from proteinworkshop.datasets.utils import create_example_batch
from proteinworkshop.features.factory import ProteinFeaturiser

# Create example batch
batch = create_example_batch()
batch.seq_pos = torch.arange(batch.coords.shape[0], dtype=torch.long)

# Test CA features
ca_featuriser = ProteinFeaturiser(
    representation="CA",
    scalar_node_features=["amino_acid_one_hot"],
    vector_node_features=[],
    edge_types=["knn_10"],
    scalar_edge_features=["edge_distance"],
    vector_edge_features=[],
)
ca_batch = ca_featuriser(batch)
print(f"CA nodes: {ca_batch.num_nodes}, edges: {ca_batch.edge_index.shape[1]}")

# Test BB features (4x more nodes)
bb_featuriser = ProteinFeaturiser(
    representation="BB",
    scalar_node_features=["amino_acid_one_hot"],
    vector_node_features=[],
    edge_types=["knn_10"],
    scalar_edge_features=["edge_distance"],
    vector_edge_features=[],
)
bb_batch = bb_featuriser(batch)
print(f"BB nodes: {bb_batch.num_nodes}, edges: {bb_batch.edge_index.shape[1]}")
```

---

## Configuration Examples

### 1. Minimal Configuration (Fast, Low Memory)

```yaml
_target_: proteinworkshop.features.factory.ProteinFeaturiser
representation: CA
scalar_node_features:
  - amino_acid_one_hot
vector_node_features: []
edge_types:
  - knn_10
scalar_edge_features:
  - edge_distance
vector_edge_features: []
```

**Use Case**: Quick prototyping, limited GPU memory

**Memory**: Base
**Speed**: Fastest

---

### 2. Invariant Full Configuration

```yaml
_target_: proteinworkshop.features.factory.ProteinFeaturiser
representation: CA
scalar_node_features:
  - amino_acid_one_hot
  - sequence_positional_encoding
  - dihedrals
  - alpha
  - kappa
  - sidechain_torsions
vector_node_features: []
edge_types:
  - knn_30
scalar_edge_features:
  - edge_distance
  - edge_type
  - sequence_distance
vector_edge_features: []
```

**Use Case**: Invariant GNNs (GearNet, DimeNet++, SchNet)

**Memory**: Moderate
**Speed**: Fast

---

### 3. Equivariant Configuration

```yaml
_target_: proteinworkshop.features.factory.ProteinFeaturiser
representation: CA
scalar_node_features:
  - dihedrals
vector_node_features:
  - orientation
edge_types:
  - knn_30
scalar_edge_features:
  - edge_distance
vector_edge_features:
  - edge_vectors
```

**Use Case**: Equivariant GNNs (EGNN, GVP-GNN, TFN)

**Memory**: Moderate
**Speed**: Moderate

---

### 4. Full-Atom Configuration

```yaml
_target_: proteinworkshop.features.factory.ProteinFeaturiser
representation: FA
scalar_node_features:
  - amino_acid_one_hot
vector_node_features: []
edge_types:
  - knn_10  # Lower k for full-atom to manage memory
scalar_edge_features:
  - edge_distance
vector_edge_features: []
```

**Use Case**: Detailed atomic interactions

**Memory**: High (15-30× CA)
**Speed**: Slow

---

## Optimization Strategies

### 1. Training Efficiency Improvements

#### A. Feature Computation Optimization

**Problem**: Features are computed on-the-fly during data loading.

**Solutions**:

1. **Pre-compute and Cache Features**

```python
# Create a preprocessing script
import torch
from tqdm import tqdm

def precompute_features(datamodule, featuriser, save_path):
    """Pre-compute and save featurized graphs."""
    datamodule.setup()
    featurized_data = []
    
    for batch in tqdm(datamodule.train_dataloader()):
        batch.seq_pos = torch.arange(batch.coords.shape[0])
        featurized_batch = featuriser(batch)
        featurized_data.append(featurized_batch.to('cpu'))
    
    torch.save(featurized_data, save_path)
    return featurized_data

# Then load during training
cached_data = torch.load('cached_features.pt')
train_loader = DataLoader(cached_data, batch_size=32, shuffle=True)
```

**Impact**: 2-5× faster data loading, reduced CPU bottleneck

---

2. **Use Mixed Precision Training**

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():
        output = model(batch)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Impact**: 1.5-2× faster training, 30-40% memory reduction

---

3. **Optimize Edge Construction**

**Current**: Multiple edge types computed separately

**Optimization**: Combine edge types to avoid redundant computation

```python
# Instead of:
edge_types = ["knn_16", "knn_32"]  # Computes KNN twice

# Use:
edge_types = ["knn_32"]  # Compute once, filter if needed
# Or if you need both, compute knn_32 and subsample for knn_16
```

**Impact**: Up to 40% reduction in edge computation time

---

4. **Parallelize Feature Computation**

```python
from torch_geometric.loader import DataLoader

# Use multiple workers for parallel data loading
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,  # Parallel feature computation
    pin_memory=True,  # Faster GPU transfer
    persistent_workers=True  # Keep workers alive between epochs
)
```

**Impact**: 2-4× faster data loading (depends on CPU cores)

---

#### B. Model Architecture Optimization

1. **Gradient Checkpointing**

```python
import torch.utils.checkpoint as checkpoint

class OptimizedModel(nn.Module):
    def forward(self, x):
        # Use checkpointing for memory-intensive layers
        x = checkpoint.checkpoint(self.expensive_layer, x)
        return x
```

**Impact**: 30-50% memory reduction, 10-20% slower

---

2. **Use Efficient Edge Update Mechanisms**

Replace message passing with more efficient alternatives:

```python
# Instead of full message passing:
for _ in range(num_layers):
    x = message_passing_layer(x, edge_index)

# Use attention pooling or graph transformers with sparse attention
```

---

### 2. Memory Usage Improvements

#### A. Representation Selection

**Memory Hierarchy** (ascending):
1. CA: Base memory
2. BB: 4× CA
3. CA_SC: 2× CA (but with coarsened sidechains)
4. BB_SC: 5× CA
5. FA: 15-30× CA

**Recommendation**:
- Start with CA for most tasks
- Use BB only if backbone detail is critical
- Avoid FA unless absolutely necessary

---

#### B. Edge Type Optimization

**Memory Impact**:
- KNN with k=10: ~10N edges
- KNN with k=30: ~30N edges
- Epsilon edges: Variable (10-50N typical)

**Strategies**:

1. **Use Lower k for Initial Training**
```python
# Phase 1: Train with knn_10 for quick convergence
edge_types = ["knn_10"]

# Phase 2: Fine-tune with knn_30 for better performance
edge_types = ["knn_30"]
```

2. **Progressive Edge Expansion**
```python
def get_edge_types(epoch):
    if epoch < 10:
        return ["knn_10"]
    elif epoch < 20:
        return ["knn_16"]
    else:
        return ["knn_30"]
```

---

#### C. Batch Size Optimization

**Memory Usage**: `Memory ∝ batch_size × num_nodes × (node_features + edge_features × avg_degree)`

**Strategies**:

1. **Dynamic Batch Sizing Based on Graph Size**

```python
from torch_geometric.loader import DataLoader

def dynamic_batch_size(graph_size):
    """Adjust batch size based on graph size."""
    if graph_size < 100:
        return 32
    elif graph_size < 300:
        return 16
    else:
        return 8

# Use with DataLoader
dataloader = DataLoader(
    dataset,
    batch_sampler=DynamicBatchSampler(dataset, dynamic_batch_size),
)
```

2. **Gradient Accumulation**

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    output = model(batch)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Impact**: Effective batch size × accumulation_steps, same memory as batch_size

---

#### D. Feature Dimensionality Reduction

**Current Feature Dimensions**:
- amino_acid_one_hot: 23D
- sequence_positional_encoding: 16D
- dihedrals: 6D
- alpha: 2D
- kappa: 2D
- sidechain_torsions: 8D

**Optimization**:

1. **Use Learned Embeddings Instead of One-Hot**

```python
class OptimizedFeaturiser(ProteinFeaturiser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace 23D one-hot with 8D learned embedding
        self.aa_embedding = nn.Embedding(23, 8)
    
    def forward(self, batch):
        # Replace amino_acid_one_hot with embedding
        batch.x = self.aa_embedding(batch.residue_type)
        # ... rest of features
        return super().forward(batch)
```

**Impact**: 23D → 8D, ~65% memory reduction for this feature

---

2. **Feature Selection**

Based on ablation studies, prioritize:
- **Essential**: `amino_acid_one_hot`, `edge_distance`
- **Important**: `dihedrals`, `kappa`
- **Optional**: `sidechain_torsions`, `alpha`

Start with essential features, add others if needed.

---

#### E. In-Place Operations

**Modify the featuriser to use in-place operations**:

```python
# In factory.py, replace:
batch.x = torch.cat([batch.x, scalar_features], dim=-1)

# With:
batch.x = torch.cat([batch.x, scalar_features], dim=-1, out=batch.x)

# Or better, pre-allocate:
batch_x = torch.empty(
    (batch.num_nodes, total_feature_dim),
    dtype=batch.x.dtype,
    device=batch.x.device
)
batch_x[:, :batch.x.shape[1]] = batch.x
batch_x[:, batch.x.shape[1]:] = scalar_features
batch.x = batch_x
```

---

#### F. Memory Profiling and Monitoring

**Add memory tracking**:

```python
from proteinworkshop.utils.memory_utils import gpu_memory_usage, clean_up_torch_gpu_memory

# During training
for epoch in range(num_epochs):
    for batch in dataloader:
        # Monitor memory
        mem_before = gpu_memory_usage()
        
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        mem_after = gpu_memory_usage()
        print(f"Memory used: {mem_after - mem_before:.2f} GB")
        
        # Clean up if needed
        if mem_after > 10.0:  # If using >10GB
            clean_up_torch_gpu_memory()
```

---

### 3. Advanced Optimization Techniques

#### A. Mixed Graph Representations

Use different representations for different stages:

```python
class MultiScaleFeaturiser(nn.Module):
    def __init__(self):
        self.ca_featuriser = ProteinFeaturiser(representation="CA", ...)
        self.bb_featuriser = ProteinFeaturiser(representation="BB", ...)
    
    def forward(self, batch, stage="coarse"):
        if stage == "coarse":
            return self.ca_featuriser(batch)
        else:
            return self.bb_featuriser(batch)

# Training loop
for epoch in range(num_epochs):
    stage = "coarse" if epoch < 20 else "fine"
    for batch in dataloader:
        featurized = featuriser(batch, stage=stage)
```

---

#### B. Sparse Feature Computation

Only compute expensive features for important nodes:

```python
def selective_feature_computation(batch, importance_threshold=0.5):
    """Compute expensive features only for important nodes."""
    # Compute cheap features for all nodes
    batch.x = amino_acid_one_hot(batch)
    
    # Compute importance scores (e.g., based on attention)
    importance = compute_node_importance(batch)
    important_mask = importance > importance_threshold
    
    # Compute expensive features only for important nodes
    if important_mask.any():
        sidechain_features = sidechain_torsion(batch.coords[important_mask])
        batch.x[important_mask] = torch.cat([
            batch.x[important_mask],
            sidechain_features
        ], dim=-1)
    
    return batch
```

---

#### C. Use Compiled Models (PyTorch 2.0+)

```python
import torch._dynamo as dynamo

# Compile the featuriser
compiled_featuriser = torch.compile(featuriser, mode="reduce-overhead")

# Use in training
for batch in dataloader:
    featurized_batch = compiled_featuriser(batch)
```

**Impact**: 10-30% speedup with PyTorch 2.0+

---

## Training Efficiency Tips

### 1. Optimal Configuration for Different Scenarios

#### Scenario A: Limited GPU Memory (<16GB)

```yaml
# Configuration
representation: CA
scalar_node_features:
  - amino_acid_one_hot
vector_node_features: []
edge_types:
  - knn_10
scalar_edge_features:
  - edge_distance
vector_edge_features: []

# Training settings
batch_size: 8
gradient_accumulation_steps: 4
mixed_precision: true
```

---

#### Scenario B: Balanced Performance (16-32GB GPU)

```yaml
# Configuration
representation: CA
scalar_node_features:
  - amino_acid_one_hot
  - dihedrals
  - kappa
vector_node_features: []
edge_types:
  - knn_16
scalar_edge_features:
  - edge_distance
  - edge_type
vector_edge_features: []

# Training settings
batch_size: 32
num_workers: 4
mixed_precision: true
```

---

#### Scenario C: High Performance (>32GB GPU)

```yaml
# Configuration
representation: BB
scalar_node_features:
  - amino_acid_one_hot
  - sequence_positional_encoding
  - dihedrals
  - alpha
  - kappa
vector_node_features:
  - orientation
edge_types:
  - knn_30
scalar_edge_features:
  - edge_distance
  - edge_type
  - pos_emb
vector_edge_features:
  - edge_vectors

# Training settings
batch_size: 64
num_workers: 8
mixed_precision: true
```

---

### 2. Profiling and Benchmarking

**Create a benchmarking script**:

```python
import time
import torch
from proteinworkshop.features.factory import ProteinFeaturiser
from proteinworkshop.datasets.utils import create_example_batch

def benchmark_featuriser(config, num_iterations=100):
    """Benchmark featurisation speed and memory."""
    featuriser = ProteinFeaturiser(**config)
    batch = create_example_batch()
    batch.seq_pos = torch.arange(batch.coords.shape[0])
    
    # Warmup
    for _ in range(10):
        _ = featuriser(batch)
    
    # Benchmark
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
    for _ in range(num_iterations):
        featurized = featuriser(batch)
        torch.cuda.synchronize()
    
    elapsed = time.time() - start_time
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    
    print(f"Time per iteration: {elapsed/num_iterations*1000:.2f} ms")
    print(f"Peak memory: {peak_memory:.2f} GB")
    print(f"Throughput: {num_iterations/elapsed:.2f} batches/sec")
    
    return {
        'time_per_iter': elapsed/num_iterations,
        'peak_memory_gb': peak_memory,
        'throughput': num_iterations/elapsed
    }

# Run benchmarks
configs = {
    'minimal': {
        'representation': 'CA',
        'scalar_node_features': ['amino_acid_one_hot'],
        'vector_node_features': [],
        'edge_types': ['knn_10'],
        'scalar_edge_features': ['edge_distance'],
        'vector_edge_features': [],
    },
    'full': {
        'representation': 'CA',
        'scalar_node_features': ['amino_acid_one_hot', 'dihedrals', 'alpha', 'kappa'],
        'vector_node_features': ['orientation'],
        'edge_types': ['knn_30'],
        'scalar_edge_features': ['edge_distance', 'edge_type'],
        'vector_edge_features': ['edge_vectors'],
    },
}

for name, config in configs.items():
    print(f"\n=== Benchmarking {name} ===")
    benchmark_featuriser(config)
```

---

## Summary of Recommendations

### Memory Optimization Priority

1. **High Impact**:
   - Use CA representation instead of BB/FA
   - Reduce k in KNN edges (knn_10 vs knn_30)
   - Use mixed precision training
   - Enable gradient checkpointing

2. **Medium Impact**:
   - Reduce batch size with gradient accumulation
   - Use learned embeddings instead of one-hot
   - Pre-compute and cache features
   - Feature selection (only essential features)

3. **Low Impact**:
   - In-place operations
   - Dynamic batch sizing
   - Memory profiling and cleanup

### Training Speed Optimization Priority

1. **High Impact**:
   - Pre-compute features
   - Use multiple workers (num_workers=4-8)
   - Mixed precision training
   - Compile models (PyTorch 2.0+)

2. **Medium Impact**:
   - Optimize edge construction
   - Use persistent workers
   - Pin memory in DataLoader
   - Reduce feature computation redundancy

3. **Low Impact**:
   - Progressive edge expansion
   - Multi-scale representations

---

## Conclusion

The ProteinWorkshop featurization module provides a flexible and powerful framework for protein structure featurization. By understanding the workflow and applying the optimization strategies outlined in this guide, you can:

1. **Reproduce** any featurization configuration using CLI, Python API, or custom configs
2. **Optimize memory usage** by up to 70% through representation selection, edge optimization, and feature reduction
3. **Improve training speed** by 2-5× through pre-computation, parallelization, and mixed precision

Start with the minimal configuration for rapid prototyping, then progressively add features and complexity as needed. Always profile your specific use case to identify bottlenecks and apply targeted optimizations.

For questions or contributions, see the [ProteinWorkshop repository](https://github.com/a-r-j/ProteinWorkshop).
