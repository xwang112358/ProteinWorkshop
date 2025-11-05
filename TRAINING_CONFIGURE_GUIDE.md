### 2.4 Task Configuration (EC ProteinShake - Accuracy Only)

**STATUS:** ✅ You can use the built-in `multiclass_graph_classification` task without creating a custom config.

The built-in task already includes accuracy monitoring:

```yaml
# proteinworkshop/config/task/multiclass_graph_classification.yaml (Built-in)
# @package _global_

defaults:
  - override /metrics:
      - accuracy        # ✅ Primary metric for your EC dataset
      - f1_score        # Additional metric (optional)
      - f1_max          # Additional metric (optional)
  - override /decoder:
      - graph_label     # ✅ Graph-level classification

callbacks:
  early_stopping:
    monitor: val/graph_label/accuracy    # ✅ Monitors validation accuracy
    mode: "max"
  model_checkpoint:
    monitor: val/graph_label/accuracy    # ✅ Saves best model by validation accuracy
    mode: "max"

task:
  task: "classification"
  classification_type: "multiclass"     # ✅ 7-class EC classification
  metric_average: "micro"

  losses:
    graph_label: cross_entropy
  label_smoothing: 0.0

  output:
    - "graph_label"
  supervise_on:
    - "graph_label"
```

**For Your EC Dataset:**
- ✅ Use `task=multiclass_graph_classification` directly (no custom task needed)
- ✅ Accuracy is already the primary metric for early stopping and checkpointing
- ✅ Test accuracy will be automatically computed and reported
- ✅ Same task config works for both `ec_random` and `ec_structure` splits

**Training Commands:**
```bash
# Random split - Use built-in task
workshop train dataset=ec_random task=multiclass_graph_classification encoder=gear_net

# Structure split - Use built-in task
workshop train dataset=ec_structure task=multiclass_graph_classification encoder=gear_net
```

**Optional: Create Custom Task Config (If You Want Accuracy ONLY)**

If you want to remove F1 metrics and use accuracy only:

```yaml
# proteinworkshop/config/task/ec_classification.yaml (OPTIONAL - Not Required)
# @package _global_

defaults:
  - override /metrics:
      - accuracy        # ONLY accuracy metric
  - override /decoder:
      - graph_label

callbacks:
  early_stopping:
    monitor: val/graph_label/accuracy
    mode: "max"
    patience: 50
  model_checkpoint:
    monitor: val/graph_label/accuracy
    mode: "max"
    save_top_k: 1

task:
  task: "classification"
  classification_type: "multiclass"
  metric_average: "micro"

  losses:
    graph_label: cross_entropy
  label_smoothing: 0.0

  output:
    - "graph_label"
  supervise_on:
    - "graph_label"
```

---

## 3. Feature Engineering for EC ProteinShake Dataset

### 3.1 Your Specified Feature Configuration

**Your Requirements:**
- **Edge Type:** `knn_30` (30-nearest neighbors)
- **Node Features:**
  - Dihedrals (backbone φ, ψ, ω angles)
  - Orientation (forward/backward vectors)
  - One-hot (amino acid encoding)
- **Edge Features:**
  - Edge distance (Euclidean distance)
  - Edge vectors (directional unit vectors)

### 3.2 Create Custom Feature Config for EC Dataset

**File:** `proteinworkshop/config/features/ec_custom.yaml`

```yaml
# Custom feature configuration for EC ProteinShake dataset
_target_: proteinworkshop.features.factory.ProteinFeaturiser
representation: CA              # C-alpha atoms as nodes

# Node features: one-hot + dihedrals + orientation
scalar_node_features:
  - amino_acid_one_hot          # 21-dim one-hot encoding (20 AA + unknown)
  - dihedrals                   # 6-dim (φ, ψ, ω angles in sin/cos: 3 angles × 2 = 6)

vector_node_features:
  - orientation                 # 2 × 3-dim (forward/backward direction vectors)

# Edge construction: k-NN with k=30
edge_types:
  - knn_30                      # 30-nearest neighbors

# Edge features: distance + directional vectors
scalar_edge_features:
  - edge_distance               # 1-dim Euclidean distance

vector_edge_features:
  - edge_vectors                # 3-dim directional unit vectors (target - source)
```

**Feature Dimensions:**
- **Scalar Node Features:** 21 (one-hot) + 6 (dihedrals) = 27 dimensions
- **Vector Node Features:** 2 × 3 = 6 dimensions (2 orientation vectors, each 3D)
- **Scalar Edge Features:** 1 dimension (distance)
- **Vector Edge Features:** 3 dimensions (unit vector)

### 3.3 How Feature Engineering Works

```
PDB File (1abc.pdb)
    ↓
Graphein Parser → AtomTensor (N_residues × 37 atoms × 3 coordinates)
    ↓
ProteinFeaturiser (proteinworkshop/config/features/ec_custom.yaml)
    ↓
    ├─ Representation: Select C-alpha atoms (N_residues nodes)
    ├─ Node Features:
    │   ├─ Scalar: amino_acid_one_hot (21-dim) + dihedrals (6-dim)
    │   └─ Vector: orientation (2 × 3-dim)
    ├─ Edge Construction: knn_30 (each node connects to 30 nearest neighbors)
    └─ Edge Features:
        ├─ Scalar: edge_distance (1-dim)
        └─ Vector: edge_vectors (3-dim)
    ↓
PyTorch Geometric Data object
    ↓
Model (GearNet / GVP / GCPNet)
```

### 3.4 Alternative Feature Configurations

**Option 1: Invariant Features Only (for GearNet)**

```yaml
# proteinworkshop/config/features/ec_invariant.yaml
_target_: proteinworkshop.features.factory.ProteinFeaturiser
representation: CA

scalar_node_features:
  - amino_acid_one_hot
  - dihedrals                   # φ, ψ, ω angles (6-dim)
  - alpha                       # Virtual torsion angle (2-dim)
  - kappa                       # Virtual bond angle (2-dim)

vector_node_features: []        # No vector features

edge_types:
  - knn_30

scalar_edge_features:
  - edge_distance

vector_edge_features: []        # No vector features
```

**Pseudo Code for GearNet Training:**
```python
# GearNet only uses scalar features
model = GearNet(
    input_dim=21 + 6 + 2 + 2,  # one-hot + dihedrals + alpha + kappa = 31
    edge_input_dim=1,           # edge_distance only
    hidden_dim=512,
    num_layers=6
)

# Training loop
for batch in dataloader:
    # batch.x: [N_nodes, 31] scalar features
    # batch.edge_attr: [N_edges, 1] scalar edge features
    # batch.edge_index: [2, N_edges] connectivity
    
    output = model(batch.x, batch.edge_index, batch.edge_attr)
    loss = cross_entropy(output, batch.graph_label)
```

**Option 2: Equivariant Features (for GVP/GCPNet)**

```yaml
# proteinworkshop/config/features/ec_equivariant.yaml (YOUR CURRENT SPEC)
_target_: proteinworkshop.features.factory.ProteinFeaturiser
representation: CA

scalar_node_features:
  - amino_acid_one_hot          # 21-dim
  - dihedrals                   # 6-dim

vector_node_features:
  - orientation                 # 2 × 3-dim (REQUIRED for equivariant models)

edge_types:
  - knn_30

scalar_edge_features:
  - edge_distance               # 1-dim

vector_edge_features:
  - edge_vectors                # 3-dim (REQUIRED for equivariant models)
```

**Pseudo Code for GVP Training:**
```python
# GVP uses both scalar and vector features
model = GVP(
    node_s_dim=27,              # scalar node features: 21 + 6
    node_v_dim=2,               # vector node features: 2 orientation vectors
    edge_s_dim=1,               # scalar edge features: distance
    edge_v_dim=1,               # vector edge features: edge_vectors
    hidden_s_dim=128,
    hidden_v_dim=16,
    num_layers=5
)

# Training loop
for batch in dataloader:
    # batch.x: [N_nodes, 27] scalar features
    # batch.orientation: [N_nodes, 2, 3] vector features
    # batch.edge_attr: [N_edges, 1] scalar edge features
    # batch.edge_vectors: [N_edges, 3] vector edge features
    
    output = model(
        node_s=batch.x,
        node_v=batch.orientation,
        edge_index=batch.edge_index,
        edge_s=batch.edge_attr,
        edge_v=batch.edge_vectors
    )
    loss = cross_entropy(output, batch.graph_label)
```

**Option 3: Multiple Edge Types (for GearNet/GCPNet)**

```yaml
# proteinworkshop/config/features/ec_multi_edge.yaml
_target_: proteinworkshop.features.factory.ProteinFeaturiser
representation: CA

scalar_node_features:
  - amino_acid_one_hot
  - dihedrals

vector_node_features:
  - orientation

edge_types:
  - knn_30                      # 30-nearest neighbors
  - eps_10                      # All edges within 10 Å
  - sequential                  # Sequential edges (i, i+1)

scalar_edge_features:
  - edge_distance
  - edge_type                   # Edge type ID (0=knn, 1=eps, 2=sequential)
  - sequence_distance           # |i - j| along sequence

vector_edge_features:
  - edge_vectors
```

**Pseudo Code for Multi-Edge GearNet:**
```python
# GearNet with relation-aware message passing
model = GearNet(
    input_dim=27,               # 21 + 6
    edge_input_dim=3,           # distance + type + seq_distance
    num_relations=3,            # 3 edge types (knn, eps, sequential)
    hidden_dim=512,
    num_layers=6
)

# Training loop with relation-aware edges
for batch in dataloader:
    # batch.edge_attr: [N_edges, 3] includes edge_type
    # Different message passing for different edge types
    
    output = model(batch.x, batch.edge_index, batch.edge_attr, batch.edge_type)
    loss = cross_entropy(output, batch.graph_label)
```

### 3.5 Feature Featuriser Pipeline Details

The `ProteinFeaturiser` automatically handles:

1. **PDB Parsing:** Reads PDB files → AtomTensor
2. **Representation Selection:** Filters to C-alpha atoms (N_res nodes)
3. **Node Feature Computation:**
   - `amino_acid_one_hot`: One-hot encodes residue types
   - `dihedrals`: Computes backbone φ, ψ, ω from coordinates
   - `orientation`: Computes forward/backward direction vectors
4. **Edge Construction:**
   - `knn_30`: For each node, finds 30 nearest neighbors by distance
   - Creates edge_index: [2, N_edges] tensor
5. **Edge Feature Computation:**
   - `edge_distance`: Euclidean distance between connected nodes
   - `edge_vectors`: (target_pos - source_pos) / ||target_pos - source_pos||
6. **PyG Data Object:** Packages everything into `Data(x, edge_index, edge_attr, ...)`

**Pseudo Code for Featurisation:**
```python
# Simplified featurisation pipeline
class ProteinFeaturiser:
    def __call__(self, protein_pdb_path):
        # Step 1: Parse PDB
        atom_tensor = parse_pdb(protein_pdb_path)  # [N_res, 37, 3]
        
        # Step 2: Select representation (C-alpha)
        ca_coords = atom_tensor[:, CA_INDEX, :]    # [N_res, 3]
        
        # Step 3: Compute node features
        node_scalar = torch.cat([
            one_hot_encode(residue_types),         # [N_res, 21]
            compute_dihedrals(atom_tensor)         # [N_res, 6]
        ], dim=-1)                                 # [N_res, 27]
        
        node_vector = compute_orientation(ca_coords)  # [N_res, 2, 3]
        
        # Step 4: Construct edges (knn_30)
        edge_index = knn_graph(ca_coords, k=30)    # [2, N_res * 30]
        
        # Step 5: Compute edge features
        edge_distance = compute_distances(ca_coords, edge_index)  # [N_edges, 1]
        edge_vectors = compute_unit_vectors(ca_coords, edge_index)  # [N_edges, 3]
        
        # Step 6: Create PyG Data object
        return Data(
            x=node_scalar,                         # [N_res, 27]
            orientation=node_vector,               # [N_res, 2, 3]
            edge_index=edge_index,                 # [2, N_edges]
            edge_attr=edge_distance,               # [N_edges, 1]
            edge_vectors=edge_vectors,             # [N_edges, 3]
            pos=ca_coords                          # [N_res, 3]
        )
```

### 3.6 Using Custom Features in Training

**Method 1: Specify Feature Config in Command Line**
```bash
workshop train \
    dataset=ec_random \
    task=multiclass_graph_classification \
    encoder=gear_net \
    features=ec_custom        # Use your custom feature config
```

**Method 2: Override Feature Config Inline**
```bash
workshop train \
    dataset=ec_random \
    task=multiclass_graph_classification \
    encoder=gvp \
    features.edge_types=[knn_30] \
    features.scalar_node_features=[amino_acid_one_hot,dihedrals]
```

**Method 3: Use in DataModule Config**
```yaml
# proteinworkshop/config/dataset/ec_random.yaml
datamodule:
  _target_: proteinworkshop.datasets.ec_proteinshake.ECPSDataModule
  path: ${env.paths.data}/ec_proteinshake/
  transforms: ${transforms}  # Transforms includes featurisation

# Transforms are automatically applied via features config
```

### 3.7 Available Feature Components

**Scalar Node Features:**
- `amino_acid_one_hot`: 21-dim (20 AA + unknown)
- `sequence_positional_encoding`: 16-dim Transformer positional encoding
- `alpha`: Virtual torsion angle (2-dim, sin/cos)
- `kappa`: Virtual bond angle (2-dim, sin/cos)
- `dihedrals`: Backbone φ, ψ, ω angles (6-dim, 3 angles × 2 sin/cos)
- `sidechain_torsions`: χ₁-χ₄ sidechain torsions (8-dim, 4 angles × 2 sin/cos)

**Vector Node Features:**
- `orientation`: Forward/backward direction vectors (2 × 3-dim)

**Edge Types:**
- `knn_X`: X-nearest neighbors (e.g., `knn_10`, `knn_30`)
- `eps_X`: All edges within X Ångströms (e.g., `eps_8`, `eps_10`)
- `sequential`: Sequential edges along chain (i, i+1)

**Scalar Edge Features:**
- `edge_distance`: Euclidean distance (1-dim)
- `edge_type`: Edge type ID (1-dim, for multi-edge graphs)
- `node_features`: Concatenated source/target node features
- `sequence_distance`: |i - j| along sequence (1-dim)

**Vector Edge Features:**
- `edge_vectors`: Directional unit vectors (target - source) / ||target - source|| (3-dim)

---

## 4. Model-Specific Configurations for EC ProteinShake

### 4.1 GearNet (Invariant Model)

**Type:** Invariant (uses only scalar features)

**Compatible with Your Features:** ⚠️ **PARTIALLY**
- GearNet is invariant and **cannot use vector features** (orientation, edge_vectors)
- You need to create an invariant-only feature config for GearNet

**Recommended Feature Config:** `proteinworkshop/config/features/ec_gearnet.yaml`

```yaml
# Feature config for GearNet (invariant only)
_target_: proteinworkshop.features.factory.ProteinFeaturiser
representation: CA

scalar_node_features:
  - amino_acid_one_hot          # 21-dim
  - dihedrals                   # 6-dim (φ, ψ, ω angles)
  - alpha                       # 2-dim (virtual torsion)
  - kappa                       # 2-dim (virtual bond angle)

vector_node_features: []        # ⚠️ Empty - GearNet doesn't use vectors

edge_types:
  - knn_30                      # Your specified k=30

scalar_edge_features:
  - edge_distance               # 1-dim

vector_edge_features: []        # ⚠️ Empty - GearNet doesn't use vectors
```

**Feature Dimensions:**
- **Scalar Node Features:** 21 + 6 + 2 + 2 = **31 dimensions**
- **Scalar Edge Features:** 1 dimension

**Model Config:** `proteinworkshop/config/encoder/gear_net.yaml`

```yaml
_target_: proteinworkshop.models.graph_encoders.gear_net.GearNet
input_dim: 31                   # Auto-computed from feature config
num_relation: 1                 # Single edge type (knn_30)
num_layers: 6                   # Number of GearNet layers
emb_dim: 512                    # Hidden dimension
short_cut: True                 # Residual connections
concat_hidden: True             # Concatenate hidden layers
batch_norm: True                # Batch normalization
num_angle_bin: null             # No angle binning
activation: "relu"              # ReLU activation
pool: sum                       # Global sum pooling
```

**Pseudo Code - GearNet Training:**
```python
# 1. Feature extraction
featuriser = ProteinFeaturiser(
    representation="CA",
    scalar_node_features=["amino_acid_one_hot", "dihedrals", "alpha", "kappa"],
    edge_types=["knn_30"],
    scalar_edge_features=["edge_distance"]
)

# 2. Model initialization
model = GearNet(
    input_dim=31,               # 21 + 6 + 2 + 2
    edge_input_dim=1,           # edge_distance only
    num_relation=1,             # single edge type
    hidden_dim=512,
    num_layers=6,
    num_classes=7               # EC classes 0-6
)

# 3. Training loop
for epoch in range(max_epochs):
    for batch in train_loader:
        # batch.x: [N_nodes, 31] - scalar features only
        # batch.edge_index: [2, N_edges] - knn_30 connectivity
        # batch.edge_attr: [N_edges, 1] - edge distances
        # batch.batch: [N_nodes] - graph assignment
        # batch.graph_label: [batch_size] - EC class labels (0-6)
        
        # Forward pass (invariant to rotations/translations)
        node_embeddings = model.encode(batch.x, batch.edge_index, batch.edge_attr)
        # node_embeddings: [N_nodes, 512]
        
        graph_embeddings = global_sum_pool(node_embeddings, batch.batch)
        # graph_embeddings: [batch_size, 512]
        
        logits = model.decode(graph_embeddings)
        # logits: [batch_size, 7]
        
        # Loss and optimization
        loss = cross_entropy(logits, batch.graph_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # Validation
    val_acc = evaluate(model, val_loader)
    print(f"Epoch {epoch}: Val Accuracy = {val_acc:.4f}")
```

**Training Command:**
```bash
workshop train \
    dataset=ec_random \
    task=multiclass_graph_classification \
    encoder=gear_net \
    features=ec_gearnet \
    dataset.datamodule.batch_size=32 \
    encoder.num_layers=6 \
    encoder.emb_dim=512
```

**Key Point:** GearNet uses relation-aware message passing but only processes scalar features.

---

### 4.2 GVP-GNN (Geometric Vector Perceptrons)

**Type:** Equivariant (requires both scalar and vector features)

**Compatible with Your Features:** ✅ **YES**
- Your specified features (dihedrals + orientation + one-hot) are perfect for GVP
- GVP requires `orientation` (vector node) and `edge_vectors` (vector edge)

**Use Your Custom Feature Config:** `proteinworkshop/config/features/ec_custom.yaml`

```yaml
# Your specified features - perfect for GVP
_target_: proteinworkshop.features.factory.ProteinFeaturiser
representation: CA

scalar_node_features:
  - amino_acid_one_hot          # 21-dim
  - dihedrals                   # 6-dim

vector_node_features:
  - orientation                 # 2 × 3-dim (REQUIRED for GVP)

edge_types:
  - knn_30                      # Your specified k=30

scalar_edge_features:
  - edge_distance               # 1-dim

vector_edge_features:
  - edge_vectors                # 3-dim (REQUIRED for GVP)
```

**Feature Dimensions:**
- **Scalar Node Features:** 21 + 6 = **27 dimensions**
- **Vector Node Features:** 2 vectors × 3 dimensions = **2 vectors**
- **Scalar Edge Features:** 1 dimension
- **Vector Edge Features:** 1 vector × 3 dimensions = **1 vector**

**Model Config:** `proteinworkshop/config/encoder/gvp.yaml`

```yaml
_target_: proteinworkshop.models.graph_encoders.gvp.GVPGNNModel
s_dim: 128                      # Scalar hidden dimension
v_dim: 16                       # Vector hidden dimension
s_dim_edge: 32                  # Edge scalar hidden dimension
v_dim_edge: 1                   # Edge vector hidden dimension
r_max: 10.0                     # Max distance for radial basis
num_bessel: 8                   # Number of radial basis functions
num_polynomial_cutoff: 8        # Polynomial cutoff order
num_layers: 5                   # Number of GVP layers
pool: "sum"                     # Global pooling
residual: True                  # Residual connections
```

**Pseudo Code - GVP Training:**
```python
# 1. Feature extraction (uses your ec_custom config)
featuriser = ProteinFeaturiser(
    representation="CA",
    scalar_node_features=["amino_acid_one_hot", "dihedrals"],
    vector_node_features=["orientation"],
    edge_types=["knn_30"],
    scalar_edge_features=["edge_distance"],
    vector_edge_features=["edge_vectors"]
)

# 2. Model initialization
model = GVP(
    node_s_in=27,               # 21 + 6
    node_v_in=2,                # 2 orientation vectors
    edge_s_in=1,                # edge_distance
    edge_v_in=1,                # edge_vectors
    node_s_dim=128,             # Hidden scalar dimension
    node_v_dim=16,              # Hidden vector dimension
    edge_s_dim=32,
    edge_v_dim=1,
    num_layers=5,
    num_classes=7               # EC classes 0-6
)

# 3. Training loop
for epoch in range(max_epochs):
    for batch in train_loader:
        # Scalar features
        # batch.x: [N_nodes, 27] - one-hot + dihedrals
        # batch.edge_attr: [N_edges, 1] - edge_distance
        
        # Vector features (equivariant to SE(3))
        # batch.orientation: [N_nodes, 2, 3] - direction vectors
        # batch.edge_vectors: [N_edges, 3] - directional edge vectors
        
        # Other
        # batch.edge_index: [2, N_edges] - knn_30 connectivity
        # batch.batch: [N_nodes] - graph assignment
        # batch.graph_label: [batch_size] - EC labels
        
        # Forward pass (equivariant to rotations, invariant to translations)
        node_s, node_v = model.encode(
            node_s=batch.x,              # [N_nodes, 27]
            node_v=batch.orientation,    # [N_nodes, 2, 3]
            edge_index=batch.edge_index,
            edge_s=batch.edge_attr,      # [N_edges, 1]
            edge_v=batch.edge_vectors    # [N_edges, 3]
        )
        # node_s: [N_nodes, 128] - scalar embeddings
        # node_v: [N_nodes, 16, 3] - vector embeddings
        
        # Global pooling (only use scalar part for classification)
        graph_s = global_sum_pool(node_s, batch.batch)
        # graph_s: [batch_size, 128]
        
        logits = model.decode(graph_s)
        # logits: [batch_size, 7]
        
        # Loss and optimization
        loss = cross_entropy(logits, batch.graph_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # Validation
    val_acc = evaluate(model, val_loader)
    print(f"Epoch {epoch}: Val Accuracy = {val_acc:.4f}")
```

**Training Command:**
```bash
workshop train \
    dataset=ec_random \
    task=multiclass_graph_classification \
    encoder=gvp \
    features=ec_custom \
    dataset.datamodule.batch_size=32 \
    encoder.num_layers=5 \
    encoder.s_dim=128 \
    encoder.v_dim=16
```

**Key Point:** GVP uses geometric message passing that preserves SE(3) equivariance using Geometric Algebra.

---

### 4.3 GCPNet (Geometric Confluence Perceptrons)

**Type:** Equivariant (requires both scalar and vector features)

**Compatible with Your Features:** ✅ **YES**
- Your specified features work perfectly with GCPNet
- GCPNet can also handle multiple edge types (unlike GVP)

**Use Your Custom Feature Config:** `proteinworkshop/config/features/ec_custom.yaml`

```yaml
# Same as GVP - your specified features
_target_: proteinworkshop.features.factory.ProteinFeaturiser
representation: CA

scalar_node_features:
  - amino_acid_one_hot          # 21-dim
  - dihedrals                   # 6-dim

vector_node_features:
  - orientation                 # 2 × 3-dim (REQUIRED for GCPNet)

edge_types:
  - knn_30                      # Your specified k=30

scalar_edge_features:
  - edge_distance               # 1-dim

vector_edge_features:
  - edge_vectors                # 3-dim (REQUIRED for GCPNet)
```

**Alternative: Multi-Edge Config for GCPNet**

```yaml
# proteinworkshop/config/features/ec_gcpnet_multi.yaml
# GCPNet supports multiple edge types
_target_: proteinworkshop.features.factory.ProteinFeaturiser
representation: CA

scalar_node_features:
  - amino_acid_one_hot
  - dihedrals

vector_node_features:
  - orientation

edge_types:
  - knn_30                      # Spatial neighbors
  - sequential                  # Sequential edges
  # - eps_10                    # Optional: distance threshold

scalar_edge_features:
  - edge_distance
  - edge_type                   # Edge type ID (0, 1, ...)
  - sequence_distance           # |i - j| along sequence

vector_edge_features:
  - edge_vectors
```

**Model Config:** `proteinworkshop/config/encoder/gcpnet.yaml`

```yaml
_target_: proteinworkshop.models.graph_encoders.gcpnet.GCPNetModel
features:
  vector_node_features: ["orientation"]    # Auto-injected from feature config
  vector_edge_features: ["edge_vectors"]   # Auto-injected from feature config
num_layers: 6                   # Number of GCP layers
emb_dim: 128                    # Hidden scalar dimension
node_s_emb_dim: 128             # Node scalar embedding
node_v_emb_dim: 16              # Node vector embedding
edge_s_emb_dim: 32              # Edge scalar embedding
edge_v_emb_dim: 4               # Edge vector embedding
r_max: 10.0                     # Max distance for radial basis
num_rbf: 8                      # Number of radial basis functions
activation: silu                # SiLU (Swish) activation
pool: sum                       # Global pooling
```

**Pseudo Code - GCPNet Training:**
```python
# 1. Feature extraction
featuriser = ProteinFeaturiser(
    representation="CA",
    scalar_node_features=["amino_acid_one_hot", "dihedrals"],
    vector_node_features=["orientation"],
    edge_types=["knn_30"],
    scalar_edge_features=["edge_distance"],
    vector_edge_features=["edge_vectors"]
)

# 2. Model initialization
model = GCPNet(
    node_s_in=27,               # 21 + 6
    node_v_in=2,                # 2 orientation vectors
    edge_s_in=1,                # edge_distance
    edge_v_in=1,                # edge_vectors
    node_s_emb_dim=128,         # Hidden scalar dimension
    node_v_emb_dim=16,          # Hidden vector dimension
    edge_s_emb_dim=32,
    edge_v_emb_dim=4,
    num_layers=6,
    num_classes=7               # EC classes 0-6
)

# 3. Training loop
for epoch in range(max_epochs):
    for batch in train_loader:
        # Scalar features
        # batch.x: [N_nodes, 27] - one-hot + dihedrals
        # batch.edge_attr: [N_edges, 1] - edge_distance
        
        # Vector features (equivariant to SE(3))
        # batch.orientation: [N_nodes, 2, 3] - direction vectors
        # batch.edge_vectors: [N_edges, 3] - directional edge vectors
        
        # Other
        # batch.edge_index: [2, N_edges] - knn_30 connectivity
        # batch.batch: [N_nodes] - graph assignment
        # batch.graph_label: [batch_size] - EC labels
        
        # Forward pass (SE(3) equivariant using geometric product)
        node_s, node_v = model.encode(
            node_s=batch.x,              # [N_nodes, 27]
            node_v=batch.orientation,    # [N_nodes, 2, 3]
            edge_index=batch.edge_index,
            edge_s=batch.edge_attr,      # [N_edges, 1]
            edge_v=batch.edge_vectors,   # [N_edges, 3]
            pos=batch.pos                # [N_nodes, 3] - coordinates
        )
        # node_s: [N_nodes, 128] - scalar embeddings
        # node_v: [N_nodes, 16, 3] - vector embeddings
        
        # Global pooling
        graph_s = global_sum_pool(node_s, batch.batch)
        # graph_s: [batch_size, 128]
        
        logits = model.decode(graph_s)
        # logits: [batch_size, 7]
        
        # Loss and optimization
        loss = cross_entropy(logits, batch.graph_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # Validation
    val_acc = evaluate(model, val_loader)
    print(f"Epoch {epoch}: Val Accuracy = {val_acc:.4f}")
```

**Training Command:**
```bash
workshop train \
    dataset=ec_random \
    task=multiclass_graph_classification \
    encoder=gcpnet \
    features=ec_custom \
    dataset.datamodule.batch_size=16 \
    encoder.num_layers=6 \
    encoder.node_s_emb_dim=128 \
    encoder.node_v_emb_dim=16
```

**Key Point:** GCPNet uses geometric product from Clifford algebra for SE(3) equivariance and supports multiple edge types.

---

### 4.4 Model Comparison Summary

| Model | Type | Vector Features | Your Features | Batch Size | Comments |
|-------|------|-----------------|---------------|------------|----------|
| **GearNet** | Invariant | ❌ Not used | ⚠️ Need invariant config | 32-64 | Fast, uses only scalars |
| **GVP-GNN** | Equivariant | ✅ Required | ✅ Compatible | 32 | Geometric Algebra, single edge type |
| **GCPNet** | Equivariant | ✅ Required | ✅ Compatible | 16-32 | Clifford algebra, multi-edge support |

**Feature Requirements:**

```
GearNet:
├─ Node: one-hot (21) + dihedrals (6) + alpha (2) + kappa (2) = 31
├─ Edge: distance (1)
└─ Total: 31 scalar node + 1 scalar edge

GVP-GNN:
├─ Node Scalar: one-hot (21) + dihedrals (6) = 27
├─ Node Vector: orientation (2 × 3)
├─ Edge Scalar: distance (1)
├─ Edge Vector: edge_vectors (1 × 3)
└─ Total: 27 scalar + 2 vector (node), 1 scalar + 1 vector (edge)

GCPNet:
├─ Node Scalar: one-hot (21) + dihedrals (6) = 27
├─ Node Vector: orientation (2 × 3)
├─ Edge Scalar: distance (1) [+ optional: type, seq_distance]
├─ Edge Vector: edge_vectors (1 × 3)
└─ Total: 27 scalar + 2 vector (node), 1+ scalar + 1 vector (edge)
```

### 4.5 Framework's Built-in Feature Recommendations

ProteinWorkshop provides several pre-configured feature sets optimized for different model types. These are found in `proteinworkshop/config/features/`:

**Built-in Feature Configs:**

| Feature Config | Type | Description | Best For |
|----------------|------|-------------|----------|
| `all_invariant_ca` | Invariant | Full scalar features: one-hot + dihedrals + sidechain torsions + alpha + kappa | GearNet, SchNet, DimeNet++ |
| `all_equivariant_ca` | Equivariant | Minimal: one-hot + orientation + edge_vectors | GVP, GCPNet, EGNN |
| `ca_angles` | Invariant | one-hot + positional encoding + alpha + kappa | General purpose invariant |
| `ca_base` | Invariant | one-hot only (minimal) | Baseline experiments |
| `ca_seq` | Invariant | one-hot + positional encoding | Sequence-aware models |
| `ca_bb` | Invariant | one-hot + backbone angles | Medium complexity |
| `ca_sc` | Invariant | one-hot + sidechain torsions | Full torsional info |

**Framework Recommendations by Model:**

**1. GearNet (Invariant):**
```yaml
# Option A: Use built-in all_invariant_ca (RECOMMENDED BY FRAMEWORK)
features: all_invariant_ca
# Includes: one-hot + dihedrals + sidechain_torsions + alpha + kappa
# Edge types: knn_10 + eps_8
# Edge features: distance + type + node_features + sequence_distance

# Option B: Use simpler ca_angles
features: ca_angles
# Includes: one-hot + positional_encoding + alpha + kappa
# Edge types: knn_16
# Edge features: distance
```

**2. GVP-GNN (Equivariant):**
```yaml
# Use built-in all_equivariant_ca (RECOMMENDED BY FRAMEWORK)
features: all_equivariant_ca
# Node scalar: one-hot
# Node vector: orientation
# Edge types: knn_10 + eps_8
# Edge scalar: distance + type + node_features + sequence_distance
# Edge vector: edge_vectors
```

**3. GCPNet (Equivariant):**
```yaml
# Use built-in all_equivariant_ca (RECOMMENDED BY FRAMEWORK)
features: all_equivariant_ca
# Same as GVP - minimal equivariant features
# GCPNet can also handle the richer feature set
```

**Hyperparameter Configs:** The framework also provides model+feature pairings in `proteinworkshop/config/hparams/`:
- `gear_net_edge_ca_angles.yaml` → lr=0.0001, dropout=0.5
- `gear_net_edge_ca_sc.yaml` → lr=0.0001, dropout=0.5
- `gcpnet_ca_angles.yaml` → lr=0.001, dropout=0.3
- `gcpnet_ca_sc.yaml` → lr=0.001, dropout=0.3
- And many more...

### 4.6 Comparing Your Custom Config vs Framework Recommendations

**Your Specification (EC Custom):**
```yaml
# Your ec_custom.yaml (for GVP/GCPNet)
node_scalar: one-hot (21) + dihedrals (6) = 27
node_vector: orientation (2 × 3) = 6
edge_scalar: distance (1)
edge_vector: edge_vectors (3)
edge_type: knn_30 (single type)
```

**Framework Recommendation (`all_equivariant_ca`):**
```yaml
# proteinworkshop/config/features/all_equivariant_ca.yaml
node_scalar: one-hot (21)
node_vector: orientation (2 × 3) = 6
edge_scalar: distance (1) + edge_type (1) + node_features (42) + sequence_distance (1)
edge_vector: edge_vectors (3)
edge_types: knn_10 + eps_8 (TWO types - richer connectivity)
```

**Key Differences:**

| Aspect | Your Config | Framework Config |
|--------|-------------|------------------|
| **Node Scalar Features** | one-hot + dihedrals (27-dim) | one-hot only (21-dim) |
| **Edge Scalar Features** | distance only (1-dim) | distance + type + node_features + seq_distance (45-dim) |
| **Edge Types** | knn_30 (single) | knn_10 + eps_8 (dual) |
| **k Value** | 30-NN | 10-NN |

**Analysis:**

✅ **Your Config Advantages:**
- **More node features:** Adding dihedrals gives 6 extra dimensions of structural information
- **Denser graph:** knn_30 vs knn_10 means more connectivity
- **Simpler:** Single edge type is easier to train

⚠️ **Framework Config Advantages:**
- **Multi-edge types:** knn + eps captures both nearest neighbors AND distance threshold
- **Richer edge features:** node_features concatenates source/target node info at edges
- **Sequence awareness:** sequence_distance helps with sequential dependencies
- **Proven:** These configs have been tested across many benchmarks

### 4.7 Recommended Configurations for EC Dataset

Based on framework conventions and your requirements:

**Configuration 1: Your Custom (As Specified)**
```bash
# Use your knn_30 + dihedrals specification
workshop train \
    dataset=ec_random \
    task=multiclass_graph_classification \
    encoder=gear_net \
    features=ec_gearnet \
    dataset.datamodule.batch_size=64
```

**Configuration 2: Framework Standard (GearNet)**
```bash
# Use framework's all_invariant_ca
workshop train \
    dataset=ec_random \
    task=multiclass_graph_classification \
    encoder=gear_net \
    features=all_invariant_ca \
    dataset.datamodule.batch_size=64 \
    hparams=gear_net_edge_ca_angles  # Use tuned hyperparameters
```

**Configuration 3: Your Custom Equivariant (GVP)**
```bash
# Your specification with knn_30
workshop train \
    dataset=ec_random \
    task=multiclass_graph_classification \
    encoder=gvp \
    features=ec_custom \
    dataset.datamodule.batch_size=32
```

**Configuration 4: Framework Standard (GVP)**
```bash
# Framework's all_equivariant_ca with dual edge types
workshop train \
    dataset=ec_random \
    task=multiclass_graph_classification \
    encoder=gvp \
    features=all_equivariant_ca \
    dataset.datamodule.batch_size=32
```

**Configuration 5: Framework Standard (GCPNet)**
```bash
# Framework's recommendation
workshop train \
    dataset=ec_random \
    task=multiclass_graph_classification \
    encoder=gcpnet \
    features=all_equivariant_ca \
    dataset.datamodule.batch_size=16 \
    hparams=gcpnet_ca_sc  # Use tuned hyperparameters
```

**Recommendation:** Try BOTH your custom config AND the framework's built-in configs to compare:
- Your config tests your hypothesis about dihedrals + knn_30
- Framework config provides a proven baseline
- Compare results to see which works better for EC classification

### 3.6 Using Transforms in Your Dataset

Transforms are automatically applied via the config:

```python
# In your datamodule __init__
if transforms is not None:
    self.transform = self.compose_transforms(
        omegaconf.OmegaConf.to_container(transforms, resolve=True)
    )
```

Then pass to `ProteinDataset`:
```python
return ProteinDataset(
    ...
    transform=self.transform,  # Applied to each loaded structure
    ...
)
```

The transform pipeline is automatically applied when loading data!

---

## 4. Model-Specific Feature Requirements

### 4.1 GVP-GNN (Geometric Vector Perceptrons)

**Type:** Equivariant (vector-valued features)

**Required Features:**
```yaml
# Use features=all_equivariant_ca or create custom config
representation: CA
scalar_node_features:
  - amino_acid_one_hot
vector_node_features:
  - orientation              # REQUIRED for GVP
edge_types:
  - knn_10
scalar_edge_features:
  - edge_distance
vector_edge_features:
  - edge_vectors             # REQUIRED for GVP
```

**Config:** `proteinworkshop/config/encoder/gvp.yaml`
```yaml
_target_: proteinworkshop.models.graph_encoders.gvp.GVPGNNModel
s_dim: 128          # Scalar dimension
v_dim: 16           # Vector dimension
s_dim_edge: 32      # Edge scalar dimension
v_dim_edge: 1       # Edge vector dimension
r_max: 10.0         # Max distance for basis functions
num_bessel: 8
num_polynomial_cutoff: 8
num_layers: 5
pool: "sum"
residual: True
```

**Key Point:** GVP **requires** vector-valued features (`orientation` and `edge_vectors`)

### 4.2 GearNet

**Type:** Invariant (scalar features only)

**Required Features:**
```yaml
# Use features=ca or features=ca_angles
representation: CA
scalar_node_features:
  - amino_acid_one_hot
  # Optional: alpha, kappa, dihedrals
vector_node_features: []      # Not used
edge_types:
  - knn_16                    # Or multiple types
scalar_edge_features:
  - edge_distance
vector_edge_features: []      # Not used
```

**Config:** `proteinworkshop/config/encoder/gear_net.yaml`
```yaml
_target_: proteinworkshop.models.graph_encoders.gear_net.GearNet
input_dim: ${resolve_feature_config_dim:${features},scalar_node_features,${task},true}
num_relation: ${resolve_num_edge_types:${features}}  # Auto-computed
num_layers: 6
emb_dim: 512
short_cut: True
concat_hidden: True
batch_norm: True
num_angle_bin: null
activation: "relu"
pool: sum
```

**Key Point:** GearNet can handle multiple edge types (uses relation-aware message passing)

### 4.3 GCPNet (Geometric Confluence Perceptrons)

**Type:** Equivariant (vector-valued features)

**Required Features:**
```yaml
# Use features=all_equivariant_ca
representation: CA
scalar_node_features:
  - amino_acid_one_hot
vector_node_features:
  - orientation              # REQUIRED
edge_types:
  - knn_10
  - eps_8                    # Multiple types supported
scalar_edge_features:
  - edge_distance
  - edge_type
  - node_features
  - sequence_distance
vector_edge_features:
  - edge_vectors             # REQUIRED
```

**Config:** `proteinworkshop/config/encoder/gcpnet.yaml`
```yaml
_target_: proteinworkshop.models.graph_encoders.gcpnet.GCPNetModel
features:
  vector_node_features: ["orientation"]    # Auto-injected
  vector_edge_features: ["edge_vectors"]   # Auto-injected
num_layers: 6
emb_dim: 128
node_s_emb_dim: 128
node_v_emb_dim: 16
edge_s_emb_dim: 32
edge_v_emb_dim: 4
r_max: 10.0
num_rbf: 8
activation: silu
pool: sum
```

**Key Point:** GCPNet **requires** equivariant features and auto-injects them via config validation

### 4.4 Summary Table

| Model | Type | Scalar Nodes | Vector Nodes | Scalar Edges | Vector Edges | Edge Types |
|-------|------|--------------|--------------|--------------|--------------|------------|
| **GVP-GNN** | Equivariant | ✓ | ✓ Required | ✓ | ✓ Required | Single |
| **GearNet** | Invariant | ✓ | ✗ | ✓ | ✗ | Multiple ✓ |
| **GCPNet** | Equivariant | ✓ | ✓ Required | ✓ | ✓ Required | Multiple ✓ |

### 4.5 Recommended Feature Configs

For your baseline experiments:

**For GearNet (invariant):**
```bash
workshop train dataset=my_dataset encoder=gear_net features=ca_angles task=multiclass_graph_classification
```

**For GVP-GNN (equivariant):**
```bash
workshop train dataset=my_dataset encoder=gvp features=all_equivariant_ca task=multiclass_graph_classification
```

**For GCPNet (equivariant):**
```bash
workshop train dataset=my_dataset encoder=gcpnet features=all_equivariant_ca task=multiclass_graph_classification
```

---

## 5. Complete Training Workflow for EC ProteinShake

### 5.1 Implementation Status

✅ **COMPLETED:**
- ✅ Dataset created via `create_raw_data.py`
- ✅ DataModule implemented: `proteinworkshop/datasets/ec_proteinshake.py`
- ✅ Config files created: `ec_random.yaml`, `ec_structure.yaml`
- ✅ DataModule tested via `test_ec_datamodule.py`

📝 **TODO (Before Training):**
- 📝 Create `proteinworkshop/config/features/ec_custom.yaml` (for GVP/GCPNet)
- 📝 Create `proteinworkshop/config/features/ec_gearnet.yaml` (for GearNet)

### 5.2 Directory Structure

```
ProteinWorkshop/
├── proteinworkshop/
│   ├── datasets/
│   │   └── ec_proteinshake.py          # ✅ DataModule
│   └── config/
│       ├── dataset/
│       │   ├── ec_random.yaml          # ✅ Random split config
│       │   └── ec_structure.yaml       # ✅ Structure split config
│       ├── features/
│       │   ├── ec_custom.yaml          # 📝 TODO: Equivariant features
│       │   └── ec_gearnet.yaml         # 📝 TODO: Invariant features
│       └── task/
│           └── multiclass_graph_classification.yaml  # ✅ Built-in (use this)
├── data/
│   └── ec_proteinshake/
│       ├── pdb/                        # ✅ PDB files (chain A only)
│       ├── random/                     # ✅ Random split CSVs
│       ├── structure/                  # ✅ Structure split CSVs
│       └── labels.csv                  # ✅ Shared labels
└── test_ec_datamodule.py               # ✅ Test script

```

### 5.3 Training Commands (All 6 Configurations)

**Configuration 1: GearNet + Random Split**
```bash
workshop train \
    dataset=ec_random \
    task=multiclass_graph_classification \
    encoder=gear_net \
    features=ec_gearnet \
    dataset.datamodule.batch_size=64 \
    encoder.num_layers=6 \
    encoder.emb_dim=512 \
    trainer.max_epochs=100
```

**Configuration 2: GearNet + Structure Split**
```bash
workshop train \
    dataset=ec_structure \
    task=multiclass_graph_classification \
    encoder=gear_net \
    features=ec_gearnet \
    dataset.datamodule.batch_size=64 \
    encoder.num_layers=6 \
    encoder.emb_dim=512 \
    trainer.max_epochs=100
```

**Configuration 3: GVP + Random Split**
```bash
workshop train \
    dataset=ec_random \
    task=multiclass_graph_classification \
    encoder=gvp \
    features=ec_custom \
    dataset.datamodule.batch_size=32 \
    encoder.num_layers=5 \
    encoder.s_dim=128 \
    encoder.v_dim=16 \
    trainer.max_epochs=100
```

**Configuration 4: GVP + Structure Split**
```bash
workshop train \
    dataset=ec_structure \
    task=multiclass_graph_classification \
    encoder=gvp \
    features=ec_custom \
    dataset.datamodule.batch_size=32 \
    encoder.num_layers=5 \
    encoder.s_dim=128 \
    encoder.v_dim=16 \
    trainer.max_epochs=100
```

**Configuration 5: GCPNet + Random Split**
```bash
workshop train \
    dataset=ec_random \
    task=multiclass_graph_classification \
    encoder=gcpnet \
    features=ec_custom \
    dataset.datamodule.batch_size=16 \
    encoder.num_layers=6 \
    encoder.node_s_emb_dim=128 \
    encoder.node_v_emb_dim=16 \
    trainer.max_epochs=100
```

**Configuration 6: GCPNet + Structure Split**
```bash
workshop train \
    dataset=ec_structure \
    task=multiclass_graph_classification \
    encoder=gcpnet \
    features=ec_custom \
    dataset.datamodule.batch_size=16 \
    encoder.num_layers=6 \
    encoder.node_s_emb_dim=128 \
    encoder.node_v_emb_dim=16 \
    trainer.max_epochs=100
```

### 5.4 Hyperparameter Exploration (Pseudo Commands)

**Explore Edge Connectivity:**
```bash
# Sparse graph (k=10)
workshop train ... features.edge_types=[knn_10]

# Medium graph (k=20)
workshop train ... features.edge_types=[knn_20]

# Your configuration (k=30)
workshop train ... features.edge_types=[knn_30]

# Dense graph (k=50)
workshop train ... features.edge_types=[knn_50]
```

**Explore Hidden Dimensions:**
```bash
# GearNet variants
workshop train ... encoder=gear_net encoder.emb_dim=256   # Small
workshop train ... encoder=gear_net encoder.emb_dim=512   # Default
workshop train ... encoder=gear_net encoder.emb_dim=1024  # Large

# GVP variants
workshop train ... encoder=gvp encoder.s_dim=64 encoder.v_dim=8      # Small
workshop train ... encoder=gvp encoder.s_dim=128 encoder.v_dim=16    # Default
workshop train ... encoder=gvp encoder.s_dim=256 encoder.v_dim=32    # Large

# GCPNet variants
workshop train ... encoder=gcpnet encoder.node_s_emb_dim=64 encoder.node_v_emb_dim=8
workshop train ... encoder=gcpnet encoder.node_s_emb_dim=128 encoder.node_v_emb_dim=16
workshop train ... encoder=gcpnet encoder.node_s_emb_dim=256 encoder.node_v_emb_dim=32
```

**Explore Network Depth:**
```bash
workshop train ... encoder.num_layers=3     # Shallow
workshop train ... encoder.num_layers=6     # Medium (default)
workshop train ... encoder.num_layers=10    # Deep
```

### 5.5 Pseudo Code - Complete Training Pipeline

```python
# ============================================================================
# Complete Training Pipeline Pseudo Code for EC ProteinShake
# ============================================================================

# ----------------------------------------------------------------------------
# STEP 1: Initialize Hydra configuration
# ----------------------------------------------------------------------------
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="proteinworkshop/config", config_name="train")
def train_ec_classification(cfg: DictConfig):
    """
    Train EC ProteinShake classification model
    
    Usage:
        workshop train dataset=ec_random encoder=gear_net features=ec_gearnet
    """
    
    # ========================================================================
    # STEP 2: Setup DataModule
    # ========================================================================
    datamodule = hydra.utils.instantiate(cfg.dataset.datamodule)
    # Instantiates: ECPSDataModule(
    #     path="data/ec_proteinshake",
    #     split_type="random",  # or "structure"
    #     batch_size=32,
    #     pdb_dir="data/ec_proteinshake/pdb",
    #     format="pdb",
    #     transforms=[ProteinFeaturiser(...)]
    # )
    
    datamodule.setup("fit")
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    
    # Each batch structure:
    # For GearNet (invariant):
    #   batch.x: [N_nodes, 31] - scalar node features
    #   batch.edge_index: [2, N_edges] - knn_30 connectivity
    #   batch.edge_attr: [N_edges, 1] - edge distances
    #   batch.batch: [N_nodes] - graph assignment
    #   batch.graph_label: [batch_size] - EC class labels (0-6)
    #
    # For GVP/GCPNet (equivariant):
    #   batch.x: [N_nodes, 27] - scalar node features
    #   batch.orientation: [N_nodes, 2, 3] - vector node features
    #   batch.edge_index: [2, N_edges] - knn_30 connectivity
    #   batch.edge_attr: [N_edges, 1] - scalar edge features
    #   batch.edge_vectors: [N_edges, 3] - vector edge features
    #   batch.batch: [N_nodes] - graph assignment
    #   batch.graph_label: [batch_size] - EC labels
    
    # ========================================================================
    # STEP 3: Setup Model
    # ========================================================================
    encoder = hydra.utils.instantiate(cfg.encoder)
    # Option 1: GearNet (invariant)
    #   encoder = GearNet(
    #       input_dim=31,
    #       edge_input_dim=1,
    #       hidden_dim=512,
    #       num_layers=6,
    #       num_relation=1
    #   )
    #
    # Option 2: GVP (equivariant)
    #   encoder = GVP(
    #       node_s_in=27, node_v_in=2,
    #       edge_s_in=1, edge_v_in=1,
    #       s_dim=128, v_dim=16,
    #       num_layers=5
    #   )
    #
    # Option 3: GCPNet (equivariant)
    #   encoder = GCPNet(
    #       node_s_in=27, node_v_in=2,
    #       edge_s_in=1, edge_v_in=1,
    #       node_s_emb_dim=128, node_v_emb_dim=16,
    #       num_layers=6
    #   )
    
    decoder = hydra.utils.instantiate(cfg.decoder)
    # decoder = MLPDecoder(input_dim=512, output_dim=7)  # 7 EC classes
    
    model = ProteinWorkshopModel(
        encoder=encoder,
        decoder=decoder,
        task="classification",
        num_classes=7
    )
    
    # ========================================================================
    # STEP 4: Setup Training Components
    # ========================================================================
    optimizer = hydra.utils.instantiate(cfg.optimiser, params=model.parameters())
    # optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)
    # scheduler = CosineAnnealingLR(optimizer, T_max=100)
    
    loss_fn = CrossEntropyLoss()
    
    # ========================================================================
    # STEP 5: Setup Callbacks
    # ========================================================================
    early_stopping = EarlyStopping(
        monitor="val/graph_label/accuracy",
        mode="max",
        patience=50
    )
    
    model_checkpoint = ModelCheckpoint(
        monitor="val/graph_label/accuracy",
        mode="max",
        save_top_k=1,
        filename="best_ec_model_epoch{epoch}_acc{val_acc:.4f}"
    )
    
    # ========================================================================
    # STEP 6: Training Loop
    # ========================================================================
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(cfg.trainer.max_epochs):
        
        # ====================================================================
        # Training Phase
        # ====================================================================
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Forward pass - model handles both invariant and equivariant
            if cfg.encoder._target_.endswith("GearNet"):
                # Invariant forward pass
                logits = model(
                    x=batch.x,                    # [N_nodes, 31]
                    edge_index=batch.edge_index,  # [2, N_edges]
                    edge_attr=batch.edge_attr,    # [N_edges, 1]
                    batch=batch.batch             # [N_nodes]
                )
            else:
                # Equivariant forward pass (GVP/GCPNet)
                logits = model(
                    x=batch.x,                    # [N_nodes, 27]
                    orientation=batch.orientation,  # [N_nodes, 2, 3]
                    edge_index=batch.edge_index,  # [2, N_edges]
                    edge_attr=batch.edge_attr,    # [N_edges, 1]
                    edge_vectors=batch.edge_vectors,  # [N_edges, 3]
                    batch=batch.batch             # [N_nodes]
                )
            # logits: [batch_size, 7]
            
            # Compute loss
            loss = loss_fn(logits, batch.graph_label)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            pred = logits.argmax(dim=-1)
            train_correct += (pred == batch.graph_label).sum().item()
            train_total += batch.graph_label.size(0)
        
        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # ====================================================================
        # Validation Phase
        # ====================================================================
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Forward pass
                if cfg.encoder._target_.endswith("GearNet"):
                    logits = model(
                        x=batch.x,
                        edge_index=batch.edge_index,
                        edge_attr=batch.edge_attr,
                        batch=batch.batch
                    )
                else:
                    logits = model(
                        x=batch.x,
                        orientation=batch.orientation,
                        edge_index=batch.edge_index,
                        edge_attr=batch.edge_attr,
                        edge_vectors=batch.edge_vectors,
                        batch=batch.batch
                    )
                
                # Compute loss
                loss = loss_fn(logits, batch.graph_label)
                
                # Track metrics
                val_loss += loss.item()
                pred = logits.argmax(dim=-1)
                val_correct += (pred == batch.graph_label).sum().item()
                val_total += batch.graph_label.size(0)
        
        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # ====================================================================
        # Logging and Callbacks
        # ====================================================================
        print(f"Epoch {epoch+1}/{cfg.trainer.max_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Model checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_checkpoint.save(model, val_acc, epoch)
            patience_counter = 0
            print(f"  ✓ New best model! Val Acc: {val_acc:.4f}")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{early_stopping.patience}")
        
        # Early stopping check
        if patience_counter >= early_stopping.patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
        
        # Step scheduler
        scheduler.step()
    
    # ========================================================================
    # STEP 7: Test Evaluation
    # ========================================================================
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()
    
    # Load best model
    model.load_state_dict(torch.load(model_checkpoint.best_path))
    model.eval()
    
    test_correct = 0
    test_total = 0
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Forward pass
            if cfg.encoder._target_.endswith("GearNet"):
                logits = model(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    batch=batch.batch
                )
            else:
                logits = model(
                    x=batch.x,
                    orientation=batch.orientation,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    edge_vectors=batch.edge_vectors,
                    batch=batch.batch
                )
            
            pred = logits.argmax(dim=-1)
            test_correct += (pred == batch.graph_label).sum().item()
            test_total += batch.graph_label.size(0)
            
            test_preds.extend(pred.cpu().numpy())
            test_labels.extend(batch.graph_label.cpu().numpy())
    
    test_acc = test_correct / test_total
    
    print(f"\n{'='*70}")
    print(f"FINAL TEST ACCURACY: {test_acc:.4f}")
    print(f"{'='*70}")
    
    # Compute per-class metrics
    from sklearn.metrics import classification_report
    print("\nPer-Class Performance:")
    print(classification_report(
        test_labels, test_preds,
        target_names=[f"EC{i+1}" for i in range(7)],
        digits=4
    ))
    
    return test_acc

# Run training
if __name__ == "__main__":
    train_ec_classification()
```

### 5.6 Expected Training Output

```
Epoch 1/100
  Train Loss: 1.8452, Train Acc: 0.3245
  Val Loss: 1.7234, Val Acc: 0.3521
  ✓ New best model! Val Acc: 0.3521

Epoch 2/100
  Train Loss: 1.6721, Train Acc: 0.3876
  Val Loss: 1.6012, Val Acc: 0.4102
  ✓ New best model! Val Acc: 0.4102

...

Epoch 45/100
  Train Loss: 0.4823, Train Acc: 0.8421
  Val Loss: 0.5347, Val Acc: 0.8156
  ✓ New best model! Val Acc: 0.8156

Epoch 46/100
  Train Loss: 0.4712, Train Acc: 0.8467
  Val Loss: 0.5389, Val Acc: 0.8134
  Patience: 1/50

...

======================================================================
FINAL TEST ACCURACY: 0.8045
======================================================================

Per-Class Performance:
              precision    recall  f1-score   support

         EC1     0.8234    0.8156    0.8195       245
         EC2     0.7845    0.8013    0.7928       198
         EC3     0.8167    0.7921    0.8042       312
         EC4     0.8089    0.8245    0.8166       267
         EC5     0.7923    0.7756    0.7839       189
         EC6     0.8301    0.8456    0.8378       294
         EC7     0.8012    0.8178    0.8094       223

    accuracy                        0.8045      1728
   macro avg     0.8082    0.8104    0.8092      1728
weighted avg     0.8048    0.8045    0.8046      1728
```

---

## 6. Troubleshooting

# Test random split
print("=" * 70)
print("Testing Random Split")
print("=" * 70)

cfg_random = omegaconf.OmegaConf.load(
    constants.SRC_PATH / "config" / "dataset" / "ec_random.yaml"
)
cfg_random.datamodule.path = "data/ec_proteinshake"
cfg_random.datamodule.pdb_dir = "data/ec_proteinshake/pdb"
cfg_random.datamodule.transforms = []

# Instantiate datamodule for random split
ds_random = hydra.utils.instantiate(cfg_random)
datamodule_random = ds_random["datamodule"]

# Test loading
datamodule_random.setup("fit")
train_dl = datamodule_random.train_dataloader()

# Check first batch
for batch in train_dl:
    print(f"Batch size: {batch.num_graphs}")
    print(f"Nodes per graph: {batch.x.shape}")
    print(f"Edges: {batch.edge_index.shape}")
    print(f"Labels: {batch.graph_label}")
    print(f"Label range: {batch.graph_label.min()} - {batch.graph_label.max()}")
    print(f"Unique labels in batch: {batch.graph_label.unique()}")
    break

# Test structure split
print("\n" + "=" * 70)
print("Testing Structure Split")
print("=" * 70)

cfg_structure = omegaconf.OmegaConf.load(
    constants.SRC_PATH / "config" / "dataset" / "ec_structure.yaml"
)
cfg_structure.datamodule.path = "data/ec_proteinshake"
cfg_structure.datamodule.pdb_dir = "data/ec_proteinshake/pdb"
cfg_structure.datamodule.transforms = []

# Instantiate datamodule for structure split
ds_structure = hydra.utils.instantiate(cfg_structure)
datamodule_structure = ds_structure["datamodule"]

# Test loading
datamodule_structure.setup("fit")
train_dl_struct = datamodule_structure.train_dataloader()

for batch in train_dl_struct:
    print(f"Batch size: {batch.num_graphs}")
    print(f"Nodes per graph: {batch.x.shape}")
    print(f"Edges: {batch.edge_index.shape}")
    print(f"Labels: {batch.graph_label}")
    break

# Verify all splits load correctly for random split
print("\n=== Random Split - Validation ===")
val_dl = datamodule_random.val_dataloader()
for batch in val_dl:
    print(f"Val batch size: {batch.num_graphs}")
    print(f"Val labels: {batch.graph_label}")
    break

print("\n=== Random Split - Test ===")
datamodule_random.setup("test")
test_dl = datamodule_random.test_dataloader()
for batch in test_dl:
    print(f"Test batch size: {batch.num_graphs}")
    print(f"Test labels: {batch.graph_label}")
    break

print("\n✓ Both split types loaded successfully!")
print("✓ Same PDB files used for both splits")
print("✓ Different train/val/test assignments verified")
```

**Expected Output:**
```
======================================================================
Testing Random Split
======================================================================
Batch size: 32
Nodes per graph: torch.Size([X, Y])  # X = total nodes, Y = feature dim
Edges: torch.Size([2, Z])            # Z = total edges
Labels: tensor([0, 5, 2, ...])       # Graph-level labels
Label range: 0 - 6                   # EC classes 0-6
Unique labels in batch: tensor([0, 1, 2, 5, 6])

======================================================================
Testing Structure Split
======================================================================
Batch size: 32
Nodes per graph: torch.Size([X2, Y])
Edges: torch.Size([2, Z2])
Labels: tensor([1, 3, 0, ...])

=== Random Split - Validation ===
Val batch size: 32
Val labels: tensor([2, 4, 1, ...])

=== Random Split - Test ===
Test batch size: 32
Test labels: tensor([3, 0, 5, ...])

✓ Both split types loaded successfully!
✓ Same PDB files used for both splits
✓ Different train/val/test assignments verified
```

---

## 6. Troubleshooting

### Common Issues for Multi-Split Datasets

**Issue 1: PDB files not found**
```
FileNotFoundError: PDB directory not found: /path/to/ec_proteinshake/pdb/
```
**Solution:** Ensure preprocessed PDB files are in the correct directory. Check that:
- `pdb_dir` path in config is correct (should be `{dataset_path}/pdb/`)
- PDB files have correct naming (e.g., `1abc.pdb`)
- Files are readable

**Issue 2: Split directory not found**
```
FileNotFoundError: Split directory not found: /path/to/ec_proteinshake/random/
```
**Solution:**
- Ensure you ran `create_raw_data.py` to generate both split directories
- Verify `split_type` parameter in config matches existing directories
- Check that split CSVs exist in the correct subdirectory

**Issue 3: All chains set to 'A' but getting chain mismatch errors**
```
Solution: Verify that your PDB files actually contain chain A in their structure
```

**Issue 4: Missing labels**
```
WARNING: Missing labels for 10 structures: ['1abc', '2xyz', ...]
```
**Solution:** 
- Check that all PDB IDs in split files have corresponding entries in `labels.csv`
- Verify label file format matches your split file format
- Ensure no typos in PDB IDs
- Remember: labels.csv is shared across all splits

**Issue 5: Memory errors with in_memory=True**
```
Solution: Set in_memory=False in config to process files on-the-fly
```

**Issue 6: Accuracy not logged during training**
```
Solution: Ensure task config includes accuracy in metrics:
defaults:
  - override /metrics:
      - accuracy
And check callbacks monitor the right metric: val/graph_label/accuracy
```

**Issue 6: Label format issues**
```
ValueError: Labels must be integers starting from 0
```
**Solution:**
- Ensure labels in `labels.csv` are integers: 0, 1, 2, ..., N-1
- Check for missing or NaN labels
- Verify label dtype is int64/long

**Issue 7: Slow data loading**
```
Solution: 
- Convert PDB to MMTF format (10x smaller, faster)
- Increase num_workers in config
- Set in_memory=True if RAM available
- Check disk I/O speed
```

### Verification Checklist

- [ ] PDB files are preprocessed to contain only chain A
- [ ] PDB files are accessible in `{dataset_path}/pdb/` directory
- [ ] Both split subdirectories exist: `random/` and `structure/`
- [ ] Each split subdirectory has train/val/test_split.csv files
- [ ] Split CSV files have correct format (pdb_id or pdb_id,chain)
- [ ] Shared `labels.csv` exists at dataset root
- [ ] Labels CSV has entry for every PDB ID in all splits
- [ ] Labels are integers starting from 0
- [ ] Config files exist for both split types
- [ ] Config files have correct `num_classes` matching your data
- [ ] Config files have correct `pdb_dir` path
- [ ] Config files specify correct `split_type` parameter
- [ ] DataModule implements all abstract methods
- [ ] Task config uses accuracy as monitoring metric
- [ ] Test script runs without errors for both splits
- [ ] First training step completes successfully
- [ ] Validation accuracy is logged during training
- [ ] Test accuracy is reported after training
- [ ] Both splits can be loaded without errors
- [ ] Same PDB files are used for both splits

---

## 8. Quick Reference for ec_proteinshake Dataset

### Minimal Setup Steps

1. **Run Data Preparation Script:**
```bash
python create_raw_data.py
```

This creates:
```bash
data/ec_proteinshake/
├── pdb/                    # Preprocessed PDB files (chain A only)
│   ├── 1abc.pdb
│   └── ...
├── random/                 # Random split CSVs
│   ├── train_split.csv
│   ├── val_split.csv
│   └── test_split.csv
├── structure/              # Structure-based split CSVs
│   ├── train_split.csv
│   ├── val_split.csv
│   └── test_split.csv
└── labels.csv              # Shared labels (pdb_id,label)
```

2. **Create DataModule:** `proteinworkshop/datasets/ec_proteinshake.py` (see Section 1.3)

3. **Create Config Files:**

`proteinworkshop/config/dataset/ec_random.yaml`:
```yaml
datamodule:
  _target_: proteinworkshop.datasets.ec_proteinshake.ECPSDataModule
  path: ${env.paths.data}/ec_proteinshake/
  split_type: "random"  # Random split
  pdb_dir: ${env.paths.data}/ec_proteinshake/pdb/
  format: "pdb"
  batch_size: 32
  num_workers: 8
  transforms: ${transforms}
  in_memory: True
num_classes: 7  # EC classes 0-6
```

`proteinworkshop/config/dataset/ec_structure.yaml`:
```yaml
datamodule:
  _target_: proteinworkshop.datasets.ec_proteinshake.ECPSDataModule
  path: ${env.paths.data}/ec_proteinshake/
  split_type: "structure"  # Structure-based split
  pdb_dir: ${env.paths.data}/ec_proteinshake/pdb/
  format: "pdb"
  batch_size: 32
  num_workers: 8
  transforms: ${transforms}
  in_memory: True
num_classes: 7  # Same classes
```

4. **Train with Different Splits:**
```bash
# Random split
workshop train \
    dataset=ec_random \
    encoder=gear_net \
    features=ca_angles \
    task=multiclass_graph_classification \
    callbacks.model_checkpoint.monitor=val/graph_label/accuracy \
    callbacks.early_stopping.monitor=val/graph_label/accuracy

# Structure split
workshop train \
    dataset=ec_structure \
    encoder=gear_net \
    features=ca_angles \
    task=multiclass_graph_classification \
    callbacks.model_checkpoint.monitor=val/graph_label/accuracy \
    callbacks.early_stopping.monitor=val/graph_label/accuracy
```

5. **Verify Setup:**
```python
# Quick test for random split
python proteinworkshop/datasets/ec_proteinshake.py
```

### Key Differences from Standard Multi-Chain Datasets

| Aspect | Standard Dataset | ec_proteinshake (Multi-Split) |
|--------|------------------|-------------------------------|
| **PDB Files** | Multi-chain, from RCSB | Preprocessed, chain A only |
| **Chain Column** | Required in splits | Optional (auto-set to 'A') |
| **PDB Directory** | Shared `/pdb/` | Local `{dataset}/pdb/` |
| **Download** | Auto-download from RCSB | Files must exist locally |
| **Label Format** | `pdb.chain` or `pdb_id,chain` | `pdb_id` only |
| **Validation Metric** | Configurable | Accuracy (specified) |
| **Multiple Splits** | Usually single split | Random + Structure splits |
| **Split Storage** | Root directory | Subdirectories (`random/`, `structure/`) |
| **Labels File** | One per split (optional) | Single shared file |

### Common Command Line Overrides

```bash
# Use different batch size
dataset.datamodule.batch_size=64

# Use different split type (if using parameterized config)
dataset.datamodule.split_type=structure

# Use subset of data for testing
dataset.datamodule.dataset_fraction=0.1

# Use different features
features=ca_bb

# Change validation metric
callbacks.model_checkpoint.monitor=val/graph_label/f1_score

# Increase workers for faster loading
dataset.datamodule.num_workers=16

# Disable in-memory loading
dataset.datamodule.in_memory=False
```

---

## 9. EC ProteinShake Quick Reference

### 9.1 Implementation Checklist

**✅ COMPLETED:**
- [x] Dataset created via `create_raw_data.py`
- [x] DataModule: `proteinworkshop/datasets/ec_proteinshake.py`
- [x] Dataset configs: `ec_random.yaml`, `ec_structure.yaml`
- [x] DataModule tested via `test_ec_datamodule.py`
- [x] Task config: Use built-in `multiclass_graph_classification`

**📝 TODO (Create These Files):**
- [ ] `proteinworkshop/config/features/ec_custom.yaml` (equivariant features)
- [ ] `proteinworkshop/config/features/ec_gearnet.yaml` (invariant features)

### 9.2 Required Feature Config Files

**Your Custom Configs (To Create):**

**File 1:** `proteinworkshop/config/features/ec_custom.yaml`
```yaml
# For GVP and GCPNet (equivariant models)
# Your specification: knn_30 + dihedrals + orientation
_target_: proteinworkshop.features.factory.ProteinFeaturiser
representation: CA
scalar_node_features:
  - amino_acid_one_hot
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

**File 2:** `proteinworkshop/config/features/ec_gearnet.yaml`
```yaml
# For GearNet (invariant model)
# Your specification: knn_30 + dihedrals + alpha + kappa
_target_: proteinworkshop.features.factory.ProteinFeaturiser
representation: CA
scalar_node_features:
  - amino_acid_one_hot
  - dihedrals
  - alpha
  - kappa
vector_node_features: []
edge_types:
  - knn_30
scalar_edge_features:
  - edge_distance
vector_edge_features: []
```

**Alternative: Use Framework's Built-in Configs (No Need to Create):**

You can also use the framework's pre-configured feature sets:

```bash
# For GearNet - use built-in all_invariant_ca
features=all_invariant_ca
# Includes: one-hot + dihedrals + sidechain_torsions + alpha + kappa
# Edges: knn_10 + eps_8 (dual edge types)

# For GVP/GCPNet - use built-in all_equivariant_ca  
features=all_equivariant_ca
# Includes: one-hot + orientation + edge_vectors
# Edges: knn_10 + eps_8 (dual edge types)
```

**Recommendation:** Create BOTH your custom configs AND try the built-in ones to compare performance!

### 9.3 Training Commands

**Option A: Your Custom Configs (knn_30 + dihedrals)**

```bash
# 1. GearNet + Random (Your Config)
workshop train dataset=ec_random task=multiclass_graph_classification encoder=gear_net features=ec_gearnet

# 2. GearNet + Structure (Your Config)
workshop train dataset=ec_structure task=multiclass_graph_classification encoder=gear_net features=ec_gearnet

# 3. GVP + Random (Your Config)
workshop train dataset=ec_random task=multiclass_graph_classification encoder=gvp features=ec_custom

# 4. GVP + Structure (Your Config)
workshop train dataset=ec_structure task=multiclass_graph_classification encoder=gvp features=ec_custom

# 5. GCPNet + Random (Your Config)
workshop train dataset=ec_random task=multiclass_graph_classification encoder=gcpnet features=ec_custom dataset.datamodule.batch_size=16

# 6. GCPNet + Structure (Your Config)
workshop train dataset=ec_structure task=multiclass_graph_classification encoder=gcpnet features=ec_custom dataset.datamodule.batch_size=16
```

**Option B: Framework's Built-in Configs (Proven Baselines)**

```bash
# 1. GearNet + Random (Framework Config)
workshop train dataset=ec_random task=multiclass_graph_classification encoder=gear_net features=all_invariant_ca hparams=gear_net_edge_ca_angles

# 2. GearNet + Structure (Framework Config)
workshop train dataset=ec_structure task=multiclass_graph_classification encoder=gear_net features=all_invariant_ca hparams=gear_net_edge_ca_angles

# 3. GVP + Random (Framework Config)
workshop train dataset=ec_random task=multiclass_graph_classification encoder=gvp features=all_equivariant_ca

# 4. GVP + Structure (Framework Config)
workshop train dataset=ec_structure task=multiclass_graph_classification encoder=gvp features=all_equivariant_ca

# 5. GCPNet + Random (Framework Config)
workshop train dataset=ec_random task=multiclass_graph_classification encoder=gcpnet features=all_equivariant_ca dataset.datamodule.batch_size=16 hparams=gcpnet_ca_sc

# 6. GCPNet + Structure (Framework Config)
workshop train dataset=ec_structure task=multiclass_graph_classification encoder=gcpnet features=all_equivariant_ca dataset.datamodule.batch_size=16 hparams=gcpnet_ca_sc
```

**Recommendation:** Run BOTH sets of experiments to compare:
- **Your configs:** Test your hypothesis about knn_30 + dihedrals being optimal for EC
- **Framework configs:** Establish proven baselines with tuned hyperparameters
- **Compare results:** See which feature set works better for enzyme classification

### 9.4 Feature Configuration Comparison

**Your Custom vs Framework Built-in:**

| Feature Aspect | Your Config | Framework Config | Notes |
|----------------|-------------|------------------|-------|
| **GearNet Features** | `ec_gearnet` | `all_invariant_ca` | |
| - Node Features | one-hot + dihedrals + alpha + kappa (31-dim) | one-hot + dihedrals + sidechain + alpha + kappa (39-dim) | Framework adds sidechain torsions |
| - Edge Types | knn_30 (single) | knn_10 + eps_8 (dual) | Framework uses multi-edge |
| - Edge Features | distance (1-dim) | distance + type + node_features + seq_distance (4+ dim) | Framework much richer |
| **GVP/GCPNet Features** | `ec_custom` | `all_equivariant_ca` | |
| - Node Scalar | one-hot + dihedrals (27-dim) | one-hot only (21-dim) | You add dihedrals |
| - Node Vector | orientation (2 × 3) | orientation (2 × 3) | ✓ Same |
| - Edge Types | knn_30 (single) | knn_10 + eps_8 (dual) | Framework uses multi-edge |
| - Edge Scalar | distance (1-dim) | distance + type + node_features + seq_distance (45-dim) | Framework much richer |
| - Edge Vector | edge_vectors (3-dim) | edge_vectors (3-dim) | ✓ Same |
| **Graph Density** | 30-NN (dense) | 10-NN + 8Å (balanced) | Your graphs are denser |
| **Hyperparameters** | Default | Pre-tuned via hparams | Framework provides tuned LR/dropout |

**Key Insights:**

1. **Your Approach:**
   - ✅ Denser graphs (knn_30 vs knn_10)
   - ✅ More node features for equivariant models (adds dihedrals)
   - ✅ Simpler (single edge type, easier to debug)
   - ❌ Missing rich edge features
   - ❌ No pre-tuned hyperparameters

2. **Framework Approach:**
   - ✅ Multi-edge types (richer structural connectivity)
   - ✅ Rich edge features (node_features, sequence_distance)
   - ✅ Pre-tuned hyperparameters
   - ✅ Proven on many benchmarks
   - ❌ Fewer node features for equivariant models
   - ❌ Sparser graphs (might miss long-range interactions)

3. **Best Practice:** Run BOTH and compare:
   - If your config > framework: Dense graphs + dihedrals help EC classification
   - If framework > your config: Rich edge features + multi-edge types are more important
   - This becomes a useful ablation study!

### 9.5 Model Comparison

| Model | Features | Node Dim | Edge Dim | Batch Size | Speed | Memory |
|-------|----------|----------|----------|------------|-------|--------|
| **GearNet** | Invariant | 31 scalar | 1 scalar | 64 | Fast | Low |
| **GVP** | Equivariant | 27 scalar + 2 vector | 1 scalar + 1 vector | 32 | Medium | Medium |
| **GCPNet** | Equivariant | 27 scalar + 2 vector | 1 scalar + 1 vector | 16 | Slow | High |

### 9.6 Framework's Hyperparameter Recommendations

The framework provides pre-tuned hyperparameter configs in `proteinworkshop/config/hparams/` that pair models with feature sets:

**Available Hyperparameter Configs:**

| Config File | Model | Features | Learning Rate | Dropout | Use Case |
|-------------|-------|----------|---------------|---------|----------|
| `gear_net_edge_ca_angles.yaml` | GearNet | ca_angles | 0.0001 | 0.5 | Baseline |
| `gear_net_edge_ca_sc.yaml` | GearNet | ca_sc | 0.0001 | 0.5 | With sidechain |
| `gcpnet_ca_angles.yaml` | GCPNet | ca_angles | 0.001 | 0.3 | Equivariant baseline |
| `gcpnet_ca_sc.yaml` | GCPNet | ca_sc | 0.001 | 0.3 | Equivariant full |

**Usage:**
```bash
# Apply framework's tuned hyperparameters
workshop train \
    dataset=ec_random \
    encoder=gear_net \
    features=all_invariant_ca \
    hparams=gear_net_edge_ca_angles  # Applies lr=0.0001, dropout=0.5
```

**Custom Hyperparameters:**
If you want to tune manually:
```bash
# Override learning rate and dropout
workshop train \
    dataset=ec_random \
    encoder=gear_net \
    features=ec_gearnet \
    optimiser.lr=0.0001 \
    encoder.decoder_dropout=0.5 \
    optimiser.weight_decay=1e-5
```

**Typical Hyperparameter Ranges:**

| Model | LR Range | Dropout Range | Weight Decay |
|-------|----------|---------------|--------------|
| GearNet | 1e-5 to 1e-3 | 0.3 to 0.5 | 1e-6 to 1e-4 |
| GVP | 5e-5 to 5e-4 | 0.1 to 0.3 | 1e-6 to 1e-5 |
| GCPNet | 5e-4 to 1e-3 | 0.2 to 0.4 | 1e-6 to 1e-5 |

### 9.7 Feature Dimensions Breakdown

**GearNet (Invariant):**
- Node features: 21 (one-hot) + 6 (dihedrals) + 2 (alpha) + 2 (kappa) = **31 scalar**
- Edge features: 1 (distance) = **1 scalar**

**GVP/GCPNet (Equivariant):**
- Node scalar: 21 (one-hot) + 6 (dihedrals) = **27 scalar**
- Node vector: 2 orientation vectors = **2 × 3 = 6 vector**
- Edge scalar: 1 (distance) = **1 scalar**
- Edge vector: 1 directional vector = **1 × 3 = 3 vector**

### 9.6 Expected Results

Based on similar enzyme classification tasks:

| Configuration | Expected Test Accuracy | Notes |
|---------------|----------------------|-------|
| GearNet + Random | 75-85% | Baseline, fast training |
| GearNet + Structure | 65-75% | Harder split (no similar structures) |
| GVP + Random | 78-88% | Equivariance helps |
| GVP + Structure | 68-78% | Better generalization |
| GCPNet + Random | 80-90% | Best performance |
| GCPNet + Structure | 70-80% | Most robust to structure splits |

**Structure split should always be 5-10% lower than random split** due to no training data leakage.

### 9.7 Troubleshooting Common Issues

**Issue 1: Feature config not found**
```
FileNotFoundError: proteinworkshop/config/features/ec_custom.yaml not found
```
**Solution:** Create the feature config files as shown in section 9.2

**Issue 2: Feature dimension mismatch**
```
RuntimeError: Expected input dimension 27, got 31
```
**Solution:** Check that you're using the right feature config for your model:
- GearNet → use `ec_gearnet` (31-dim scalars)
- GVP/GCPNet → use `ec_custom` (27-dim scalars + vectors)

**Issue 3: Missing vector features**
```
AttributeError: 'Data' object has no attribute 'orientation'
```
**Solution:** GVP/GCPNet require equivariant features. Use `features=ec_custom`, not `features=ec_gearnet`

**Issue 4: OOM (Out of Memory)**
```
RuntimeError: CUDA out of memory
```
**Solution:** Reduce batch size:
- GearNet: Try 32 or 16
- GVP: Try 16 or 8
- GCPNet: Try 8 or 4

**Issue 5: Accuracy not improving**
```
Val accuracy stuck at ~14% (random guessing for 7 classes)
```
**Solution:**
- Check learning rate (try 1e-4, 5e-5, 1e-5)
- Verify labels are correct (0-6 for EC 1-7)
- Check class imbalance - may need weighted loss
- Try different encoder.emb_dim (256, 512, 1024)

### 9.8 Key Differences from Tutorial Examples

| Aspect | Tutorial (CATH) | Your EC Dataset |
|--------|----------------|-----------------|
| **Split Types** | Single (random) | Dual (random + structure) |
| **PDB Source** | Multi-chain from RCSB | Single-chain preprocessed |
| **Labels** | One per split | Shared `labels.csv` |
| **Task** | Fold classification | EC classification |
| **Metric** | Accuracy (general) | **Accuracy only** (your requirement) |
| **Features** | Various built-in | **Custom** (knn_30, dihedrals, orientation) |

### 9.9 Next Steps After Training

1. **Compare Split Performance:**
   ```python
   # Compare random vs structure split results
   random_acc = results["ec_random"]["test_acc"]
   structure_acc = results["ec_structure"]["test_acc"]
   gap = random_acc - structure_acc
   print(f"Generalization gap: {gap:.2%}")
   ```

2. **Analyze Per-Class Performance:**
   - Which EC classes are hardest to classify?
   - Are there specific confusions (e.g., EC1 ↔ EC2)?
   - Does class imbalance affect performance?

3. **Ablation Studies:**
   - Remove dihedrals: How much does accuracy drop?
   - Change k in knn_k: What's optimal?
   - Try different pooling: sum vs mean vs max

4. **Hyperparameter Tuning:**
   - Learning rate sweep: [1e-5, 5e-5, 1e-4, 5e-4]
   - Hidden dimension sweep: [128, 256, 512, 1024]
   - Layer depth sweep: [3, 4, 5, 6, 8, 10]

---

## 10. Summary: Framework Feature Recommendations vs Your Specifications

### 10.1 What the Framework Recommends

The ProteinWorkshop framework has **built-in feature configurations** that have been tested across multiple benchmarks:

**For Invariant Models (GearNet):**
- **Config:** `all_invariant_ca`
- **Node Features:** one-hot + dihedrals + sidechain_torsions + alpha + kappa (39-dim)
- **Edge Types:** knn_10 + eps_8 (dual edge types)
- **Edge Features:** distance + edge_type + node_features + sequence_distance
- **Tuned Hyperparams:** lr=0.0001, dropout=0.5 (via `gear_net_edge_ca_angles`)

**For Equivariant Models (GVP/GCPNet):**
- **Config:** `all_equivariant_ca`
- **Node Scalar:** one-hot only (21-dim)
- **Node Vector:** orientation (2 × 3)
- **Edge Types:** knn_10 + eps_8 (dual edge types)
- **Edge Scalar:** distance + edge_type + node_features + sequence_distance (45-dim)
- **Edge Vector:** edge_vectors (3-dim)
- **Tuned Hyperparams:** lr=0.001, dropout=0.3 (via `gcpnet_ca_sc`)

### 10.2 Your Specifications

**For Invariant (GearNet):**
- knn_30 (denser than framework's knn_10)
- one-hot + dihedrals + alpha + kappa (31-dim, no sidechain)
- edge_distance only (simpler than framework's multi-feature edges)

**For Equivariant (GVP/GCPNet):**
- knn_30 (denser than framework's knn_10)
- one-hot + dihedrals (27-dim, MORE than framework's 21-dim scalars)
- orientation + edge_vectors (same as framework)
- edge_distance only (simpler than framework's multi-feature edges)

### 10.3 Key Differences and Tradeoffs

| Aspect | Your Approach | Framework Approach | Implication |
|--------|---------------|-------------------|-------------|
| **Graph Density** | knn_30 (denser) | knn_10 + eps_8 (balanced) | Yours: More edges, captures long-range. Framework: Less memory, dual connectivity |
| **Node Features (Equivariant)** | Add dihedrals (+6 dim) | Minimal (one-hot only) | Yours: More structural info. Framework: Relies on vector features |
| **Edge Features** | Distance only | Distance + type + node_features + seq_distance | Yours: Simpler, faster. Framework: Richer information |
| **Edge Types** | Single (knn only) | Dual (knn + eps) | Yours: Easier to train. Framework: Captures multiple relationships |
| **Hyperparameters** | Need tuning | Pre-tuned | Framework provides starting point |

### 10.4 Recommended Experimental Strategy

**Phase 1: Establish Baselines (Use Framework Configs)**
```bash
# GearNet baseline
workshop train dataset=ec_random encoder=gear_net features=all_invariant_ca hparams=gear_net_edge_ca_angles

# GVP baseline  
workshop train dataset=ec_random encoder=gvp features=all_equivariant_ca

# GCPNet baseline
workshop train dataset=ec_random encoder=gcpnet features=all_equivariant_ca hparams=gcpnet_ca_sc
```
→ These give you proven baselines with tuned hyperparameters

**Phase 2: Test Your Hypotheses (Use Custom Configs)**
```bash
# GearNet with your config
workshop train dataset=ec_random encoder=gear_net features=ec_gearnet

# GVP with your config
workshop train dataset=ec_random encoder=gvp features=ec_custom

# GCPNet with your config
workshop train dataset=ec_random encoder=gcpnet features=ec_custom
```
→ These test if denser graphs + dihedrals improve EC classification

**Phase 3: Ablation Studies**
```bash
# Test k values
workshop train ... features.edge_types=[knn_10]  # Framework's k
workshop train ... features.edge_types=[knn_20]  # Middle ground
workshop train ... features.edge_types=[knn_30]  # Your k
workshop train ... features.edge_types=[knn_50]  # Even denser

# Test dihedrals importance
workshop train ... features=all_equivariant_ca  # No dihedrals
workshop train ... features=ec_custom            # With dihedrals

# Test edge feature richness
# (Requires creating intermediate configs)
```

**Phase 4: Structure Split Comparison**
```bash
# Run best configs on structure split
workshop train dataset=ec_structure encoder=BEST_MODEL features=BEST_FEATURES
```
→ Tests generalization to unseen protein folds

### 10.5 Expected Outcomes

**If Framework Configs Win:**
- Multi-edge types (knn + eps) are crucial for enzyme classification
- Rich edge features help capture functional relationships
- Proven hyperparameters matter more than dense graphs

**If Your Configs Win:**
- Denser graphs (knn_30) capture important long-range interactions in enzymes
- Adding dihedrals to equivariant models provides valuable structural context
- Simpler feature sets can be more effective with proper graph density

**If Similar Performance:**
- Both approaches are viable
- Choice depends on computational budget (your configs are denser → slower)
- Framework configs better for quick experiments (pre-tuned)

### 10.6 Documentation References

**Framework Feature Configs:**
- `proteinworkshop/config/features/all_invariant_ca.yaml`
- `proteinworkshop/config/features/all_equivariant_ca.yaml`
- `proteinworkshop/config/features/ca_angles.yaml`
- `proteinworkshop/config/features/ca_sc.yaml`

**Framework Hyperparameters:**
- `proteinworkshop/config/hparams/gear_net_edge_*.yaml`
- `proteinworkshop/config/hparams/gcpnet_*.yaml`

**Your Custom Configs (To Create):**
- `proteinworkshop/config/features/ec_custom.yaml`
- `proteinworkshop/config/features/ec_gearnet.yaml`

---

## 11. Additional Resources

- **Tutorial Notebooks:**
  - `notebooks/adding_new_dataset_tutorial.ipynb`
  - `notebooks/adding_new_task_tutorial.ipynb`
- **Example Implementations:**
  - EC Reaction: `proteinworkshop/datasets/ec_reaction.py`
  - CATH: `proteinworkshop/datasets/cath.py`
- **Your Implementation:**
  - DataModule: `proteinworkshop/datasets/ec_proteinshake.py`
  - Data Script: `create_raw_data.py`
  - Test Script: `test_ec_datamodule.py`
- **Documentation:**
  - Main docs: https://www.proteins.sh
  - Feature factory: `proteinworkshop/features/factory.py`
  - Base classes: `proteinworkshop/datasets/base.py`

---

## 12. Contact

For questions or issues:
1. Create an issue on [GitHub](https://github.com/a-r-j/ProteinWorkshop/issues)
2. Check existing datasets for examples
3. Review the tutorial notebooks

Good luck with your EC ProteinShake experiments! 🧬🔬
