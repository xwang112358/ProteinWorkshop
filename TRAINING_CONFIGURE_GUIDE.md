

## 1\. Task Configuration (EC ProteinShake - Accuracy Only)

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

  * ✅ Use `task=multiclass_graph_classification` directly.
  * ✅ Accuracy is already the primary metric for early stopping and checkpointing.
  * ✅ Test accuracy will be automatically computed and reported.
  * ✅ This same task config works for both `ec_random` and `ec_structure` splits.

-----

## 2\. Feature Engineering Overview

This diagram shows the flow from a PDB file to the data object used by the model, based on your custom feature requirements.

```
PDB File (1abc.pdb)
    ↓
Graphein Parser → AtomTensor (N_residues × 37 atoms × 3 coordinates)
    ↓
ProteinFeaturiser (e.g., ec_custom.yaml)
    ↓
    ├─ Representation: Select C-alpha atoms (N_residues nodes)
    ├─ Node Features:
    │    ├─ Scalar: amino_acid_one_hot (21-dim) + dihedrals (6-dim)
    │    └─ Vector: orientation (2 × 3-dim)
    ├─ Edge Construction: knn_30 (each node connects to 30 nearest neighbors)
    └─ Edge Features:
          ├─ Scalar: edge_distance (1-dim)
          └─ Vector: edge_vectors (3-dim)
    ↓
PyTorch Geometric Data object
    ↓
Model (GearNet / GVP / GCPNet)
```

-----

## 3\. Model-Specific Configurations

The features you need depend on whether the model is **invariant** (like GearNet) or **equivariant** (like GVP/GCPNet).

### 3.1 Summary: Invariant vs. Equivariant Models

| Model | Type | Scalar Nodes | Vector Nodes | Scalar Edges | Vector Edges | Edge Types |
|-------|------|--------------|--------------|--------------|--------------|------------|
| **GearNet** | Invariant | ✓ | ✗ | ✓ | ✗ | Multiple ✓ |
| **GVP-GNN** | Equivariant | ✓ | ✓ Required | ✓ | ✓ Required | Single |
| **GCPNet** | Equivariant | ✓ | ✓ Required | ✓ | ✓ Required | Multiple ✓ |

-----

### 3.2 Model 1: GearNet (Invariant)

GearNet is an **invariant** model that only uses scalar features.

**Feature Config:** `proteinworkshop/config/features/ec_gearnet.yaml`

```yaml
# Feature config for GearNet (invariant only)
_target_: proteinworkshop.features.factory.ProteinFeaturiser
representation: CA

scalar_node_features:
  - amino_acid_one_hot      # 21-dim
  - dihedrals               # 6-dim (φ, ψ, ω angles)
  - alpha                   # 2-dim (virtual torsion)
  - kappa                   # 2-dim (virtual bond angle)

vector_node_features: []    # ⚠️ Empty - GearNet doesn't use vectors

edge_types:
  - knn_30                  # Your specified k=30

scalar_edge_features:
  - edge_distance           # 1-dim

vector_edge_features: []    # ⚠️ Empty - GearNet doesn't use vectors
```

  * **Total Scalar Node Features:** 21 + 6 + 2 + 2 = **31 dimensions**
  * **Total Scalar Edge Features:** 1 dimension

**Model Config:** `proteinworkshop/config/encoder/gear_net.yaml` (Example)

```yaml
_target_: proteinworkshop.models.graph_encoders.gear_net.GearNet
input_dim: 31               # Auto-computed from feature config
num_relation: 1             # Single edge type (knn_30)
num_layers: 6
emb_dim: 512
...
pool: sum
```

**Pseudo-Code: GearNet Training**

```python
# 1. Model initialization
model = GearNet(
    input_dim=31,         # 21 + 6 + 2 + 2
    edge_input_dim=1,     # edge_distance only
    num_relation=1,       # single edge type
    hidden_dim=512,
    num_layers=6,
    num_classes=7         # EC classes 0-6
)

# 2. Training loop
for batch in train_loader:
    # batch.x: [N_nodes, 31] - scalar features only
    # batch.edge_index: [2, N_edges] - knn_30 connectivity
    # batch.edge_attr: [N_edges, 1] - edge distances
    # batch.batch: [N_nodes] - graph assignment
    # batch.graph_label: [batch_size] - EC class labels (0-6)
    
    # Forward pass (invariant to rotations/translations)
    node_embeddings = model.encode(batch.x, batch.edge_index, batch.edge_attr)
    graph_embeddings = global_sum_pool(node_embeddings, batch.batch)
    logits = model.decode(graph_embeddings) # [batch_size, 7]
    
    loss = cross_entropy(logits, batch.graph_label)
    ...
```

-----

### 3.3 Model 2: GVP-GNN (Equivariant)

GVP is an **equivariant** model that requires both scalar and vector features. Your custom spec (dihedrals + orientation) is perfect for this.

**Feature Config:** `proteinworkshop/config/features/ec_custom.yaml`

```yaml
# Your specified features - perfect for GVP/GCPNet
_target_: proteinworkshop.features.factory.ProteinFeaturiser
representation: CA

scalar_node_features:
  - amino_acid_one_hot      # 21-dim
  - dihedrals               # 6-dim

vector_node_features:
  - orientation             # 2 × 3-dim (REQUIRED for GVP)

edge_types:
  - knn_30                  # Your specified k=30

scalar_edge_features:
  - edge_distance           # 1-dim

vector_edge_features:
  - edge_vectors            # 3-dim (REQUIRED for GVP)
```

  * **Scalar Node Features:** 21 + 6 = **27 dimensions**
  * **Vector Node Features:** **2 vectors** (orientation)
  * **Scalar Edge Features:** 1 dimension
  * **Vector Edge Features:** **1 vector** (edge\_vectors)

**Model Config:** `proteinworkshop/config/encoder/gvp.yaml` (Example)

```yaml
_target_: proteinworkshop.models.graph_encoders.gvp.GVPGNNModel
s_dim: 128                  # Scalar hidden dimension
v_dim: 16                   # Vector hidden dimension
num_layers: 5
pool: "sum"
...
```

**Pseudo-Code: GVP Training**

```python
# 1. Model initialization
model = GVP(
    node_s_in=27,         # 21 + 6
    node_v_in=2,          # 2 orientation vectors
    edge_s_in=1,          # edge_distance
    edge_v_in=1,          # edge_vectors
    node_s_dim=128,       # Hidden scalar dimension
    node_v_dim=16,        # Hidden vector dimension
    num_layers=5,
    num_classes=7         # EC classes 0-6
)

# 2. Training loop
for batch in train_loader:
    # Scalar features
    # batch.x: [N_nodes, 27] - one-hot + dihedrals
    # batch.edge_attr: [N_edges, 1] - edge_distance
    
    # Vector features (equivariant to SE(3))
    # batch.orientation: [N_nodes, 2, 3] - direction vectors
    # batch.edge_vectors: [N_edges, 3] - directional edge vectors
    
    # Forward pass (equivariant)
    node_s, node_v = model.encode(
        node_s=batch.x,
        node_v=batch.orientation,
        edge_index=batch.edge_index,
        edge_s=batch.edge_attr,
        edge_v=batch.edge_vectors
    )
    # node_s: [N_nodes, 128]
    # node_v: [N_nodes, 16, 3]
    
    # Global pooling (only use scalar part for classification)
    graph_s = global_sum_pool(node_s, batch.batch)
    logits = model.decode(graph_s) # [batch_size, 7]
    
    loss = cross_entropy(logits, batch.graph_label)
    ...
```

-----

### 3.4 Model 3: GCPNet (Equivariant)

GCPNet is also **equivariant** and uses the same features as GVP. Its main advantage is that it can natively handle multiple edge types (e.g., `knn_30` and `sequential`).

**Feature Config:** `proteinworkshop/config/features/ec_custom.yaml` (Same as GVP)

```yaml
# Use the same 'ec_custom.yaml' config as GVP.
# ... (see GVP section)
```

  * **Features:** Identical to GVP (27 scalar nodes, 2 vector nodes, etc.)

**Model Config:** `proteinworkshop/config/encoder/gcpnet.yaml` (Example)

```yaml
_target_: proteinworkshop.models.graph_encoders.gcpnet.GCPNetModel
num_layers: 6
emb_dim: 128                # Hidden scalar dimension
node_s_emb_dim: 128
node_v_emb_dim: 16
...
pool: sum
```

**Pseudo-Code: GCPNet Training**

```python
# 1. Model initialization
model = GCPNet(
    node_s_in=27,         # 21 + 6
    node_v_in=2,          # 2 orientation vectors
    edge_s_in=1,          # edge_distance
    edge_v_in=1,          # edge_vectors
    node_s_emb_dim=128,   # Hidden scalar dimension
    node_v_emb_dim=16,    # Hidden vector dimension
    num_layers=6,
    num_classes=7         # EC classes 0-6
)

# 2. Training loop
for batch in train_loader:
    # Scalar features
    # batch.x: [N_nodes, 27]
    # batch.edge_attr: [N_edges, 1]
    
    # Vector features
    # batch.orientation: [N_nodes, 2, 3]
    # batch.edge_vectors: [N_edges, 3]
    
    # Forward pass (equivariant)
    node_s, node_v = model.encode(
        node_s=batch.x,
        node_v=batch.orientation,
        edge_index=batch.edge_index,
        edge_s=batch.edge_attr,
        edge_v=batch.edge_vectors,
        pos=batch.pos # GCPNet can also use coordinates
    )
    # node_s: [N_nodes, 128]
    # node_v: [N_nodes, 16, 3]
    
    graph_s = global_sum_pool(node_s, batch.batch)
    logits = model.decode(graph_s) # [batch_size, 7]
    
    loss = cross_entropy(logits, batch.graph_label)
    ...
```

-----

## 4\. Alternative: Framework-Recommended Features

Instead of your custom configs, you can use the framework's built-in feature sets. These are pre-configured and optimized.

| Feature Config | Type | Description | Best For |
|---|---|---|---|
| `all_invariant_ca` | Invariant | Full scalar features: one-hot + all torsions + virtual angles | GearNet, SchNet |
| `all_equivariant_ca` | Equivariant | Minimal: one-hot + orientation + edge\_vectors | GVP, GCPNet, EGNN |
| `ca_angles` | Invariant | one-hot + positional encoding + virtual angles | General invariant |

**Framework Recommendations by Model:**

  * **For GearNet (Invariant):** Use `features=all_invariant_ca` or `features=ca_angles`.
    ```bash
    workshop train dataset=ec_random encoder=gear_net features=all_invariant_ca
    ```
  * **For GVP/GCPNet (Equivariant):** Use `features=all_equivariant_ca`.
    ```bash
    workshop train dataset=ec_random encoder=gvp features=all_equivariant_ca
    ```

-----

## 5\. Complete Training Workflow

### 5.1 Implementation Checklist

**✅ COMPLETED:**

  * [x] Dataset created via `create_raw_data.py`
  * [x] DataModule: `proteinworkshop/datasets/ec_proteinshake.py`
  * [x] Dataset configs: `ec_random.yaml`, `ec_structure.yaml`
  * [x] DataModule tested via `test_ec_datamodule.py`
  * [x] Task config: Use built-in `multiclass_graph_classification`

**📝 TODO (Create These Files):**

  * [ ] `proteinworkshop/config/features/ec_custom.yaml` (for GVP/GCPNet)
  * [ ] `proteinworkshop/config/features/ec_gearnet.yaml` (for GearNet)

### 5.2 Directory Structure

```
ProteinWorkshop/
├── proteinworkshop/
│   ├── datasets/
│   │   └── ec_proteinshake.py       # ✅ DataModule
│   └── config/
│       ├── dataset/
│       │   ├── ec_random.yaml       # ✅ Random split config
│       │   └── ec_structure.yaml    # ✅ Structure split config
│       ├── features/
│       │   ├── ec_custom.yaml       # 📝 TODO: Equivariant features
│       │   └── ec_gearnet.yaml      # 📝 TODO: Invariant features
│       └── task/
│           └── multiclass_graph_classification.yaml  # ✅ Built-in (use this)
├── data/
│   └── ec_proteinshake/
│       ├── pdb/                     # ✅ PDB files (chain A only)
│       ├── random/                  # ✅ Random split CSVs
│       ├── structure/               # ✅ Structure split CSVs
│       └── labels.csv               # ✅ Shared labels
└── test_ec_datamodule.py            # ✅ Test script
```

### 5.3 Expected Training Output

You should see an output log similar to this, with validation accuracy being monitored and the test accuracy reported at the very end.

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

...

======================================================================
FINAL TEST ACCURACY: 0.8045
======================================================================

Per-Class Performance:
              precision    recall  f1-score   support
         EC1     0.8234    0.8156    0.8195       245
         EC2     0.7845    0.8013    0.7928       198
...
```

-----

## 6\. Quick Reference: Training Commands

This is the consolidated list of commands for all configurations.

### 6.1 Option A: Your Custom Configs (knn\_30 + dihedrals)

**GearNet (Invariant)**

```bash
# Random split
workshop train dataset=ec_random task=multiclass_graph_classification encoder=gear_net features=ec_gearnet

# Structure split
workshop train dataset=ec_structure task=multiclass_graph_classification encoder=gear_net features=ec_gearnet
```

**GVP (Equivariant)**

```bash
# Random split
workshop train dataset=ec_random task=multiclass_graph_classification encoder=gvp features=ec_custom

# Structure split
workshop train dataset=ec_structure task=multiclass_graph_classification encoder=gvp features=ec_custom
```

**GCPNet (Equivariant)**

```bash
# Random split (smaller batch size recommended)
workshop train dataset=ec_random task=multiclass_graph_classification encoder=gcpnet features=ec_custom dataset.datamodule.batch_size=16

# Structure split (smaller batch size recommended)
workshop train dataset=ec_structure task=multiclass_graph_classification encoder=gcpnet features=ec_custom dataset.datammodule.batch_size=16
```

### 6.2 Option B: Framework's Built-in Configs (Proven Baselines)

**GearNet (Invariant)**

```bash
# Random split
workshop train dataset=ec_random task=multiclass_graph_classification encoder=gear_net features=all_invariant_ca hparams=gear_net_edge_ca_angles

# Structure split
workshop train dataset=ec_structure task=multiclass_graph_classification encoder=gear_net features=all_invariant_ca hparams=gear_net_edge_ca_angles
```

**GVP (Equivariant)**

```bash
# Random split
workshop train dataset=ec_random task=multiclass_graph_classification encoder=gvp features=all_equivariant_ca

# Structure split
workshop train dataset=ec_structure task=multiclass_graph_classification encoder=gvp features=all_equivariant_ca
```

**GCPNet (Equivariant)**

```bash
# Random split
workshop train dataset=ec_random task=multiclass_graph_classification encoder=gcpnet features=all_equivariant_ca hparams=gcpnet_ca_sc dataset.datamodule.batch_size=16

# Structure split
workshop train dataset=ec_structure task=multiclass_graph_classification encoder=gcpnet features=all_equivariant_ca hparams=gcpnet_ca_sc dataset.datamodule.batch_size=16
```

### 6.3 Custom Hyperparameter Tuning

You can override any setting from the command line.

```bash
# Example: Override learning rate and dropout for a GearNet run
workshop train \
    dataset=ec_random \
    encoder=gear_net \
    features=ec_gearnet \
    optimiser.lr=0.0001 \
    encoder.decoder_dropout=0.5 \
    optimiser.weight_decay=1e-5
```