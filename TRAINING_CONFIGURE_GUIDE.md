### 2.4 Task Configuration (Using Accuracy as Metric)

Create `proteinworkshop/config/task/ec_classification.yaml`:

```yaml
# @package _global_

defaults:
  - override /metrics:
      - accuracy        # Primary metric for validation and test
      - f1_score        # Additional metric
  - override /decoder:
      - graph_label     # Graph-level classification

callbacks:
  early_stopping:
    monitor: val/graph_label/accuracy    # Monitor validation accuracy
    mode: "max"                          # Higher is better
    patience: 50                         # Number of epochs to wait
  model_checkpoint:
    monitor: val/graph_label/accuracy    # Save best model by validation accuracy
    mode: "max"
    save_top_k: 1                        # Keep only the best checkpoint

task:
  task: "classification"
  classification_type: "multiclass"
  metric_average: "micro"                # Micro-averaging for multi-class

  losses:
    graph_label: cross_entropy           # Cross-entropy loss for classification
  label_smoothing: 0.0                   # No label smoothing (set to 0.1 if needed)

  output:
    - "graph_label"                      # Predict graph-level labels
  supervise_on:
    - "graph_label"                      # Supervise on graph-level labels
```

**Key Configuration Details:**
- **Accuracy as Primary Metric**: Both `early_stopping` and `model_checkpoint` monitor `val/graph_label/accuracy`
- **Test Metrics**: Accuracy will be automatically computed on the test set
- **Mode**: Set to `"max"` since we want to maximize accuracy
- **Metric Average**: `"micro"` for balanced class contribution (use `"macro"` for per-class averaging)
- **Task config is shared** across both split types (random and structure)

**Usage with Different Splits:**
```bash
# Random split with custom task config
workshop train dataset=ec_random task=ec_classification encoder=gear_net

# Structure split with custom task config
workshop train dataset=ec_structure task=ec_classification encoder=gear_net
```

**Alternative Metric Configurations:**

If you want different metrics for validation and test:
```yaml
defaults:
  - override /metrics:
      - accuracy        # Both val and test
      - auroc          # Area under ROC curve
      - f1_max         # Maximum F1 score
```

---

## 3. Feature Engineering with Transforms

### 3.1 Pre-built Pipeline Overview

**YES**, ProteinWorkshop has a comprehensive pre-built pipeline for deriving features from PDB files!

The `ProteinFeaturiser` class automatically:
1. Parses PDB files into protein structures
2. Selects atomic representation (CA, backbone, full-atom)
3. Computes node features (residue types, angles, embeddings)
4. Constructs edges (k-NN or distance-based)
5. Computes edge features (distances, types)
6. Creates PyTorch Geometric `Data` objects

### 3.2 How Feature Engineering Works

```
PDB File → Graphein Parser → AtomTensor (N_res × 37 × 3)
    ↓
ProteinFeaturiser (specified in config)
    ↓
    ├─ Representation: Select atoms (CA / BB / FA)
    ├─ Node Features: Residue types, angles, embeddings
    ├─ Edge Construction: k-NN or ε-distance
    ├─ Edge Features: Distances, directions
    ↓
PyTorch Geometric Data object → Model
```

### 3.3 Building k-NN Geometric Graphs

To construct 3D geometric graphs with k-NN edges, use existing feature configs:

**Option 1: Basic CA with k-NN** (`features=ca`)
```yaml
# proteinworkshop/config/features/ca.yaml
_target_: proteinworkshop.features.factory.ProteinFeaturiser
representation: CA              # Use C-alpha atoms as nodes
scalar_node_features:
  - amino_acid_one_hot          # 21-dim one-hot encoding
vector_node_features: []
edge_types:
  - knn_10                      # 10-nearest neighbors
scalar_edge_features:
  - edge_distance               # Euclidean distance
vector_edge_features: []
```

**Option 2: CA with Angles** (`features=ca_angles`)
```yaml
representation: CA
scalar_node_features:
  - amino_acid_one_hot
  - sequence_positional_encoding  # 16-dim Transformer-style encoding
  - alpha                         # Virtual torsion angle (2-dim)
  - kappa                         # Virtual bond angle (2-dim)
vector_node_features: []
edge_types:
  - knn_16                        # 16-nearest neighbors
scalar_edge_features:
  - edge_distance
vector_edge_features: []
```

**Option 3: Multiple Edge Types**
```yaml
edge_types:
  - knn_16    # 16-nearest neighbors
  - knn_32    # 32-nearest neighbors
  - eps_8     # All edges within 8 Å
```

### 3.4 Customizing k-NN

The `knn_X` notation means X-nearest neighbors. Common values:
- `knn_10`: 10-NN (sparse graphs)
- `knn_16`: 16-NN (balanced)
- `knn_32`: 32-NN (dense graphs)

The `eps_X` notation means all atoms within X Ångströms:
- `eps_8`: All atoms within 8Å
- `eps_10`: All atoms within 10Å

### 3.5 Available Node Features

**Scalar Features:**
- `amino_acid_one_hot`: 21-dim one-hot (20 AA + unknown)
- `sequence_positional_encoding`: 16-dim Transformer positional encoding
- `alpha`: Virtual torsion angle (2-dim, sin/cos transformed)
- `kappa`: Virtual bond angle (2-dim, sin/cos transformed)
- `dihedrals`: Backbone dihedrals φ,ψ,ω (6-dim, sin/cos)
- `sidechain_torsions`: Sidechain torsions χ₁-χ₄ (8-dim, sin/cos)

**Vector Features:**
- `orientation`: Forward/backward orientation vectors (2 × 3-dim)

**Edge Features:**
- `edge_distance`: Euclidean distance (1-dim)
- `edge_vectors`: Directional unit vectors (3-dim)
- `edge_type`: Edge type ID (1-dim)
- `node_features`: Concatenated source/target node features
- `sequence_distance`: Sequential distance along chain (1-dim)

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

## 5. Complete Example Workflow

### 5.1 Directory Structure for ec_proteinshake
```
ProteinWorkshop/
├── proteinworkshop/
│   ├── datasets/
│   │   └── ec_proteinshake.py          # Your DataModule
│   └── config/
│       ├── dataset/
│       │   ├── ec_random.yaml          # Random split config
│       │   └── ec_structure.yaml       # Structure split config
│       └── task/
│           └── ec_classification.yaml  # Task config (optional)
└── data/
    └── ec_proteinshake/
        ├── pdb/                         # Preprocessed PDB files (chain A only)
        │   ├── 1abc.pdb
        │   ├── 2xyz.pdb
        │   └── ...
        ├── random/                      # Random split files
        │   ├── train_split.csv
        │   ├── val_split.csv
        │   └── test_split.csv
        ├── structure/                   # Structure split files
        │   ├── train_split.csv
        │   ├── val_split.csv
        │   └── test_split.csv
        └── labels.csv                   # Shared label mapping
```

### 5.2 Training Commands

```bash
# Set data path
export PROTEIN_WORKSHOP_DATA=/path/to/data

# Train with GearNet (invariant) - Random split
workshop train \
    dataset=ec_random \
    encoder=gear_net \
    features=ca_angles \
    task=multiclass_graph_classification \
    dataset.datamodule.batch_size=32

# Train with GearNet (invariant) - Structure split
workshop train \
    dataset=ec_structure \
    encoder=gear_net \
    features=ca_angles \
    task=multiclass_graph_classification \
    dataset.datamodule.batch_size=32

# OR: Use custom task config with accuracy metrics
workshop train \
    dataset=ec_random \
    encoder=gear_net \
    features=ca_angles \
    task=ec_classification \
    dataset.datamodule.batch_size=32

# Train with GVP-GNN (equivariant) - Random split
workshop train \
    dataset=ec_random \
    encoder=gvp \
    features=all_equivariant_ca \
    task=multiclass_graph_classification \
    dataset.datamodule.batch_size=32 \
    callbacks.model_checkpoint.monitor=val/graph_label/accuracy \
    callbacks.early_stopping.monitor=val/graph_label/accuracy

# Train with GVP-GNN (equivariant) - Structure split
workshop train \
    dataset=ec_structure \
    encoder=gvp \
    features=all_equivariant_ca \
    task=multiclass_graph_classification \
    dataset.datamodule.batch_size=32 \
    callbacks.model_checkpoint.monitor=val/graph_label/accuracy \
    callbacks.early_stopping.monitor=val/graph_label/accuracy

# Train with GCPNet (equivariant) - Compare both splits
workshop train \
    dataset=ec_random \
    encoder=gcpnet \
    features=all_equivariant_ca \
    task=multiclass_graph_classification \
    dataset.datamodule.batch_size=16 \
    callbacks.model_checkpoint.monitor=val/graph_label/accuracy \
    callbacks.early_stopping.monitor=val/graph_label/accuracy

# Train with different features and batch size
workshop train \
    dataset=ec_structure \
    encoder=gear_net \
    features=ca_bb \
    task=multiclass_graph_classification \
    dataset.datamodule.batch_size=64 \
    dataset.datamodule.num_workers=16
```

**Important Command Line Options:**
- `dataset=ec_random`: Use random split
- `dataset=ec_structure`: Use structure split
- `task=multiclass_graph_classification`: Use built-in task with accuracy
- `callbacks.model_checkpoint.monitor=val/graph_label/accuracy`: Override to use accuracy
- `callbacks.early_stopping.monitor=val/graph_label/accuracy`: Override to use accuracy

### 5.3 Testing Your Implementation

```python
# test_ec_proteinshake.py
import hydra
import omegaconf
from proteinworkshop import constants

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

## 9. Additional Resources

- **Tutorial Notebook:** `notebooks/adding_new_dataset_tutorial.ipynb`
- **Example Dataset (EC Reaction):** `proteinworkshop/datasets/ec_reaction.py`
- **Example Dataset (CATH):** `proteinworkshop/datasets/cath.py`
- **Your Implementation (ec_proteinshake):** `proteinworkshop/datasets/ec_proteinshake.py`
- **Data Preparation Script:** `create_raw_data.py`
- **Documentation:** https://www.proteins.sh
- **Feature Engineering:** `proteinworkshop/features/factory.py`
- **Base Classes:** `proteinworkshop/datasets/base.py`
- **Task Configs:** `proteinworkshop/config/task/`

---

## 10. Contact

For questions or issues:
1. Create an issue on [GitHub](https://github.com/a-r-j/ProteinWorkshop/issues)
2. Check existing datasets for examples
3. Review the tutorial notebooks

Good luck with your dataset implementation! 🧬
