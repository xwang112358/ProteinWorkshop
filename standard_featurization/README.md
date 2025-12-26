# Standard Featurization Pipeline

A self-contained pipeline for converting raw protein PDB data into featurized dataloaders ready for model training.

## Workflow

```
Raw PDB Files → ECPSDataModule → ProteinDataLoader → ProteinFeaturiser → Featurized Batch
```

1. **Raw Data**: PDB files and split CSVs created by `create_raw_data_test.py`
2. **DataModule**: `ECPSDataModule` loads and organizes train/val/test splits
3. **DataLoader**: `ProteinDataLoader` creates batches of protein graphs
4. **Featuriser**: `ProteinFeaturiser` computes node/edge features

## Directory Structure

```
standard_featurization/
├── config/
│   └── features/           # Feature configuration files
│       ├── ca_base.yaml    # Basic CA representation
│       ├── ca_angles.yaml  # CA with angle features
│       ├── ca_bb.yaml      # CA with backbone dihedrals
│       └── ...
├── features/               # Feature computation modules
│   ├── factory.py          # ProteinFeaturiser class
│   ├── node_features.py    # Node feature functions
│   ├── edge_features.py    # Edge feature functions
│   └── ...
├── data/
│   └── ec_proteinshake/    # Dataset (created by create_raw_data_test.py)
│       ├── pdb/            # Raw PDB files
│       ├── labels.csv      # Protein labels
│       ├── random/         # Random split CSVs
│       └── structure/      # Structure-based split CSVs
├── datamodule.py           # ECPSDataModule and ProteinDataset
├── custom_types.py         # Type definitions
├── test_feature.py         # Pipeline test script
└── README.md
```

## Usage

### Quick Start

```python
from standard_featurization import get_featurized_dataloaders, featurize_batch

# Get dataloaders and featuriser
train_loader, val_loader, test_loader, featuriser = get_featurized_dataloaders(
    feature_config="ca_base.yaml",
    split_type="random",
    batch_size=4,
)

# Apply featurization to a batch
for batch in train_loader:
    featurized = featurize_batch(batch, featuriser)
    print(f"Node features: {featurized.x.shape}")
    print(f"Edge index: {featurized.edge_index.shape}")
    break
```

### Run Test Script

```bash
cd ProteinWorkshop
python -m standard_featurization.test_feature
```

### Manual Pipeline Setup

```python
import hydra
import omegaconf
from standard_featurization.datamodule import ECPSDataModule
from standard_featurization.features.factory import ProteinFeaturiser

# Load feature config
cfg = omegaconf.OmegaConf.load("standard_featurization/config/features/ca_base.yaml")
featuriser = hydra.utils.instantiate(cfg)

# Create datamodule
datamodule = ECPSDataModule(
    path="standard_featurization/data/ec_proteinshake",
    split_type="random",
    batch_size=32,
)
datamodule.setup("fit")

# Get dataloader
train_loader = datamodule.train_dataloader()

# Featurize batches
for batch in train_loader:
    featurized = featuriser(batch)
    # Use featurized batch for training
```

## Feature Configurations

| Config | Node Features | Edge Features |
|--------|--------------|---------------|
| `ca_base.yaml` | amino_acid_one_hot | edge_distance |
| `ca_angles.yaml` | amino_acid_one_hot, positional_encoding, alpha, kappa | edge_distance |
| `ca_bb.yaml` | amino_acid_one_hot, positional_encoding, alpha, kappa, dihedrals | edge_distance |
| `ca_sc.yaml` | amino_acid_one_hot, positional_encoding, alpha, kappa, dihedrals, sidechain_torsions | edge_distance |
| `all_invariant_ca.yaml` | dihedrals, alpha, kappa | edge_distance |
| `all_equivariant_ca.yaml` | dihedrals, orientation (vector) | edge_distance, edge_vectors (vector) |

## Prerequisites

Generate raw data first:

```bash
python create_raw_data_test.py
```

This creates the `data/ec_proteinshake/` directory with PDB files and split CSVs.

