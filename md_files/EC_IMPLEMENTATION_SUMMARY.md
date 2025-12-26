# EC ProteinShake DataModule - Implementation Summary

## Files Created

### 1. DataModule Implementation
**File:** `proteinworkshop/datasets/ec_proteinshake.py`

**Key Features:**
- Supports both `random` and `structure` split types via `split_type` parameter
- Shares PDB files and labels across all splits (no redundancy)
- Loads split CSVs from split-specific subdirectories (`random/`, `structure/`)
- Loads labels from shared `labels.csv` file
- Implements all required methods: `train_dataset()`, `val_dataset()`, `test_dataset()`
- Provides convenient dataloaders with configurable batch size, workers, etc.

**Main Parameters:**
- `path`: Root dataset directory (e.g., `data/ec_proteinshake/`)
- `split_type`: Which split to use (`"random"` or `"structure"`)
- `batch_size`: Batch size for dataloaders (default: 32)
- `pdb_dir`: PDB directory (defaults to `{path}/pdb/`)
- `format`: PDB file format (`"pdb"`, `"mmtf"`, or `"ent"`)
- `dataset_fraction`: Use fraction of dataset (for debugging)
- `shuffle_labels`: Shuffle labels (for negative control experiments)

### 2. Config Files

#### Random Split Config
**File:** `proteinworkshop/config/dataset/ec_random.yaml`

```yaml
datamodule:
  _target_: proteinworkshop.datasets.ec_proteinshake.ECPSDataModule
  split_type: "random"
  # ... other parameters
num_classes: 7
```

#### Structure Split Config
**File:** `proteinworkshop/config/dataset/ec_structure.yaml`

```yaml
datamodule:
  _target_: proteinworkshop.datasets.ec_proteinshake.ECPSDataModule
  split_type: "structure"
  # ... other parameters
num_classes: 7
```

**Note:** Both configs point to the same dataset path and PDB directory, only differing in `split_type`.

### 3. Test Script
**File:** `test_ec_datamodule.py`

A comprehensive test script that:
1. Tests loading both split types
2. Inspects dataset sizes and batch contents
3. Compares split statistics
4. Verifies shared resources (PDB files, labels)

## Usage Examples

### 1. Direct Python Usage

```python
from proteinworkshop.datasets.ec_proteinshake import ECPSDataModule

# Load random split
datamodule = ECPSDataModule(
    path="data/ec_proteinshake",
    split_type="random",
    batch_size=32
)

datamodule.setup("fit")
train_dl = datamodule.train_dataloader()

for batch in train_dl:
    print(batch.num_graphs)  # Number of proteins in batch
    print(batch.x.shape)     # Node features
    print(batch.graph_label) # EC class labels (0-6)
    break
```

### 2. Training with ProteinWorkshop CLI

```bash
# Train with random split
workshop train dataset=ec_random encoder=gear_net features=ca_angles

# Train with structure split
workshop train dataset=ec_structure encoder=gear_net features=ca_angles
```

### 3. Running the Test Script

```bash
python test_ec_datamodule.py
```

This will:
- Load both split types
- Show dataset statistics
- Compare splits
- Verify everything works correctly

## DataModule Architecture

### Directory Structure Expected
```
ec_proteinshake/
├── pdb/                    # Shared PDB files
│   ├── 1abc.pdb
│   └── ...
├── random/                 # Random split CSVs
│   ├── train_split.csv
│   ├── val_split.csv
│   └── test_split.csv
├── structure/              # Structure split CSVs
│   ├── train_split.csv
│   ├── val_split.csv
│   └── test_split.csv
└── labels.csv              # Shared labels
```

### How It Works

1. **Initialization**: 
   - Takes `path` and `split_type` parameters
   - Sets `split_dir` to `{path}/{split_type}/`
   - Validates directories exist

2. **Loading Labels**:
   - `parse_labels()` loads `{path}/labels.csv`
   - Returns dict: `pdb_id -> label`
   - Labels are 0-indexed (EC 1-7 → labels 0-6)

3. **Loading Splits**:
   - `_load_split(split_name)` loads `{split_dir}/{split_name}_split.csv`
   - Adds chain='A' if not specified (single-chain dataset)
   - Merges with labels from `labels.csv`
   - Returns DataFrame with columns: `pdb_id, chain, label`

4. **Creating Datasets**:
   - Each split method (`train_dataset()`, etc.) calls `_load_split()`
   - Creates `ProteinDataset` with PDB codes, chains, and labels
   - Uses shared `pdb_dir` for structure files

5. **Creating Dataloaders**:
   - Standard PyTorch dataloaders with batching
   - Configurable batch size, workers, shuffling, etc.

## Testing Your Implementation

### Step 1: Verify Dataset Files
```bash
ls proteinworkshop/data/ec_proteinshake/
# Should show: pdb/, random/, structure/, labels.csv

ls proteinworkshop/data/ec_proteinshake/random/
# Should show: train_split.csv, val_split.csv, test_split.csv

ls proteinworkshop/data/ec_proteinshake/structure/
# Should show: train_split.csv, val_split.csv, test_split.csv
```

### Step 2: Run Test Script
```bash
python test_ec_datamodule.py
```

Expected output:
- Configuration details for both splits
- Dataset sizes (train/val/test)
- First batch inspection
- Split comparison statistics
- ✓ Success messages

### Step 3: Train a Model
```bash
# Quick test with small model
workshop train \
  dataset=ec_random \
  encoder=gear_net \
  features=ca_angles \
  trainer.max_epochs=2 \
  trainer.limit_train_batches=10
```

## Troubleshooting

### Issue: "Dataset directory not found"
**Solution:** Run `create_raw_data.py` first to generate the dataset files.

### Issue: "Split directory not found"
**Solution:** Check that `random/` and `structure/` subdirectories exist with CSV files.

### Issue: "Labels file not found"
**Solution:** Ensure `labels.csv` exists in the root dataset directory.

### Issue: "Missing labels for structures"
**Solution:** Check that all PDB IDs in split CSVs have corresponding entries in `labels.csv`.

## Next Steps

1. **Test the implementation:**
   ```bash
   python test_ec_datamodule.py
   ```

2. **Train baseline models on both splits:**
   ```bash
   # Random split
   workshop train dataset=ec_random encoder=gear_net features=ca_angles
   
   # Structure split
   workshop train dataset=ec_structure encoder=gear_net features=ca_angles
   ```

3. **Compare performance:**
   - Structure split is typically harder (no similar structures in train/test)
   - Random split may have data leakage from similar structures
   - Compare metrics (accuracy, F1) between splits

4. **Add more splits (optional):**
   - Create new subdirectory (e.g., `temporal/`)
   - Add new config file (e.g., `ec_temporal.yaml`)
   - No changes to DataModule needed!

## Key Design Decisions

1. **Shared Resources**: PDB files and labels are stored once and shared across all splits
2. **Split Subdirectories**: Each split type has its own folder for train/val/test CSVs
3. **Single DataModule Class**: One class supports all split types via parameter
4. **Config Files**: Separate YAML for each split, all using same DataModule
5. **Minimal Redundancy**: No duplicate PDB processing or label storage

This design makes it easy to:
- Add new split types without changing code
- Compare models across different splits
- Minimize storage and processing overhead
