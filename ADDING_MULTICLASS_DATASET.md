# Guide: Adding a New Labeled Graph-Level Dataset to ProteinWorkshop

This guide explains how to add a new **multi-class graph-level prediction dataset** (e.g., enzyme classification) to ProteinWorkshop.
We use the `ec_proteinshake` dataset—Enzyme Commission classification from ProteinShake—as an example.
It supports **multiple split types** (random and structure-based) while sharing the same PDB files and labels.

---

## Table of Contents

1. [Prepare the Raw Dataset](#1-prepare-the-raw-dataset)
2. [Implement the DataModule](#2-implement-the-datamodule)
3. [Configure Dataset and Task](#3-configure-dataset-and-task)

---

## 1. Prepare the Raw Dataset

### 1.1 Directory Layout

All split types (e.g., random and structure-based) should share the same PDB files and label file.
Organize the dataset as follows:

```
data/
└── ec_proteinshake/
    ├── pdb/                   # Shared PDB files (chain A only)
    │   ├── 1abc.pdb
    │   ├── 2xyz.pdb
    │   └── ...
    ├── random/                # Random split CSVs
    │   ├── train_split.csv
    │   ├── val_split.csv
    │   └── test_split.csv
    ├── structure/             # Structure-based split CSVs
    │   ├── train_split.csv
    │   ├── val_split.csv
    │   └── test_split.csv
    └── labels.csv             # Shared labels
```

**Design Principles**

* **Single `pdb/` directory** for all structures
* **Single `labels.csv`** shared across all splits
* **Split-specific folders** (`random/`, `structure/`, etc.) for train/val/test files
* **No redundancy** in structure files or labels

This organization allows efficient storage, easy addition of new split types, and clear separation of configurations.

**Notes for Preprocessed PDB Files**

* Each file contains **only chain A** → no need to specify chains in splits.
* PDBs may be `.pdb`, `.mmtf`, or `.ent` (MMTF recommended for smaller size and faster loading).
* Since structures are preprocessed, **ProteinWorkshop will not download** from RCSB PDB.

---

### 1.2 Required CSV Files

#### Split files

Store train/val/test IDs in each split folder.
For single-chain datasets, use the simple format:

```csv
pdb_id
1abc
2xyz
3efg
```

(Optional explicit chain format:)

```csv
pdb_id,chain
1abc,A
2xyz,A
3efg,A
```

#### Label file

Map each PDB ID to a class label (integer, 0-indexed):

```csv
pdb_id,label
1abc,0
2xyz,15
3efg,2
```

**Rules**

* Use a **single `labels.csv`** shared by all splits.
* Keep consistent CSV formatting.
* Chains are optional since all are “A”.

**Example summary**

```
ec_proteinshake/
├── pdb/            (1000 PDBs)
├── random/         (train/val/test)
├── structure/      (train/val/test)
└── labels.csv      (1000 entries)
```

---

## 2. Implement the DataModule

**File:** `proteinworkshop/datasets/ec_proteinshake.py`

Create a class `ECPSDataModule` that supports multiple split types using shared PDB and label files.

### 2.1 Requirements

1. **Parameter**:
   `split_type: Literal["random", "structure"]` → selects which subdirectory to load splits from.
2. **Split file paths**:
   `{path}/{split_type}/{train,val,test}_split.csv`
3. **Shared labels**:
   `{path}/labels.csv`
4. **Default PDB directory**:
   `{path}/pdb/`
5. **Automatic chain assignment**:
   All entries use chain `"A"` if not specified.

### 2.2 Suggested Structure

```python
class ECPSDataModule(ProteinDataModule):
    """Enzyme Commission classification dataset from ProteinShake."""

    def __init__(
        self,
        path: str,
        split_type: Literal["random", "structure"] = "random",
        batch_size: int = 32,
        pdb_dir: Optional[str] = None,
        format: Literal["pdb", "mmtf"] = "pdb",
        **kwargs,
    ):
        # Set paths and parameters
        # Default pdb_dir = {path}/pdb/
        # Split CSVs from {path}/{split_type}/

    def parse_labels(self) -> Dict[str, int]:
        # Load {path}/labels.csv → dict[pdb_id -> label]

    def _load_split(self, split_name: str) -> pd.DataFrame:
        # Load {split_dir}/{split_name}_split.csv
        # Add chain='A' if missing
        # Merge with labels
        # Return columns: pdb_id, chain, label

    def train_dataset(self):
        # Load and return ProteinDataset for train split

    def val_dataset(self):
        # Same for validation

    def test_dataset(self):
        # Same for test
```

**Implementation Notes**

* Inherit from `ProteinDataModule` (see `ec_reaction.py` or `cath.py` for reference).
* Use `_load_split()` for all splits to avoid duplication.
* Validate that files exist and log the split type used.

**Usage Example**

```python
# Random split
dm_random = ECPSDataModule(
    path="data/ec_proteinshake",
    split_type="random",
    batch_size=32
)

# Structure-based split
dm_structure = ECPSDataModule(
    path="data/ec_proteinshake",
    split_type="structure",
    batch_size=32
)
```

Both will use:

* Shared PDBs: `data/ec_proteinshake/pdb/`
* Shared labels: `data/ec_proteinshake/labels.csv`

---

## 3. Configure Dataset and Task

### 3.1 Data Creation Script

A helper script (`create_raw_data.py`) can generate all required files:

```bash
python create_raw_data.py
```

This script should:

1. Load ProteinShake’s EnzymeClassTask for both split types.
2. Export PDBs once to `ec_proteinshake/pdb/`.
3. Write split CSVs to `ec_proteinshake/random/` and `ec_proteinshake/structure/`.
4. Save a shared `labels.csv`.

---

### 3.2 Register the DataModule

Add the import if needed:

```python
# proteinworkshop/datasets/__init__.py
from .ec_proteinshake import ECPSDataModule
```

---

### 3.3 Config Files

Create one YAML file per split type.

**File 1:** `config/dataset/ec_random.yaml`

```yaml
datamodule:
  _target_: proteinworkshop.datasets.ec_proteinshake.ECPSDataModule
  path: ${env.paths.data}/ec_proteinshake/
  split_type: "random"
  pdb_dir: ${env.paths.data}/ec_proteinshake/pdb/
  format: "pdb"
  batch_size: 32
  num_workers: 8
  pin_memory: True
  dataset_fraction: 1.0
  shuffle_labels: False
  transforms: ${transforms}
  overwrite: False
  in_memory: True

num_classes: 7  # EC classes (0–6)
```

**File 2:** `config/dataset/ec_structure.yaml`

```yaml
datamodule:
  _target_: proteinworkshop.datasets.ec_proteinshake.ECPSDataModule
  path: ${env.paths.data}/ec_proteinshake/
  split_type: "structure"
  pdb_dir: ${env.paths.data}/ec_proteinshake/pdb/
  format: "pdb"
  batch_size: 32
  num_workers: 8
  pin_memory: True
  dataset_fraction: 1.0
  shuffle_labels: False
  transforms: ${transforms}
  overwrite: False
  in_memory: True

num_classes: 7
```

**Notes**

* Same class, same path, different `split_type`.
* Shared PDBs and labels ensure fair comparison between splits.

**Training Example**

```bash
# Random split
workshop train dataset=ec_random encoder=gear_net features=ca_angles

# Structure split
workshop train dataset=ec_structure encoder=gear_net features=ca_angles
```

---

### 3.4 Summary

| Component           | Shared Across Splits | Split-Specific            |
| ------------------- | -------------------- | ------------------------- |
| PDB files           | ✅ `pdb/`             | ❌                         |
| Labels              | ✅ `labels.csv`       | ❌                         |
| Train/Val/Test CSVs | ❌                    | ✅ `random/`, `structure/` |
| Config YAML         | ❌                    | ✅ one per split type      |

This setup keeps data consistent, minimizes redundancy, and makes it simple to compare models under different splitting schemes.

---
