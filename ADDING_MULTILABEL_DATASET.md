# Guide: Adding a New Multilabel Graph-Level Dataset to ProteinWorkshop

This guide explains how to add a new **multilabel graph-level prediction dataset** (e.g., Gene Ontology classification) to ProteinWorkshop.
We use the `go_proteinshake` dataset—Gene Ontology term prediction from ProteinShake—as an example.
It supports **multiple split types** (random and structure-based) while sharing the same PDB files and labels.

**Key Difference from Multiclass:** In multilabel classification, each protein can be assigned **multiple labels** simultaneously (e.g., a protein may have GO terms from multiple functional categories), whereas in multiclass classification each protein has exactly one label.

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
└── go_proteinshake/
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
    └── labels.csv             # Shared labels (multilabel format)
```

**Design Principles**

* **Single `pdb/` directory** for all structures
* **Single `labels.csv`** shared across all splits (with multilabel format)
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

#### Label file (Multilabel Format)

**Critical difference:** For multilabel datasets, each PDB ID maps to a **list of label indices** (space-separated or comma-separated integers):

```csv
pdb_id,label
1abc,"0 5 12"
2xyz,"3 7 15 22"
3efg,"1 2"
```

**Alternative format (comma-separated):**

```csv
pdb_id,label
1abc,"0,5,12"
2xyz,"3,7,15,22"
3efg,"1,2"
```

**Rules**

* Use a **single `labels.csv`** shared by all splits.
* Each protein can have **multiple labels** (indices separated by spaces or commas).
* Labels are 0-indexed integers representing different GO terms (or other category IDs).
* Empty labels are possible: use empty string `""` or omit the protein from the file.
* Keep consistent CSV formatting and quoting for lists.

**Example summary**

```
go_proteinshake/
├── pdb/            (e.g., 5000 PDBs)
├── random/         (train/val/test)
├── structure/      (train/val/test)
└── labels.csv      (5000 entries, each with 1+ labels)
```

---

## 2. Implement the DataModule

**File:** `proteinworkshop/datasets/go_proteinshake.py`

Create a class `GOPSDataModule` that supports multiple split types using shared PDB and label files.

### 2.1 Requirements

1. **Parameter**:
   `split_type: Literal["random", "structure"]` → selects which subdirectory to load splits from.
2. **Split file paths**:
   `{path}/{split_type}/{train,val,test}_split.csv`
3. **Shared labels**:
   `{path}/labels.csv` (multilabel format)
4. **Default PDB directory**:
   `{path}/pdb/`
5. **Automatic chain assignment**:
   All entries use chain `"A"` if not specified.
6. **Label parsing**:
   Parse string of space/comma-separated integers into a list or tensor of label indices.

### 2.2 Key Differences from Multiclass Implementation

The main difference is in how labels are handled:

**Multiclass (EC dataset):**
```python
# labels.csv: pdb_id,label
# 1abc,0
# Parse: label_dict["1abc"] = 0 (single integer)

# In dataset creation:
graph_labels=[torch.tensor(label) for label in df["label"]]
# Results in: torch.tensor(0), torch.tensor(5), etc.
```

**Multilabel (GO dataset):**
```python
# labels.csv: pdb_id,label
# 1abc,"0 5 12"
# Parse: label_dict["1abc"] = [0, 5, 12] (list of integers)

# In dataset creation:
graph_labels=[torch.tensor(label) for label in df["label"]]
# Results in: torch.tensor([0, 5, 12]), torch.tensor([3, 7]), etc.
```

### 2.3 Suggested Structure

```python
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Literal, Optional

import omegaconf
import pandas as pd
import torch
from graphein.protein.tensor.dataloader import ProteinDataLoader
from loguru import logger

from proteinworkshop.datasets.base import ProteinDataModule, ProteinDataset


class GOPSDataModule(ProteinDataModule):
    """Gene Ontology classification dataset from ProteinShake."""

    def __init__(
        self,
        path: str,
        split_type: Literal["random", "structure"] = "random",
        batch_size: int = 32,
        pdb_dir: Optional[str] = None,
        format: Literal["pdb", "mmtf"] = "pdb",
        in_memory: bool = False,
        pin_memory: bool = True,
        num_workers: int = 16,
        dataset_fraction: float = 1.0,
        shuffle_labels: bool = False,
        transforms: Optional[Iterable[Callable]] = None,
        overwrite: bool = False,
    ) -> None:
        super().__init__()

        self.data_dir = Path(path)
        self.split_type = split_type

        # Set default PDB directory to {path}/pdb/ if not provided
        if pdb_dir is None:
            self.pdb_dir = str(self.data_dir / "pdb")
        else:
            self.pdb_dir = pdb_dir

        # Set split directory based on split_type
        self.split_dir = self.data_dir / split_type

        # Validate that directories exist
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")
        if not os.path.exists(self.split_dir):
            available_splits = [
                d.name for d in self.data_dir.iterdir() 
                if d.is_dir() and d.name not in ['pdb', 'raw', 'processed']
            ]
            raise FileNotFoundError(
                f"Split directory not found: {self.split_dir}. "
                f"Available split types: {available_splits}"
            )

        # Setup transforms
        if transforms is not None:
            if omegaconf.OmegaConf.is_config(transforms):
                transforms_list = omegaconf.OmegaConf.to_container(
                    transforms, resolve=True
                )
            else:
                transforms_list = transforms
            self.transform = self.compose_transforms(transforms_list)
        else:
            self.transform = None

        # Store parameters
        self.batch_size = batch_size
        self.format = format
        self.in_memory = in_memory
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.dataset_fraction = dataset_fraction
        self.shuffle_labels = shuffle_labels
        self.overwrite = overwrite

        self.prepare_data_per_node = True

        logger.info(
            f"Setting up GO ProteinShake dataset with {split_type} split. "
            f"Fraction: {self.dataset_fraction}"
        )

    def download(self):
        """Not needed - data should be generated by create_raw_data.py"""
        pass

    def exclude_pdbs(self):
        """Not needed for preprocessed data"""
        return []

    def parse_dataset(self, split: str) -> pd.DataFrame:
        """
        Parse dataset for a given split.

        Args:
            split: "train", "val", or "test"

        Returns:
            DataFrame with columns: pdb_id, chain, label
        """
        return self._load_split(split)

    def parse_labels(self) -> Dict[str, List[int]]:
        """
        Load multilabel labels from {path}/labels.csv

        Returns:
            Dict mapping pdb_id -> list of label indices
        """
        labels_file = self.data_dir / "labels.csv"

        if not os.path.exists(labels_file):
            raise FileNotFoundError(
                f"Labels file not found: {labels_file}. "
                "Please run create_raw_data.py to generate dataset files."
            )

        df = pd.read_csv(labels_file)

        # Handle format: pdb_id, label
        if "pdb_id" not in df.columns or "label" not in df.columns:
            raise ValueError(
                f"Invalid labels file format. Expected columns: pdb_id, label"
            )

        label_dict = {}
        for _, row in df.iterrows():
            pdb_id = row["pdb_id"]
            label_str = str(row["label"]).strip()
            
            # Parse space or comma-separated integers
            if label_str == "" or label_str == "nan":
                # Handle empty labels
                label_indices = []
            else:
                # Try comma-separated first, then space-separated
                if "," in label_str:
                    label_indices = [int(x.strip()) for x in label_str.split(",")]
                else:
                    label_indices = [int(x.strip()) for x in label_str.split()]
            
            label_dict[pdb_id] = label_indices

        logger.info(f"Loaded {len(label_dict)} multilabel entries from {labels_file}")
        
        # Calculate statistics
        total_labels = sum(len(labels) for labels in label_dict.values())
        avg_labels = total_labels / len(label_dict) if label_dict else 0
        max_labels = max((len(labels) for labels in label_dict.values()), default=0)
        all_label_indices = set(idx for labels in label_dict.values() for idx in labels)
        
        logger.info(
            f"Label statistics: "
            f"avg={avg_labels:.2f} labels/protein, "
            f"max={max_labels}, "
            f"unique_labels={len(all_label_indices)}"
        )

        return label_dict

    def _load_split(self, split_name: str) -> pd.DataFrame:
        """
        Load split from {split_dir}/{split_name}_split.csv

        Args:
            split_name: "train", "val", or "test"

        Returns:
            DataFrame with columns: pdb_id, chain, label (list of ints)
        """
        split_file = self.split_dir / f"{split_name}_split.csv"

        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"Split file not found: {split_file}. "
                "Please run create_raw_data.py to generate dataset files."
            )

        # Load split CSV
        df = pd.read_csv(split_file)

        # Handle both formats: simple (pdb_id only) or explicit (pdb_id, chain)
        if "pdb_id" in df.columns:
            if "chain" not in df.columns:
                df["chain"] = "A"
        elif len(df.columns) == 1:
            df.columns = ["pdb_id"]
            df["chain"] = "A"
        elif len(df.columns) == 2:
            df.columns = ["pdb_id", "chain"]
        else:
            raise ValueError(
                f"Invalid split file format. Expected 1-2 columns, got {len(df.columns)}"
            )

        # Load labels
        label_dict = self.parse_labels()

        # Merge with labels
        df["label"] = df["pdb_id"].map(label_dict)

        # Check for missing labels
        missing_labels = df[df["label"].isna()]
        if len(missing_labels) > 0:
            logger.warning(
                f"Missing labels for {len(missing_labels)} structures: "
                f"{missing_labels['pdb_id'].tolist()[:10]}..."
            )
            df = df.dropna(subset=["label"])

        # Shuffle labels if requested (for negative control)
        if self.shuffle_labels:
            logger.warning("Shuffling labels for negative control experiment!")
            df["label"] = df["label"].sample(frac=1.0).values

        # Apply dataset fraction
        if self.dataset_fraction < 1.0:
            original_size = len(df)
            df = df.sample(frac=self.dataset_fraction, random_state=42)
            logger.info(
                f"Using {self.dataset_fraction:.1%} of {split_name} split: "
                f"{len(df)}/{original_size} proteins"
            )

        # Count unique labels in split
        all_labels = set(idx for labels in df["label"] for idx in labels)
        avg_labels_per_protein = sum(len(labels) for labels in df["label"]) / len(df)
        
        logger.info(
            f"Loaded {split_name} split ({self.split_type}): "
            f"{len(df)} proteins, "
            f"{len(all_labels)} unique labels, "
            f"avg {avg_labels_per_protein:.2f} labels/protein"
        )

        return df

    def train_dataset(self) -> ProteinDataset:
        """Load training split"""
        df = self._load_split("train")

        return ProteinDataset(
            root=str(self.data_dir),
            pdb_dir=self.pdb_dir,
            pdb_codes=list(df["pdb_id"]),
            chains=list(df["chain"]),
            # Important: Pass list of tensors, each containing multiple label indices
            graph_labels=[torch.tensor(label) for label in df["label"]],
            transform=self.transform,
            format=self.format,
            in_memory=self.in_memory,
            overwrite=self.overwrite,
        )

    def val_dataset(self) -> ProteinDataset:
        """Load validation split"""
        df = self._load_split("val")

        return ProteinDataset(
            root=str(self.data_dir),
            pdb_dir=self.pdb_dir,
            pdb_codes=list(df["pdb_id"]),
            chains=list(df["chain"]),
            graph_labels=[torch.tensor(label) for label in df["label"]],
            transform=self.transform,
            format=self.format,
            in_memory=self.in_memory,
            overwrite=self.overwrite,
        )

    def test_dataset(self) -> ProteinDataset:
        """Load test split"""
        df = self._load_split("test")

        return ProteinDataset(
            root=str(self.data_dir),
            pdb_dir=self.pdb_dir,
            pdb_codes=list(df["pdb_id"]),
            chains=list(df["chain"]),
            graph_labels=[torch.tensor(label) for label in df["label"]],
            transform=self.transform,
            format=self.format,
            in_memory=self.in_memory,
            overwrite=self.overwrite,
        )

    def train_dataloader(self) -> ProteinDataLoader:
        """Training dataloader"""
        if not hasattr(self, "train_ds"):
            self.train_ds = self.train_dataset()
        return ProteinDataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> ProteinDataLoader:
        """Validation dataloader"""
        if not hasattr(self, "val_ds"):
            self.val_ds = self.val_dataset()
        return ProteinDataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> ProteinDataLoader:
        """Test dataloader"""
        if not hasattr(self, "test_ds"):
            self.test_ds = self.test_dataset()
        return ProteinDataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
```

**Implementation Notes**

* Inherit from `ProteinDataModule` (see `go.py` or `ec_proteinshake.py` for reference).
* The key difference is in `parse_labels()`: return `Dict[str, List[int]]` instead of `Dict[str, int]`.
* In dataset creation methods, convert label lists to tensors: `[torch.tensor(label) for label in df["label"]]`.
* Handle both comma and space-separated label formats for flexibility.

---

## 3. Configure Dataset and Task

### 3.1 Data Creation Script

Modify `create_raw_data.py` to support GO dataset generation:

**Add to `DATASET_CONFIGS` dictionary:**

```python
DATASET_CONFIGS = {
    "ec_proteinshake": {
        # ... existing config ...
    },
    "scop_proteinshake": {
        # ... existing config ...
    },
    "go_proteinshake": {
        "task_class": GeneOntologyTask,  # Import from proteinshake.tasks
        "label_field": "GO",  # Adjust based on ProteinShake's field name
        "label_extractor": lambda x: x,  # GO labels are already lists
        "description": "Gene Ontology term prediction"
    }
}
```

**Update label processing for multilabel format:**

The script needs to handle the fact that GO labels are **lists** rather than single values:

```python
# In the protein processing loop:
label_value = protein["protein"][dataset_config["label_field"]]
# label_value is now a list like ["GO:0001", "GO:0023", ...]

# Convert GO terms to indices using token_map
label_indices = [token_map[term] for term in label_value if term in token_map]

# Store as space-separated string for CSV
protein_info.append({
    "pdb_id": protein_id,
    "split": split_name,
    "label": " ".join(map(str, label_indices))  # e.g., "0 5 12"
})
```

**Run the script:**

```bash
# Edit create_raw_data.py to set DATASET_NAME = "go_proteinshake"
python create_raw_data.py
```

This will create:
- `go_proteinshake/pdb/` with shared PDB files
- `go_proteinshake/random/` and `go_proteinshake/structure/` with split CSVs
- `go_proteinshake/labels.csv` with multilabel format

---

### 3.2 Register the DataModule

Add the import to make the DataModule discoverable:

```python
# proteinworkshop/datasets/__init__.py
from .go_proteinshake import GOPSDataModule
```

---

### 3.3 Config Files

Create one YAML file per split type. **Crucially, use the `multilabel_graph_classification` task.**

**File 1:** `config/dataset/go_random.yaml`

```yaml
name: "go_random"  # Dataset name for logging and output directories

datamodule:
  _target_: proteinworkshop.datasets.go_proteinshake.GOPSDataModule
  path: ${env.paths.data}/go_proteinshake/ # Directory where the dataset is stored
  split_type: "random" # Type of split to use (random or structure)
  pdb_dir: ${env.paths.data}/go_proteinshake/pdb/ # Directory where raw PDB files are stored
  format: "pdb" # Format of the raw PDB files
  batch_size: 32 # Batch size for dataloader
  pin_memory: True # Pin memory for dataloader
  num_workers: 8 # Number of workers for dataloader
  dataset_fraction: 1.0 # Fraction of the dataset to use
  shuffle_labels: False # Whether to shuffle labels for permutation testing
  transforms: ${transforms} # Transforms to apply to dataset examples
  overwrite: False # Whether to overwrite cached processed files
  in_memory: False # Load entire dataset into memory

num_classes: 489  # Number of unique GO terms (adjust based on your actual data)
```

**File 2:** `config/dataset/go_structure.yaml`

```yaml
name: "go_structure"  # Dataset name for logging and output directories

datamodule:
  _target_: proteinworkshop.datasets.go_proteinshake.GOPSDataModule
  path: ${env.paths.data}/go_proteinshake/ # Directory where the dataset is stored
  split_type: "structure" # Type of split to use (random or structure)
  pdb_dir: ${env.paths.data}/go_proteinshake/pdb/ # Directory where raw PDB files are stored
  format: "pdb" # Format of the raw PDB files
  batch_size: 32 # Batch size for dataloader
  pin_memory: True # Pin memory for dataloader
  num_workers: 8 # Number of workers for dataloader
  dataset_fraction: 1.0 # Fraction of the dataset to use
  shuffle_labels: False # Whether to shuffle labels for permutation testing
  transforms: ${transforms} # Transforms to apply to dataset examples
  overwrite: False # Whether to overwrite cached processed files
  in_memory: False # Load entire dataset into memory

num_classes: 489  # Number of unique GO terms (adjust based on your actual data)
```

**Notes**

* Same class, same path, different `split_type`.
* `num_classes` should be the **total number of unique GO terms** across all proteins.
* Shared PDBs and labels ensure fair comparison between splits.

---

### 3.4 Task Configuration

**Critical:** Use the existing `multilabel_graph_classification` task config.

The task is already defined at `proteinworkshop/config/task/multilabel_graph_classification.yaml`:

```yaml
# @package _global_

defaults:
  - override /metrics:
      - accuracy
      - f1_score
      - f1_max
      - rocauc
      - auprc
  - override /decoder:
      - graph_label
  - override /transforms:
    - remove_missing_ca
    - multihot_label_encoding  # This transform converts label indices to multihot vectors

metrics:
  accuracy:
    num_labels: ${dataset.num_classes}
  f1_score:
    num_labels: ${dataset.num_classes}
  rocauc:
    num_labels: ${dataset.num_classes}

callbacks:
  early_stopping:
    monitor: val/graph_label/accuracy
    mode: "max"
  model_checkpoint:
    monitor: val/graph_label/accuracy
    mode: "max"

task:
  task: "classification"
  classification_type: "multilabel"  # Important!
  metric_average: "micro"
  
  losses:
    graph_label: bce  # Binary cross-entropy for multilabel
  label_smoothing: 0.0

  output:
    - "graph_label"
  supervise_on:
    - "graph_label"
```

**Key components:**

1. **`multihot_label_encoding` transform**: Converts label indices `[0, 5, 12]` to a multihot vector `[1, 0, 0, 0, 0, 1, ..., 1, ...]` of shape `[1, num_classes]`.
   
2. **BCE loss**: Binary cross-entropy is used instead of cross-entropy for multilabel.

3. **Multilabel metrics**: Accuracy, F1, ROC-AUC, and AUPRC are configured for multilabel tasks.

**Training Example**

```bash
# Random split
workshop train \
  dataset=go_random \
  task=multilabel_graph_classification \
  encoder=gear_net \
  features=ca_angles

# Structure split
workshop train \
  dataset=go_structure \
  task=multilabel_graph_classification \
  encoder=gear_net \
  features=ca_angles
```

---

### 3.5 Comparison Table: Multiclass vs Multilabel

| Component                  | Multiclass (EC)           | Multilabel (GO)                    |
| -------------------------- | ------------------------- | ---------------------------------- |
| Labels per protein         | 1                         | 1 or more                          |
| Label format in CSV        | `"0"` (single int)        | `"0 5 12"` (space/comma-separated) |
| Python label type          | `int`                     | `List[int]`                        |
| PyTorch label tensor       | `torch.tensor(0)`         | `torch.tensor([0, 5, 12])`         |
| Transform                  | None (or custom)          | `multihot_label_encoding`          |
| Final label shape          | `[batch_size]`            | `[batch_size, num_classes]`        |
| Loss function              | Cross-entropy             | Binary cross-entropy (BCE)         |
| Task config                | `graph_classification`    | `multilabel_graph_classification`  |
| Metrics                    | Accuracy, macro F1        | Micro/macro F1, ROC-AUC, AUPRC     |
| ProteinWorkshop example    | `ec_proteinshake.py`      | `go.py`                            |

---

### 3.6 Summary

| Component           | Shared Across Splits | Split-Specific            |
| ------------------- | -------------------- | ------------------------- |
| PDB files           | ✅ `pdb/`             | ❌                         |
| Labels              | ✅ `labels.csv`       | ❌                         |
| Train/Val/Test CSVs | ❌                    | ✅ `random/`, `structure/` |
| Config YAML         | ❌                    | ✅ one per split type      |

**Label Format Example:**

```csv
pdb_id,label
1abc,"0 5 12"
2xyz,"3"
3def,"1 2 7 15"
```

This setup keeps data consistent, minimizes redundancy, and makes it simple to compare models under different splitting schemes while handling multilabel scenarios correctly.

---

## 4. Testing the Implementation

After implementation, test your datamodule:

```python
# Add to go_proteinshake.py at the end:
if __name__ == "__main__":
    """Quick test of the datamodule"""
    import pathlib
    import hydra
    import omegaconf
    from proteinworkshop import constants

    # Test random split
    print("=" * 70)
    print("Testing GO ProteinShake DataModule - Random Split")
    print("=" * 70)

    cfg_random = omegaconf.OmegaConf.load(
        constants.SRC_PATH / "config" / "dataset" / "go_random.yaml"
    )
    cfg_random.datamodule.path = pathlib.Path(constants.DATA_PATH) / "go_proteinshake"
    cfg_random.datamodule.transforms = []

    ds_random = hydra.utils.instantiate(cfg_random)
    datamodule_random = ds_random["datamodule"]

    print("\nDataModule info:")
    print(f"  Split type: {datamodule_random.split_type}")
    print(f"  Num classes: {cfg_random.num_classes}")

    # Test loading
    train_dl = datamodule_random.train_dataloader()

    print("\nFirst batch:")
    for batch in train_dl:
        print(f"  Num graphs: {batch.num_graphs}")
        print(f"  Graph labels shape: {batch.graph_label.shape}")
        print(f"  Graph labels dtype: {batch.graph_label.dtype}")
        print(f"  Example label (first protein): {batch.graph_label[0]}")
        print(f"  Num positive labels: {batch.graph_label[0].sum()}")
        break

    print("\n✓ Random split loaded successfully!")
```

Run the test:

```bash
python proteinworkshop/datasets/go_proteinshake.py
```

---

## 5. Summary

This guide covered:

1. **Data preparation**: Creating multilabel CSV files with space/comma-separated label indices
2. **DataModule implementation**: Parsing multilabel labels and converting them to tensors
3. **Configuration**: Using the `multilabel_graph_classification` task with `multihot_label_encoding` transform
4. **Key differences**: Understanding how multilabel differs from multiclass in terms of data format, transforms, and loss functions

Next steps:
1. Modify `create_raw_data.py` to support GO dataset
2. Implement `proteinworkshop/datasets/go_proteinshake.py`
3. Create config files `go_random.yaml` and `go_structure.yaml`
4. Test the implementation
5. Train models using `task=multilabel_graph_classification`

**Key Takeaway:** The main adaptations for multilabel are: (1) storing labels as lists, (2) using the `multihot_label_encoding` transform, (3) using the `multilabel_graph_classification` task config with BCE loss.
