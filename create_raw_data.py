"""
Script to create EC dataset from ProteinShake with multiple split types.

This script:
1. Loads the EnzymeClassTask with both random and structure splits
2. Saves preprocessed PDB files (chain A only) ONCE to ./proteinworkshop/data/ec_proteinshake/pdb/
3. Creates train/val/test split CSV files for BOTH split types in separate subdirectories
4. Creates a SINGLE shared labels CSV file mapping PDB IDs to EC class labels

Directory structure created:
ec_proteinshake/
├── pdb/                    # Shared PDB files (processed once)
├── random/                 # Random split CSVs
│   ├── train_split.csv
│   ├── val_split.csv
│   └── test_split.csv
├── structure/              # Structure-based split CSVs
│   ├── train_split.csv
│   ├── val_split.csv
│   └── test_split.csv
└── labels.csv              # Shared labels file

Raw data is already saved in ./ps_raw by ProteinShake.
"""

from pathlib import Path

import pandas as pd
from proteinshake.tasks import EnzymeClassTask
from tqdm import tqdm

from visualization import protein_to_pdb

# ============================================================================
# Configuration
# ============================================================================
DATASET_NAME = "ec_proteinshake"
RAW_DATA_DIR = "./ps_raw"  # ProteinShake raw data location
OUTPUT_BASE_DIR = f"./proteinworkshop/data/{DATASET_NAME}"
SPLIT_TYPES = ["random", "structure"]  # Both split types to generate
SIMILARITY_THRESHOLD = 0.7  # For structure-based split

# ============================================================================
# Setup directories
# ============================================================================
print(f"Setting up {DATASET_NAME} dataset with multiple split types...")
print(f"Raw data location: {RAW_DATA_DIR}")
print(f"Output directory: {OUTPUT_BASE_DIR}")
print(f"Split types to generate: {SPLIT_TYPES}")

# Create output directories
pdb_save_dir = Path(OUTPUT_BASE_DIR) / "pdb"
pdb_save_dir.mkdir(parents=True, exist_ok=True)

# Create subdirectories for each split type
for split_type in SPLIT_TYPES:
    split_dir = Path(OUTPUT_BASE_DIR) / split_type
    split_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Process both split types
# ============================================================================
all_protein_info = {}  # Dictionary to store protein info for each split type
all_labels = {}  # Dictionary to collect all labels

for split_type in SPLIT_TYPES:
    print("\n" + "=" * 70)
    print(f"PROCESSING {split_type.upper()} SPLIT")
    print("=" * 70)
    
    # Load ProteinShake task with current split type
    print(f"\nLoading EnzymeClassTask with {split_type} split...")
    task = EnzymeClassTask(
        split=split_type, 
        split_similarity_threshold=SIMILARITY_THRESHOLD, 
        root=RAW_DATA_DIR
    )
    dataset = task.dataset

    # Get split indices
    train_idx = task.train_index
    val_idx = task.val_index
    test_idx = task.test_index

    print(f"Train set size: {len(train_idx)}")
    print(f"Validation set size: {len(val_idx)}")
    print(f"Test set size: {len(test_idx)}")
    print(f"Total proteins: {len(train_idx) + len(val_idx) + len(test_idx)}")

    # Process proteins
    print(f"\nProcessing proteins for {split_type} split...")
    protein_info = []  # List of dicts: {pdb_id, split, label}
    
    protein_generator = dataset.proteins(resolution="atom")
    
    desc = f"Processing {split_type}"
    for idx, protein in enumerate(tqdm(protein_generator, desc=desc)):
        # Extract protein information
        protein_id = protein["protein"]["ID"].lower()
        ec_full = protein["protein"]["EC"]
        ec_class = ec_full.split(".")[0]  # Get first digit (EC class: 1-7)
        label = int(ec_class) - 1  # Convert to 0-indexed (EC 1-7 -> 0-6)
        
        # Determine which split this protein belongs to
        if idx in train_idx:
            split_name = "train"
        elif idx in val_idx:
            split_name = "val"
        elif idx in test_idx:
            split_name = "test"
        else:
            continue  # Skip if not in any split
        
        # Save PDB file only once (on first split type iteration)
        if split_type == SPLIT_TYPES[0]:
            pdb_filename = f"{protein_id}.pdb"
            pdb_filepath = pdb_save_dir / pdb_filename
            protein_to_pdb(protein, str(pdb_filepath))
        
        # Store protein info for this split
        protein_info.append({
            "pdb_id": protein_id,
            "split": split_name,
            "label": label
        })
        
        # Collect label for shared labels file
        all_labels[protein_id] = label
    
    # Store protein info for this split type
    all_protein_info[split_type] = protein_info
    
    print(f"Processed {len(protein_info)} proteins for {split_type} split")

# ============================================================================
# Save PDB files summary
# ============================================================================
print("\n" + "=" * 70)
print("PDB FILES SAVED")
print("=" * 70)
print(f"Saved {len(all_labels)} unique PDB files to {pdb_save_dir}")

# ============================================================================
# Create split CSV files for each split type
# ============================================================================
print("\n" + "=" * 70)
print("CREATING SPLIT CSV FILES")
print("=" * 70)

for split_type in SPLIT_TYPES:
    print(f"\n--- {split_type.upper()} Split ---")
    
    df_all = pd.DataFrame(all_protein_info[split_type])
    
    # Create separate DataFrames for each split
    df_train = df_all[df_all["split"] == "train"][["pdb_id"]]
    df_val = df_all[df_all["split"] == "val"][["pdb_id"]]
    df_test = df_all[df_all["split"] == "test"][["pdb_id"]]
    
    # Save split files to split-specific subdirectory
    split_dir = Path(OUTPUT_BASE_DIR) / split_type
    train_csv = split_dir / "train_split.csv"
    val_csv = split_dir / "val_split.csv"
    test_csv = split_dir / "test_split.csv"
    
    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv, index=False)
    df_test.to_csv(test_csv, index=False)
    
    print(f"✓ Saved {split_type}/train_split.csv ({len(df_train)} proteins)")
    print(f"✓ Saved {split_type}/val_split.csv ({len(df_val)} proteins)")
    print(f"✓ Saved {split_type}/test_split.csv ({len(df_test)} proteins)")

# ============================================================================
# Create single shared labels CSV file
# ============================================================================
print("\n" + "=" * 70)
print("CREATING SHARED LABELS FILE")
print("=" * 70)

# Create labels DataFrame (pdb_id -> label) from all collected labels
df_labels = pd.DataFrame(
    list(all_labels.items()), 
    columns=["pdb_id", "label"]
)
df_labels = df_labels.sort_values("pdb_id")  # Sort for easier inspection

labels_csv = Path(OUTPUT_BASE_DIR) / "labels.csv"
df_labels.to_csv(labels_csv, index=False)

print(f"✓ Saved labels.csv ({len(df_labels)} proteins)")

# ============================================================================
# Print summary statistics
# ============================================================================
print("\n" + "=" * 70)
print("DATASET CREATION SUMMARY")
print("=" * 70)

print(f"\nOutput directory: {OUTPUT_BASE_DIR}")
print(f"  ├── pdb/ ({len(all_labels)} PDB files)")
for split_type in SPLIT_TYPES:
    df_split = pd.DataFrame(all_protein_info[split_type])
    df_train = df_split[df_split["split"] == "train"]
    df_val = df_split[df_split["split"] == "val"]
    df_test = df_split[df_split["split"] == "test"]
    print(f"  ├── {split_type}/")
    print(f"  │   ├── train_split.csv ({len(df_train)} proteins)")
    print(f"  │   ├── val_split.csv ({len(df_val)} proteins)")
    print(f"  │   └── test_split.csv ({len(df_test)} proteins)")
print(f"  └── labels.csv ({len(df_labels)} labels)")

print("\nLabel distribution (shared across all splits):")
label_counts = df_labels["label"].value_counts().sort_index()
for label, count in label_counts.items():
    ec_num = label + 1  # Convert back to EC class (1-7)
    print(f"  EC Class {ec_num} (label {label}): {count} proteins")

print(f"\nNumber of classes: {df_labels['label'].nunique()}")
print(f"Label range: {df_labels['label'].min()} - {df_labels['label'].max()}")

# Print split comparison
print("\nSplit type comparison:")
for split_type in SPLIT_TYPES:
    df_split = pd.DataFrame(all_protein_info[split_type])
    total = len(df_split)
    train_count = len(df_split[df_split["split"] == "train"])
    val_count = len(df_split[df_split["split"] == "val"])
    test_count = len(df_split[df_split["split"] == "test"])
    print(f"  {split_type}: {total} total "
          f"(train={train_count}, val={val_count}, test={test_count})")

print("\n✓ Dataset creation complete!")
print("\nNext steps:")
print(f"1. Verify files in {OUTPUT_BASE_DIR}/")
print("2. Create proteinworkshop/datasets/ec_proteinshake.py")
print("3. Create proteinworkshop/config/dataset/ec_random.yaml")
print("4. Create proteinworkshop/config/dataset/ec_structure.yaml")
print("5. Run: python proteinworkshop/datasets/ec_proteinshake.py (to test)")
print("=" * 70)
