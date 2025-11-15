"""
Script to create Gene Ontology multilabel classification dataset from ProteinShake.

This script creates the go_proteinshake dataset for multilabel GO term prediction.
Unlike multiclass datasets (EC, SCOP), each protein can have MULTIPLE GO term labels.

Key differences from multiclass datasets:
- Labels are LISTS of GO terms per protein
- Labels CSV format: pdb_id,"0 5 12" (space-separated label indices)
- Uses GeneOntologyTask from ProteinShake
- Token map: {"GO:0000009": 0, "GO:0000014": 1, ...}

The script:
1. Loads GeneOntologyTask with both random and structure splits
2. Saves preprocessed PDB files (chain A only) ONCE to ./proteinworkshop/data/go_proteinshake/pdb/
3. Creates train/val/test split CSV files for BOTH split types in separate subdirectories
4. Creates a SINGLE shared labels CSV file with multilabel format

Directory structure created:
go_proteinshake/
├── pdb/                    # Shared PDB files (processed once)
├── random/                 # Random split CSVs
│   ├── train_split.csv
│   ├── val_split.csv
│   └── test_split.csv
├── structure/              # Structure-based split CSVs
│   ├── train_split.csv
│   ├── val_split.csv
│   └── test_split.csv
└── labels.csv              # Shared labels file (multilabel format)

Raw data is already saved in ./ps_raw by ProteinShake.
"""

from collections import Counter
from pathlib import Path

import pandas as pd
from proteinshake.tasks import GeneOntologyTask
from tqdm import tqdm

from visualization import protein_to_pdb

# ============================================================================
# Configuration
# ============================================================================
DATASET_NAME = "go_proteinshake"
RAW_DATA_DIR = "/data/oliver_lab/wangx86/ps_raw"  # ProteinShake raw data location
OUTPUT_BASE_DIR = f"/data/oliver_lab/wangx86/ps_data/{DATASET_NAME}"
SPLIT_TYPES = ["random", "structure"]  # Both split types to generate
SIMILARITY_THRESHOLD = 0.7  # For structure-based split

# Gene Ontology specific settings
# Options: "molecular_function", "biological_process", "cellular_component"
GO_ASPECT = "molecular_function"

# ============================================================================
# Setup directories
# ============================================================================
print(f"Setting up {DATASET_NAME} dataset with multiple split types...")
print("Description: Gene Ontology term prediction (multilabel)")
print(f"GO Aspect: {GO_ASPECT}")
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
all_labels = {}  # Dictionary to collect all labels (multilabel format)
token_map = None  # Will be populated from task

for split_type in SPLIT_TYPES:
    print("\n" + "=" * 70)
    print(f"PROCESSING {split_type.upper()} SPLIT")
    print("=" * 70)
    
    # Load ProteinShake GeneOntologyTask with current split type
    print(f"\nLoading GeneOntologyTask with {split_type} split...")
    task = GeneOntologyTask(
        split=split_type,
        split_similarity_threshold=SIMILARITY_THRESHOLD,
        root=RAW_DATA_DIR
    )
    dataset = task.dataset
    
    # Get token map from task (only on first iteration)
    if token_map is None:
        token_map = task.token_map
        print(f"Loaded token map with {len(token_map)} GO terms")
        print(f"Sample mappings: {dict(list(token_map.items())[:5])}")

    # Get split indices
    train_idx = task.train_index
    val_idx = task.val_index
    test_idx = task.test_index

    print(f"Train set size: {len(train_idx)}")
    print(f"Validation set size: {len(val_idx)}")
    print(f"Test set size: {len(test_idx)}")
    total = len(train_idx) + len(val_idx) + len(test_idx)
    print(f"Total proteins: {total}")

    # Process proteins
    print(f"\nProcessing proteins for {split_type} split...")
    protein_info = []  # List of dicts: {pdb_id, split, label_indices}
    
    protein_generator = dataset.proteins(resolution="atom")
    
    desc = f"Processing {split_type}"
    for idx, protein in enumerate(tqdm(protein_generator, desc=desc)):
        # Extract protein information
        protein_id = protein["protein"]["ID"].lower()
        
        # Extract GO terms list from the specified aspect
        # Note: Different aspects have different field names in ProteinShake
        if GO_ASPECT == "molecular_function":
            go_terms = protein["protein"].get("molecular_function", [])
        elif GO_ASPECT == "biological_process":
            go_terms = protein["protein"].get("biological_process", [])
        elif GO_ASPECT == "cellular_component":
            go_terms = protein["protein"].get("cellular_component", [])
        else:
            raise ValueError(f"Unknown GO aspect: {GO_ASPECT}")
        
        # Skip if no GO terms
        if not go_terms or len(go_terms) == 0:
            continue
        
        # Convert GO terms to label indices using token_map
        label_indices = []
        for go_term in go_terms:
            if go_term in token_map:
                label_indices.append(token_map[go_term])
            else:
                # Skip unknown GO terms (may be filtered by ProteinShake)
                pass
        
        # Skip if no valid labels after filtering
        if len(label_indices) == 0:
            continue
        
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
            "label_indices": label_indices  # List of integers
        })
        
        # Collect labels for shared labels file (convert list to space-separated string)
        all_labels[protein_id] = " ".join(map(str, label_indices))
    
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
# Create single shared labels CSV file (MULTILABEL FORMAT)
# ============================================================================
print("\n" + "=" * 70)
print("CREATING SHARED LABELS FILE (MULTILABEL)")
print("=" * 70)

# Create labels DataFrame (pdb_id -> space-separated label indices)
df_labels = pd.DataFrame(
    list(all_labels.items()), 
    columns=["pdb_id", "label"]
)
df_labels = df_labels.sort_values("pdb_id")  # Sort for easier inspection

labels_csv = Path(OUTPUT_BASE_DIR) / "labels.csv"
df_labels.to_csv(labels_csv, index=False)

print(f"✓ Saved labels.csv ({len(df_labels)} proteins)")
print('  Format: pdb_id,"0 5 12" (space-separated label indices)')

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

# Calculate multilabel statistics
print("\nMultilabel statistics:")

# Parse label strings to get label counts
label_counts_per_protein = []
all_unique_labels = set()
for label_str in df_labels["label"]:
    label_indices = [int(x) for x in label_str.split()]
    label_counts_per_protein.append(len(label_indices))
    all_unique_labels.update(label_indices)

avg_labels = sum(label_counts_per_protein) / len(label_counts_per_protein)
min_labels = min(label_counts_per_protein)
max_labels = max(label_counts_per_protein)

print(f"  Total unique GO terms: {len(all_unique_labels)}")
print(f"  Average labels per protein: {avg_labels:.2f}")
print(f"  Min labels per protein: {min_labels}")
print(f"  Max labels per protein: {max_labels}")

# Label distribution (how many proteins have each label count)
label_count_dist = Counter(label_counts_per_protein)
print("\n  Label count distribution:")
for count in sorted(label_count_dist.keys())[:10]:
    print(f"    {count} labels: {label_count_dist[count]} proteins")
if len(label_count_dist) > 10:
    print(f"    ... and {len(label_count_dist) - 10} more")

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
print("2. Create proteinworkshop/datasets/go_proteinshake.py")
print("3. Create proteinworkshop/config/dataset/go_random.yaml")
print("4. Create proteinworkshop/config/dataset/go_structure.yaml")
print("5. Run: python proteinworkshop/datasets/go_proteinshake.py (to test)")
print(
    "6. Train with: workshop train dataset=go_random "
    "task=multilabel_graph_classification"
)
print("=" * 70)
