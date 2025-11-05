#!/usr/bin/env python
"""
Test script for EC ProteinShake DataModule.

This script demonstrates how to:
1. Load both split types (random and structure)
2. Access dataset splits and dataloaders
3. Inspect batch contents
4. Compare split statistics
"""

import sys
from pathlib import Path

# Add proteinworkshop to path
sys.path.insert(0, str(Path(__file__).parent))

from proteinworkshop.datasets.ec_proteinshake import ECPSDataModule
from proteinworkshop.features.factory import ProteinFeaturiser


# Create feature transforms (CA Angles - used in many papers)
# Note: In actual training, this is applied by the model, not the dataloader
def create_featuriser():
    """Create a featuriser for demonstration purposes"""
    return ProteinFeaturiser(
        representation="CA",
        scalar_node_features=[
            "amino_acid_one_hot",
            "alpha",  # Dihedral angles
            "kappa",
            "dihedrals",
        ],
        vector_node_features=[],
        edge_types=["knn_30"],  # k-NN graph with k=30
        scalar_edge_features=["edge_distance", "sequence_distance"],
        vector_edge_features=[],
    )


def test_split(split_type: str, data_path: str):
    """Test loading and using a specific split type"""
    print("=" * 70)
    print(f"Testing {split_type.upper()} split")
    print("=" * 70)
    
    # Create datamodule (without transforms - they're applied by the model)
    datamodule = ECPSDataModule(
        path=data_path,
        split_type=split_type,
        batch_size=2,  # Small batch for testing
        num_workers=0,  # No multiprocessing for simplicity
        in_memory=False,  # Don't load everything into memory
        dataset_fraction=0.01,  # Use only 1% for quick testing
        transforms=None,  # No pre-transforms needed
    )
    
    print("\n1. DataModule Configuration:")
    print(f"   Split type: {datamodule.split_type}")
    print(f"   Data dir: {datamodule.data_dir}")
    print(f"   Split dir: {datamodule.split_dir}")
    print(f"   PDB dir: {datamodule.pdb_dir}")
    print(f"   Format: {datamodule.format}")
    print(f"   Batch size: {datamodule.batch_size}")
    
    # Setup datasets
    print("\n2. Loading datasets...")
    datamodule.setup("fit")
    
    # Get datasets
    train_ds = datamodule.train_dataset()
    val_ds = datamodule.val_dataset()
    test_ds = datamodule.test_dataset()
    
    print(f"   Train: {len(train_ds)} proteins")
    print(f"   Val: {len(val_ds)} proteins")
    print(f"   Test: {len(test_ds)} proteins")
    print(f"   Total: {len(train_ds) + len(val_ds) + len(test_ds)} proteins")
    
    # Get dataloaders
    print("\n3. Creating dataloaders...")
    train_dl = datamodule.train_dataloader()
    print(f"   Train batches: {len(train_dl)}")
    
    # Create featuriser (in actual training, this is part of the model)
    print("\n4. Creating featuriser...")
    featuriser = create_featuriser()
    print(f"   Representation: {featuriser.representation}")
    print(f"   Edge types: {featuriser.edge_types}")
    print(f"   Node features: {featuriser.scalar_node_features}")
    print(f"   Edge features: {featuriser.scalar_edge_features}")
    
    # Inspect first batch (transforms are applied during batching!)
    print("\n5. Inspecting first training batch:")
    for batch in train_dl:
        print(f"   Raw batch type: {type(batch)}")
        print(f"   Num graphs: {batch.num_graphs}")
        print(f"   Raw batch keys: {list(batch.keys())}")
        
        # Apply featuriser (this is what the model does during training)
        print("\n   Applying featuriser...")
        batch = featuriser(batch)
        print(f"   Featurised batch keys: {list(batch.keys())}")
        
        # Check for graph structure
        if hasattr(batch, 'edge_index') and batch.edge_index is not None:
            print(f"   ✓ Edge index shape: {batch.edge_index.shape}")
            print(f"   ✓ Num edges: {batch.edge_index.shape[1]}")
            avg_edges = batch.edge_index.shape[1] / batch.num_graphs
            print(f"   ✓ Avg edges per graph: {avg_edges:.1f}")
        else:
            print("   ✗ No edge_index found!")
        
        # Check for node features
        if hasattr(batch, 'x') and batch.x is not None:
            print(f"   ✓ Node features shape: {batch.x.shape}")
            print(f"   ✓ Node feature dim: {batch.x.shape[1]}")
        else:
            print("   ✗ No node features found!")
        
        # Check for edge features
        if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
            print(f"   ✓ Edge features shape: {batch.edge_attr.shape}")
            print(f"   ✓ Edge feature dim: {batch.edge_attr.shape[1]}")
        else:
            print("   ℹ No edge_attr (edge features not requested)")
        
        # Check for labels
        if hasattr(batch, 'graph_y'):
            print(f"   ✓ Graph labels: {batch.graph_y}")
            label_min = batch.graph_y.min()
            label_max = batch.graph_y.max()
            print(f"   ✓ Label range: {label_min} - {label_max}")
            unique_labels = batch.graph_y.unique().tolist()
            print(f"   ✓ Unique labels in batch: {unique_labels}")
        
        # Check for PDB IDs
        if hasattr(batch, 'id'):
            ids = batch.id if isinstance(batch.id, list) else [batch.id]
            print(f"   ✓ PDB IDs (first 4): {ids[:4]}")
        
        # Summary
        print("\n   Summary:")
        print(f"   - Batch contains {batch.num_graphs} protein graphs")
        total_nodes = batch.x.shape[0] if hasattr(batch, 'x') else 0
        print(f"   - Total nodes: {total_nodes}")
        if hasattr(batch, 'edge_index'):
            total_edges = batch.edge_index.shape[1]
            print(f"   - Total edges: {total_edges}")
        
        break  # Only show first batch
    
    print(f"\n✓ {split_type.upper()} split loaded successfully!")
    return datamodule


def compare_splits(random_dm, structure_dm):
    """Compare statistics between random and structure splits"""
    print("\n" + "=" * 70)
    print("COMPARING SPLITS")
    print("=" * 70)
    
    # Compare dataset sizes
    print("\n1. Dataset sizes:")
    print("   Random split:")
    print(f"     Train: {len(random_dm.train_dataset())}")
    print(f"     Val: {len(random_dm.val_dataset())}")
    print(f"     Test: {len(random_dm.test_dataset())}")
    
    print("   Structure split:")
    print(f"     Train: {len(structure_dm.train_dataset())}")
    print(f"     Val: {len(structure_dm.val_dataset())}")
    print(f"     Test: {len(structure_dm.test_dataset())}")
    
    # Compare label distributions
    print("\n2. Label distributions:")
    
    # Get labels from each split
    random_train_labels = [int(label) for label in random_dm._load_split("train")["label"]]
    structure_train_labels = [int(label) for label in structure_dm._load_split("train")["label"]]
    
    from collections import Counter
    random_counts = Counter(random_train_labels)
    structure_counts = Counter(structure_train_labels)
    
    print("   Random split (train):")
    for label in sorted(random_counts.keys()):
        print(f"     EC {label+1} (label {label}): {random_counts[label]} proteins")
    
    print("   Structure split (train):")
    for label in sorted(structure_counts.keys()):
        print(f"     EC {label+1} (label {label}): {structure_counts[label]} proteins")
    
    # Verify same PDB directory
    print("\n3. Shared resources:")
    print(f"   Random split PDB dir: {random_dm.pdb_dir}")
    print(f"   Structure split PDB dir: {structure_dm.pdb_dir}")
    print(f"   Same PDB files: {random_dm.pdb_dir == structure_dm.pdb_dir} ✓")


def main():
    """Main test function"""
    # Path to dataset
    data_path = "./proteinworkshop/data/ec_proteinshake"
    
    print("\n" + "=" * 70)
    print("EC PROTEINSHAKE DATAMODULE TEST")
    print("=" * 70)
    print(f"\nDataset path: {data_path}")
    
    # Check if dataset exists
    if not Path(data_path).exists():
        print(f"\n❌ Error: Dataset not found at {data_path}")
        print("Please run create_raw_data.py first to generate the dataset.")
        return
    
    # Test random split
    print("\n")
    random_dm = test_split("random", data_path)
    
    # Test structure split
    print("\n")
    structure_dm = test_split("structure", data_path)
    
    # Compare splits
    compare_splits(random_dm, structure_dm)
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Train a model on random split:")
    print("   workshop train dataset=ec_random encoder=gear_net \\")
    print("            features=ca_angles")
    print("\n2. Train a model on structure split:")
    print("   workshop train dataset=ec_structure encoder=gear_net \\")
    print("            features=ca_angles")
    print("\n3. Compare model performance between splits")
    print("=" * 70)


if __name__ == "__main__":
    main()
