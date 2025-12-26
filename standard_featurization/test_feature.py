"""
Test script for the standard featurization pipeline.

This script demonstrates the complete pipeline:
1. Load raw data created by create_raw_data_test.py
2. Create train/val/test dataloaders
3. Apply featurization to batches
4. Return ready-to-train dataloaders

Usage:
    python test_feature.py
"""
import sys
from pathlib import Path
from typing import Tuple

import hydra
import omegaconf
import torch
from graphein.protein.tensor.dataloader import ProteinDataLoader
from loguru import logger

import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from standard_featurization.datamodule import ECPSDataModule
from standard_featurization.features.factory import ProteinFeaturiser


def get_featurized_dataloaders(
    data_path: str = None,
    feature_config: str = "ca_base.yaml",
    split_type: str = "random",
    batch_size: int = 4,
    num_workers: int = 0,
) -> Tuple[ProteinDataLoader, ProteinDataLoader, ProteinDataLoader, ProteinFeaturiser]:
    """
    Create featurized dataloaders from raw protein data.
    
    Args:
        data_path: Path to dataset directory (default: ./data/ec_proteinshake)
        feature_config: Name of feature config file (e.g., "ca_base.yaml")
        split_type: Type of split ("random" or "structure")
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, featuriser)
    """
    # Set default data path
    if data_path is None:
        data_path = str(Path(__file__).parent / "data" / "ec_proteinshake")
    
    # Load feature configuration
    config_path = Path(__file__).parent / "config" / "features" / feature_config
    if not config_path.exists():
        raise FileNotFoundError(f"Feature config not found: {config_path}")
    
    logger.info(f"Loading feature config from: {config_path}")
    feature_cfg = omegaconf.OmegaConf.load(config_path)
    
    # Instantiate featuriser
    logger.info("Instantiating ProteinFeaturiser...")
    featuriser = hydra.utils.instantiate(feature_cfg)
    logger.info(f"Featuriser: {featuriser}")
    
    # Create datamodule
    logger.info(f"Creating datamodule with data from: {data_path}")
    datamodule = ECPSDataModule(
        path=data_path,
        split_type=split_type,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,  # Disable for easier debugging
    )
    
    # Setup datasets
    logger.info("Setting up datasets...")
    datamodule.setup("fit")
    datamodule.setup("test")
    
    # Get dataloaders
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    
    logger.info(f"Train loader: {len(train_loader)} batches")
    logger.info(f"Val loader: {len(val_loader)} batches")
    logger.info(f"Test loader: {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader, featuriser


def featurize_batch(batch, featuriser: ProteinFeaturiser):
    """
    Apply featurization to a batch.
    
    Args:
        batch: A batch from ProteinDataLoader
        featuriser: The ProteinFeaturiser instance
        
    Returns:
        Featurized batch
    """
    with torch.no_grad():
        featurized_batch = featuriser(batch)
    return featurized_batch


def main():
    """Main function demonstrating the featurization pipeline."""
    print("=" * 70)
    print("Standard Featurization Pipeline Test")
    print("=" * 70)
    
    # Get dataloaders and featuriser
    try:
        train_loader, val_loader, test_loader, featuriser = get_featurized_dataloaders(
            feature_config="all_invariant_ca.yaml",
            split_type="random",
            batch_size=4,
            num_workers=0,
        )
    except FileNotFoundError as e:
        logger.error(f"Data not found: {e}")
        logger.info("Please run create_raw_data_test.py first to generate test data.")
        return None
    
    print("\n" + "=" * 70)
    print("Testing Featurization on Sample Batches")
    print("=" * 70)
    
    # Test featurization on train batch
    print("\n--- Training Batch ---")
    for batch in train_loader:
        print(f"Raw batch:")
        print(f"  All raw batch keys: {batch.keys()}")
        print(batch[0])
        
        # Apply featurization
        featurized = featurize_batch(batch, featuriser)
        
        print(f"\nFeaturized batch (full):")
        print(f"  All keys: {featurized.keys()}")
        print(f"  representation used: {featuriser.representation}")
        
        # # Check critical attributes
        # print(f"\n  pos exists: {hasattr(featurized, 'pos') and featurized.pos is not None}")
        # if hasattr(featurized, 'pos') and featurized.pos is not None:
        #     print(f"  pos shape: {featurized.pos.shape}")
        
        # print(f"  edge_index exists: {hasattr(featurized, 'edge_index') and featurized.edge_index is not None}")
        # if hasattr(featurized, 'edge_index') and featurized.edge_index is not None:
        #     print(f"  edge_index shape: {featurized.edge_index.shape}")
        
        # print(f"  edge_attr exists: {hasattr(featurized, 'edge_attr') and featurized.edge_attr is not None}")
        # if hasattr(featurized, 'edge_attr') and featurized.edge_attr is not None:
        #     print(f"  edge_attr shape: {featurized.edge_attr.shape}")
        
        print(featurized.edge_attr)

        break
    
    # Test featurization on val batch
    # print("\n--- Validation Batch ---")
    # for batch in val_loader:
    #     featurized = featurize_batch(batch, featuriser)
    #     print(f"  Node features (x) shape: {featurized.x.shape}")
    #     print(f"  Edge index shape: {featurized.edge_index.shape}")
    #     break
    
    # Test featurization on test batch
    # print("\n--- Test Batch ---")
    # for batch in test_loader:
    #     featurized = featurize_batch(batch, featuriser)
    #     print(f"  Node features (x) shape: {featurized.x.shape}")
    #     print(f"  Edge index shape: {featurized.edge_index.shape}")
    #     break
    
    print("\n" + "=" * 70)
    print("Pipeline Test Complete!")
    print("=" * 70)
    
    print("\nAvailable feature configurations:")
    config_dir = Path(__file__).parent / "config" / "features"
    for config_file in sorted(config_dir.glob("*.yaml")):
        print(f"  - {config_file.name}")
    
    print("\nTo use different features, call:")
    print('  train_loader, val_loader, test_loader, featuriser = get_featurized_dataloaders(')
    print('      feature_config="all_invariant_ca.yaml"')
    print('  )')
    
    return train_loader, val_loader, test_loader, featuriser


if __name__ == "__main__":
    result = main()
    if result is not None:
        train_loader, val_loader, test_loader, featuriser = result
        print("\n✓ Dataloaders and featuriser returned successfully!")

