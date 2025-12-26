"""
Standard Featurization Module for Protein Structure Data.

This module provides a self-contained pipeline for featurizing protein
structures from raw PDB data to train/val/test dataloaders.
"""

from .datamodule import ECPSDataModule, ProteinDataset
from .features.factory import ProteinFeaturiser
from .test_feature import get_featurized_dataloaders, featurize_batch

__all__ = [
    "ECPSDataModule",
    "ProteinDataset",
    "ProteinFeaturiser",
    "get_featurized_dataloaders",
    "featurize_batch",
]

