"""
Simplified DataModule and Dataset for protein structure data.
Adapted from proteinworkshop for standalone use in standard_featurization.
"""
import copy
import os
import pathlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import lightning as L
import omegaconf
import pandas as pd
import torch
import torch.nn.functional as F
from beartype import beartype as typechecker
from graphein import verbose
from graphein.protein.tensor.dataloader import ProteinDataLoader
from graphein.protein.tensor.io import protein_to_pyg
from loguru import logger
from torch_geometric import transforms as T
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

verbose(False)


def amino_acid_one_hot(
    x: Union[Data, Any], num_classes: int = 23
) -> torch.Tensor:
    """Returns one-hot encoding of amino acid sequence."""
    return F.one_hot(x.residue_type, num_classes=num_classes).float()


class ProteinDataset(Dataset):
    """Dataset for loading protein structures.

    :param pdb_codes: List of PDB codes to load.
    :type pdb_codes: List[str]
    :param root: Path to root directory.
    :type root: Optional[str]
    :param pdb_dir: Path to directory containing raw PDB files.
    :type pdb_dir: Optional[str]
    :param chains: List of chains to load for each PDB code.
    :type chains: Optional[List[str]]
    :param graph_labels: List of tensors to set as graph labels.
    :type graph_labels: Optional[List[torch.Tensor]]
    :param transform: List of transforms to apply to each example.
    :type transform: Optional[List[Callable]]
    :param format: PDB file format ("pdb", "mmtf", "ent").
    :type format: Literal["mmtf", "pdb", "ent"]
    :param in_memory: Whether to load data into memory.
    :type in_memory: bool
    :param overwrite: Whether to overwrite existing files.
    :type overwrite: bool
    """

    def __init__(
        self,
        pdb_codes: List[str],
        root: Optional[str] = None,
        pdb_dir: Optional[str] = None,
        chains: Optional[List[str]] = None,
        graph_labels: Optional[List[torch.Tensor]] = None,
        node_labels: Optional[List[torch.Tensor]] = None,
        transform: Optional[List[Callable]] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        log: bool = True,
        overwrite: bool = False,
        format: Literal["mmtf", "pdb", "ent"] = "pdb",
        in_memory: bool = False,
        store_het: bool = False,
        out_names: Optional[List[str]] = None,
    ):
        self.pdb_codes = [pdb.lower() for pdb in pdb_codes]
        self.pdb_dir = pdb_dir
        self.overwrite = overwrite
        self.chains = chains
        self.node_labels = node_labels
        self.graph_labels = graph_labels
        self.format = format
        self.root = root
        self.in_memory = in_memory
        self.store_het = store_het
        self.out_names = out_names

        self._processed_files = []

        # Determine whether to skip download/processing
        if not self.overwrite and all(
            os.path.exists(Path(self.root) / "processed" / p)
            for p in self.processed_file_names
        ):
            logger.info(
                "All structures already processed and overwrite=False. Skipping download."
            )
            self._skip_download = True
        else:
            self._skip_download = False

        super().__init__(root, transform, pre_transform, pre_filter, log)
        self.structures = pdb_codes
        if self.in_memory:
            logger.info("Reading data into memory")
            self.data = [
                torch.load(pathlib.Path(self.root) / "processed" / f, weights_only=False)
                for f in tqdm(self.processed_file_names)
            ]

    def download(self):
        """Skip download - data should be pre-downloaded."""
        if self._skip_download:
            logger.info("All structures already processed. Skipping download.")
            return

    def len(self) -> int:
        """Return length of the dataset."""
        return len(self.pdb_codes)

    @property
    def raw_dir(self) -> str:
        """Returns the path to the raw data directory."""
        return os.path.join(self.root, "raw") if self.pdb_dir is None else self.pdb_dir

    @property
    def raw_file_names(self) -> List[str]:
        """Returns the raw file names."""
        if self._skip_download:
            return []
        return [f"{pdb}.{self.format}" for pdb in self.pdb_codes]

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        """Returns the processed file names."""
        if self._processed_files:
            return self._processed_files
        if self.overwrite:
            return ["this_forces_a_processing_cycle"]
        if self.out_names is not None:
            return [f"{name}.pt" for name in self.out_names]
        if self.chains is not None:
            return [
                f"{pdb}_{chain}.pt"
                for pdb, chain in zip(self.pdb_codes, self.chains)
            ]
        else:
            return [f"{pdb}.pt" for pdb in self.pdb_codes]

    def process(self):
        """Process raw data into PyTorch Geometric Data objects."""
        if not self.overwrite:
            if self.chains is not None:
                index_pdb_tuples = [
                    (i, pdb)
                    for i, pdb in enumerate(self.pdb_codes)
                    if not os.path.exists(
                        Path(self.processed_dir) / f"{pdb}_{self.chains[i]}.pt"
                    )
                ]
            else:
                index_pdb_tuples = [
                    (i, pdb)
                    for i, pdb in enumerate(self.pdb_codes)
                    if not os.path.exists(
                        Path(self.processed_dir) / f"{pdb}.pt"
                    )
                ]
            logger.info(
                f"Processing {len(index_pdb_tuples)} unprocessed structures"
            )
        else:
            index_pdb_tuples = [
                (i, pdb) for i, pdb in enumerate(self.pdb_codes)
            ]

        raw_dir = Path(self.raw_dir)
        for index_pdb_tuple in tqdm(index_pdb_tuples):
            try:
                i, pdb = index_pdb_tuple
                path = raw_dir / f"{pdb}.{self.format}"
                if path.exists():
                    path = str(path)
                elif path.with_suffix("." + self.format + ".gz").exists():
                    path = str(path.with_suffix("." + self.format + ".gz"))
                else:
                    raise FileNotFoundError(
                        f"{pdb} not found in raw directory. Are you sure it's downloaded and has the format {self.format}?"
                    )
                graph = protein_to_pyg(
                    path=path,
                    chain_selection=self.chains[i]
                    if self.chains is not None
                    else "all",
                    keep_insertions=True,
                    store_het=self.store_het,
                )
            except Exception as e:
                logger.error(f"Error processing {pdb} {self.chains[i] if self.chains else ''}: {e}")
                raise e

            if self.out_names is not None:
                fname = self.out_names[i] + ".pt"
            else:
                fname = (
                    f"{pdb}.pt"
                    if self.chains is None
                    else f"{pdb}_{self.chains[i]}.pt"
                )

            graph.id = fname.split(".")[0]

            if self.graph_labels is not None:
                graph.graph_y = self.graph_labels[i]

            if self.node_labels is not None:
                graph.node_y = self.node_labels[i]

            torch.save(graph, Path(self.processed_dir) / fname)
            self._processed_files.append(fname)
        logger.info("Completed processing.")

    def get(self, idx: int) -> Data:
        """Return PyTorch Geometric Data object for a given index."""
        if self.in_memory:
            return self._batch_format(copy.deepcopy(self.data[idx]))

        if self.out_names is not None:
            fname = f"{self.out_names[idx]}.pt"
        elif self.chains is not None:
            fname = f"{self.pdb_codes[idx]}_{self.chains[idx]}.pt"
        else:
            fname = f"{self.pdb_codes[idx]}.pt"

        return self._batch_format(
            torch.load(Path(self.processed_dir) / fname, weights_only=False)
        )

    def _batch_format(self, x: Data) -> Data:
        """Format data for batching."""
        x.x = torch.zeros(x.coords.shape[0])
        x.amino_acid_one_hot = amino_acid_one_hot(x)
        x.seq_pos = torch.arange(x.coords.shape[0]).unsqueeze(-1)
        return x


class ECPSDataModule(L.LightningDataModule):
    """
    DataModule for Enzyme Commission classification from ProteinShake.

    Args:
        path: Root path to dataset (e.g., data/ec_proteinshake/)
        split_type: Type of split to use ("random" or "structure")
        batch_size: Batch size for dataloaders
        pdb_dir: Optional PDB directory (defaults to {path}/pdb/)
        format: PDB file format ("pdb", "mmtf", "ent")
        in_memory: Load entire dataset into memory
        pin_memory: Pin memory for dataloaders
        num_workers: Number of workers for dataloaders
        dataset_fraction: Fraction of dataset to use (for debugging)
        shuffle_labels: Shuffle labels (for negative control experiments)
        transforms: List of transforms to apply
        overwrite: Overwrite cached processed files
    """

    def __init__(
        self,
        path: str,
        split_type: Literal["random", "structure"] = "random",
        batch_size: int = 32,
        pdb_dir: Optional[str] = None,
        format: Literal["mmtf", "pdb", "ent"] = "pdb",
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
            raise FileNotFoundError(
                f"Split directory not found: {self.split_dir}. "
                f"Available split types: {[d.name for d in self.data_dir.iterdir() if d.is_dir() and d.name not in ['pdb', 'raw', 'processed']]}"
            )

        # Setup transforms
        if transforms is not None:
            if omegaconf.OmegaConf.is_config(transforms):
                transforms_list = omegaconf.OmegaConf.to_container(
                    transforms, resolve=True
                )
            else:
                transforms_list = transforms
            self.transform = self._compose_transforms(transforms_list)
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
            f"Setting up EC ProteinShake dataset with {split_type} split. "
            f"Fraction: {self.dataset_fraction}"
        )

    @typechecker
    def _compose_transforms(self, transforms: Iterable[Callable]) -> T.Compose:
        """Compose an iterable of Transforms into a single transform."""
        if isinstance(transforms, list):
            return T.Compose(transforms)
        elif isinstance(transforms, dict):
            return T.Compose(list(transforms.values()))
        else:
            raise ValueError("Transforms must be a list or dict")

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for train/val/test."""
        if stage == "fit" or stage is None:
            logger.info("Preprocessing training data")
            self.train_ds = self.train_dataset()
            logger.info("Preprocessing validation data")
            self.val_ds = self.val_dataset()
        elif stage == "test":
            logger.info("Preprocessing test data")
            self.test_ds = self.test_dataset()
        elif stage == "lazy_init":
            logger.info("Preprocessing validation data")
            self.val_ds = self.val_dataset()

    def parse_labels(self) -> Dict[str, int]:
        """Load labels from {path}/labels.csv"""
        labels_file = self.data_dir / "labels.csv"

        if not os.path.exists(labels_file):
            raise FileNotFoundError(
                f"Labels file not found: {labels_file}. "
                "Please run create_raw_data.py to generate dataset files."
            )

        df = pd.read_csv(labels_file)

        if "pdb_id" in df.columns and "label" in df.columns:
            label_dict = dict(zip(df["pdb_id"], df["label"]))
        elif len(df.columns) == 2:
            df.columns = ["pdb_id", "label"]
            label_dict = dict(zip(df["pdb_id"], df["label"]))
        else:
            raise ValueError(
                f"Invalid labels file format. Expected 2 columns, got {len(df.columns)}"
            )

        logger.info(f"Loaded {len(label_dict)} labels from {labels_file}")
        logger.info(
            f"Label range: {min(label_dict.values())} - {max(label_dict.values())}"
        )

        return label_dict

    def _load_split(self, split_name: str) -> pd.DataFrame:
        """Load split from {split_dir}/{split_name}_split.csv"""
        split_file = self.split_dir / f"{split_name}_split.csv"

        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"Split file not found: {split_file}. "
                "Please run create_raw_data.py to generate dataset files."
            )

        df = pd.read_csv(split_file)

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

        # Convert labels to int
        df["label"] = df["label"].astype(int)

        logger.info(
            f"Loaded {split_name} split ({self.split_type}): "
            f"{len(df)} proteins, "
            f"{df['label'].nunique()} unique labels"
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


if __name__ == "__main__":
    """Quick test of the datamodule"""
    from pathlib import Path

    # Test with local data
    print("=" * 70)
    print("Testing ECPSDataModule")
    print("=" * 70)

    data_path = Path(__file__).parent / "data" / "ec_proteinshake"

    if data_path.exists():
        datamodule = ECPSDataModule(
            path=str(data_path),
            split_type="random",
            batch_size=4,
            num_workers=0,
        )

        print("\nDataModule info:")
        print(f"  Split type: {datamodule.split_type}")
        print(f"  Data dir: {datamodule.data_dir}")
        print(f"  Split dir: {datamodule.split_dir}")
        print(f"  PDB dir: {datamodule.pdb_dir}")

        datamodule.setup("fit")
        train_dl = datamodule.train_dataloader()

        print("\nDataloader info:")
        print(f"  Batch size: {train_dl.batch_size}")

        print("\nFirst batch:")
        for batch in train_dl:
            print(f"  Num graphs: {batch.num_graphs}")
            print(f"  Coords shape: {batch.coords.shape}")
            print(f"  Graph labels: {batch.graph_y}")
            break

        print("\n✓ Datamodule loaded successfully!")
    else:
        print(f"Data path not found: {data_path}")
        print("Run create_raw_data_test.py first to generate test data.")

