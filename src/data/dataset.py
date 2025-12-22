"""
Cricket Dataset for PyTorch Geometric

On-disk Dataset implementation for cricket ball prediction.
Saves each sample as a separate file for memory-efficient loading.
Handles data loading, processing, and train/val/test splitting.
"""

import json
import os
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from .entity_mapper import EntityMapper
from .hetero_data_builder import create_samples_from_match


class CricketDataset(Dataset):
    """
    PyTorch Geometric dataset for cricket ball prediction.

    Each sample is a HeteroData object representing the match state
    before a specific ball, with the label being that ball's outcome.

    The dataset processes raw match JSON files and caches processed
    data for efficient loading. Each sample is saved as a separate
    file on disk to avoid loading all data into memory.

    IMPORTANT: Split by MATCH, not by ball, to avoid data leakage.

    Args:
        root: Root directory for processed data cache
        split: One of 'train', 'val', 'test', or 'all'
        raw_data_dir: Directory containing raw JSON match files (optional).
                      If not provided, defaults to {root}/raw/matches/
        transform: Optional transform to apply to samples
        pre_transform: Optional pre-transform to apply during processing
        min_history: Minimum balls of history required for prediction
        train_ratio: Fraction of matches for training (default 0.8)
        val_ratio: Fraction of matches for validation (default 0.1)
        seed: Random seed for reproducible splits
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        raw_data_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        min_history: int = 1,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.split = split
        self._raw_data_dir = raw_data_dir
        self.min_history = min_history
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed

        # These will be loaded after super().__init__ triggers process()
        self._split_indices: Optional[List[int]] = None

        super().__init__(root, transform, pre_transform)

        # Load split indices after processing
        self._load_split_indices()

    def _load_split_indices(self) -> None:
        """Load the indices for this split from the split_indices.json file."""
        split_indices_path = os.path.join(self.processed_dir, 'split_indices.json')
        with open(split_indices_path, 'r') as f:
            all_splits = json.load(f)
        self._split_indices = all_splits[self.split]

    @property
    def raw_dir(self) -> str:
        """Return custom raw data directory if provided, else default."""
        if self._raw_data_dir is not None:
            return self._raw_data_dir
        return os.path.join(self.root, 'raw', 'matches')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        """Expected raw files - any .json files in raw_dir."""
        # Return empty list to skip download check when using custom raw_data_dir
        # The actual file existence is checked in process()
        return []

    @property
    def processed_file_names(self) -> List[str]:
        """Files that must exist for processing to be skipped."""
        # We check for essential metadata files
        # Individual sample files (data_*.pt) are tracked via split_indices.json
        return [
            'entity_mapper.pkl',
            'metadata.json',
            'split_indices.json',
        ]

    def download(self):
        """Download not implemented - data should be provided."""
        raise RuntimeError(
            f"Dataset not found at {self.raw_dir}. "
            "Please provide match JSON files in raw/matches/ directory."
        )

    def process(self):
        """Process raw match files into individual HeteroData sample files."""
        # Find all match JSON files in raw_dir
        if not os.path.exists(self.raw_dir):
            raise RuntimeError(
                f"Raw data directory not found at {self.raw_dir}. "
                "Please provide match JSON files or set raw_data_dir parameter."
            )

        match_files = sorted(Path(self.raw_dir).glob('*.json'))
        print(f"Found {len(match_files)} match files")

        if len(match_files) == 0:
            raise RuntimeError("No match JSON files found")

        # Step 1: Build entity mapper from ALL matches
        print("Building entity mapper...")
        entity_mapper = EntityMapper()
        entity_mapper.build_from_matches(match_files)
        print(f"Entity mapper: {entity_mapper}")

        # Save entity mapper
        entity_mapper.save(os.path.join(self.processed_dir, 'entity_mapper.pkl'))

        # Step 2: Split matches (not balls!)
        random.seed(self.seed)
        match_files = list(match_files)
        random.shuffle(match_files)

        n_matches = len(match_files)
        n_train = int(n_matches * self.train_ratio)
        n_val = int(n_matches * self.val_ratio)

        train_matches = match_files[:n_train]
        val_matches = match_files[n_train:n_train + n_val]
        test_matches = match_files[n_train + n_val:]

        print(f"Split: {len(train_matches)} train, {len(val_matches)} val, "
              f"{len(test_matches)} test matches")

        # Step 3: Process all matches and save each sample individually
        # Track which global indices belong to which split
        split_indices = {
            'train': [],
            'val': [],
            'test': [],
            'all': [],
        }

        global_idx = 0

        # Process train matches
        print("Processing train split...")
        for match_file in tqdm(train_matches, desc="Train matches"):
            samples = self._process_single_match(match_file, entity_mapper)
            for sample in samples:
                self._save_sample(sample, global_idx)
                split_indices['train'].append(global_idx)
                split_indices['all'].append(global_idx)
                global_idx += 1
        print(f"  {len(split_indices['train'])} samples")

        # Process val matches
        print("Processing val split...")
        for match_file in tqdm(val_matches, desc="Val matches"):
            samples = self._process_single_match(match_file, entity_mapper)
            for sample in samples:
                self._save_sample(sample, global_idx)
                split_indices['val'].append(global_idx)
                split_indices['all'].append(global_idx)
                global_idx += 1
        print(f"  {len(split_indices['val'])} samples")

        # Process test matches
        print("Processing test split...")
        for match_file in tqdm(test_matches, desc="Test matches"):
            samples = self._process_single_match(match_file, entity_mapper)
            for sample in samples:
                self._save_sample(sample, global_idx)
                split_indices['test'].append(global_idx)
                split_indices['all'].append(global_idx)
                global_idx += 1
        print(f"  {len(split_indices['test'])} samples")

        # Save split indices
        split_indices_path = os.path.join(self.processed_dir, 'split_indices.json')
        with open(split_indices_path, 'w') as f:
            json.dump(split_indices, f)

        # Save metadata
        metadata = {
            'num_matches': n_matches,
            'num_train_matches': len(train_matches),
            'num_val_matches': len(val_matches),
            'num_test_matches': len(test_matches),
            'num_train_samples': len(split_indices['train']),
            'num_val_samples': len(split_indices['val']),
            'num_test_samples': len(split_indices['test']),
            'num_total_samples': global_idx,
            'num_venues': entity_mapper.num_venues,
            'num_teams': entity_mapper.num_teams,
            'num_players': entity_mapper.num_players,
            'min_history': self.min_history,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'seed': self.seed,
        }
        with open(os.path.join(self.processed_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Total samples saved: {global_idx}")

    def _process_single_match(
        self,
        match_file: Path,
        entity_mapper: EntityMapper
    ) -> List[HeteroData]:
        """Process a single match file into HeteroData samples."""
        try:
            with open(match_file, 'r') as f:
                match_data = json.load(f)

            samples = create_samples_from_match(
                match_data=match_data,
                entity_mapper=entity_mapper,
                min_history=self.min_history
            )

            # Apply pre_transform if specified
            if self.pre_transform is not None:
                samples = [self.pre_transform(s) for s in samples]

            return samples

        except Exception as e:
            print(f"Warning: Failed to process {match_file}: {e}")
            return []

    def _save_sample(self, sample: HeteroData, idx: int) -> None:
        """Save a single sample to disk."""
        path = os.path.join(self.processed_dir, f'data_{idx}.pt')
        torch.save(sample, path)

    def len(self) -> int:
        """Return the number of samples in this split."""
        return len(self._split_indices)

    def get(self, idx: int) -> HeteroData:
        """Load and return a single sample by index."""
        # Map split-local index to global index
        global_idx = self._split_indices[idx]
        path = os.path.join(self.processed_dir, f'data_{global_idx}.pt')
        return torch.load(path)

    def get_entity_mapper(self) -> EntityMapper:
        """Load and return the entity mapper."""
        mapper_path = os.path.join(self.processed_dir, 'entity_mapper.pkl')
        return EntityMapper.load(mapper_path)

    def get_metadata(self) -> dict:
        """Load and return dataset metadata."""
        metadata_path = os.path.join(self.processed_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            return json.load(f)


def create_dataloaders(
    root: str,
    batch_size: int = 64,
    num_workers: int = 4,
    raw_data_dir: Optional[str] = None,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        root: Root directory for processed data cache
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        raw_data_dir: Directory containing raw JSON match files (optional).
                      If not provided, defaults to {root}/raw/matches/
        **dataset_kwargs: Additional arguments for CricketDataset
                         (min_history, train_ratio, val_ratio, seed, etc.)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = CricketDataset(root, split='train', raw_data_dir=raw_data_dir, **dataset_kwargs)
    val_dataset = CricketDataset(root, split='val', raw_data_dir=raw_data_dir, **dataset_kwargs)
    test_dataset = CricketDataset(root, split='test', raw_data_dir=raw_data_dir, **dataset_kwargs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def compute_class_weights(dataset: CricketDataset) -> torch.Tensor:
    """
    Compute inverse frequency class weights for imbalanced data.

    Args:
        dataset: CricketDataset to compute weights from

    Returns:
        Tensor of class weights [num_classes]
    """
    from collections import Counter

    class_counts = Counter()
    for data in dataset:
        class_counts[data.y.item()] += 1

    total = sum(class_counts.values())
    num_classes = 7  # 0-6

    weights = []
    for c in range(num_classes):
        if class_counts[c] > 0:
            weights.append(total / (num_classes * class_counts[c]))
        else:
            weights.append(1.0)

    return torch.tensor(weights, dtype=torch.float)


def get_class_distribution(dataset: CricketDataset) -> dict:
    """
    Get class distribution statistics.

    Args:
        dataset: CricketDataset to analyze

    Returns:
        Dict with class names and their percentages
    """
    from collections import Counter

    class_names = ['Dot', 'Single', 'Two', 'Three', 'Four', 'Six', 'Wicket']
    class_counts = Counter()

    for data in dataset:
        class_counts[data.y.item()] += 1

    total = sum(class_counts.values())

    distribution = {}
    for c in range(7):
        count = class_counts[c]
        percentage = (count / total * 100) if total > 0 else 0
        distribution[class_names[c]] = {
            'count': count,
            'percentage': round(percentage, 2)
        }

    return distribution
