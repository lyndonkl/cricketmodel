"""
Cricket Dataset for PyTorch Geometric

InMemoryDataset implementation for cricket ball prediction.
Handles data loading, processing, and train/val/test splitting.
"""

import json
import os
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from torch_geometric.data import HeteroData, InMemoryDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from .entity_mapper import EntityMapper
from .hetero_data_builder import create_samples_from_match


class CricketDataset(InMemoryDataset):
    """
    PyTorch Geometric dataset for cricket ball prediction.

    Each sample is a HeteroData object representing the match state
    before a specific ball, with the label being that ball's outcome.

    The dataset processes raw match JSON files and caches processed
    data for efficient loading.

    IMPORTANT: Split by MATCH, not by ball, to avoid data leakage.

    Args:
        root: Root directory containing raw and processed data
        split: One of 'train', 'val', 'test', or 'all'
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
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        min_history: int = 1,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.split = split
        self.min_history = min_history
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed

        super().__init__(root, transform, pre_transform)

        # Load the appropriate split
        split_idx = {'train': 0, 'val': 1, 'test': 2, 'all': 3}[split]
        self.load(self.processed_paths[split_idx])

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        """Expected raw files."""
        return ['matches']  # Directory containing .json files

    @property
    def processed_file_names(self) -> List[str]:
        """Processed data files for each split."""
        return [
            'train_data.pt',
            'val_data.pt',
            'test_data.pt',
            'all_data.pt',
            'entity_mapper.pkl',
            'metadata.json',
        ]

    def download(self):
        """Download not implemented - data should be provided."""
        raise RuntimeError(
            f"Dataset not found at {self.raw_dir}. "
            "Please provide match JSON files in raw/matches/ directory."
        )

    def process(self):
        """Process raw match files into HeteroData samples."""
        # Find all match JSON files
        matches_dir = os.path.join(self.raw_dir, 'matches')
        if not os.path.exists(matches_dir):
            raise RuntimeError(
                f"Matches directory not found at {matches_dir}. "
                "Please provide match JSON files."
            )

        match_files = sorted(Path(matches_dir).glob('*.json'))
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

        # Step 3: Process each split
        splits = {
            'train': train_matches,
            'val': val_matches,
            'test': test_matches,
            'all': match_files,
        }

        for split_name, split_files in splits.items():
            print(f"Processing {split_name} split...")
            samples = self._process_matches(split_files, entity_mapper)
            print(f"  {len(samples)} samples")

            if self.pre_transform is not None:
                samples = [self.pre_transform(s) for s in samples]

            # Save processed data
            path = os.path.join(self.processed_dir, f'{split_name}_data.pt')
            self.save(samples, path)

        # Save metadata
        metadata = {
            'num_matches': n_matches,
            'num_train_matches': len(train_matches),
            'num_val_matches': len(val_matches),
            'num_test_matches': len(test_matches),
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

    def _process_matches(
        self,
        match_files: List[Path],
        entity_mapper: EntityMapper
    ) -> List[HeteroData]:
        """Process list of match files into HeteroData samples."""
        samples = []

        for match_file in tqdm(match_files, desc="Processing matches"):
            try:
                with open(match_file, 'r') as f:
                    match_data = json.load(f)

                match_samples = create_samples_from_match(
                    match_data=match_data,
                    entity_mapper=entity_mapper,
                    min_history=self.min_history
                )
                samples.extend(match_samples)

            except Exception as e:
                print(f"Warning: Failed to process {match_file}: {e}")

        return samples

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
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        root: Root directory for dataset
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        **dataset_kwargs: Additional arguments for CricketDataset

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = CricketDataset(root, split='train', **dataset_kwargs)
    val_dataset = CricketDataset(root, split='val', **dataset_kwargs)
    test_dataset = CricketDataset(root, split='test', **dataset_kwargs)

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
