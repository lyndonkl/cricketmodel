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
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
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
        ollama_workers: int = 4,
    ):
        self.split = split
        self._raw_data_dir = raw_data_dir
        self.min_history = min_history
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed
        self._ollama_workers = ollama_workers

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
        return self.root  # root already IS the processed directory

    @property
    def raw_file_names(self) -> List[str]:
        """Expected raw files - any .json files in raw_dir."""
        # Dynamically find JSON files in raw_dir to satisfy PyG's files_exist() check
        # This prevents download() from being called when files already exist
        if os.path.exists(self.raw_dir):
            files = [f for f in os.listdir(self.raw_dir) if f.endswith('.json')]
            if files:
                return files[:1]  # Return at least one file to pass files_exist() check
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
        checkpoint_path = os.path.join(self.processed_dir, 'entity_mapper.pkl')
        entity_mapper.build_from_matches(
            match_files,
            checkpoint_path=checkpoint_path,
            num_workers=getattr(self, '_ollama_workers', 4),
            checkpoint_every=100,
        )
        print(f"Entity mapper: {entity_mapper}")

        # Save entity mapper (final save, may be redundant but ensures completeness)
        entity_mapper.save(checkpoint_path)

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
        # weights_only=False needed for PyTorch 2.6+ to load PyG HeteroData objects
        return torch.load(path, weights_only=False)

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
    train_fraction: float = 1.0,
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
        train_fraction: Fraction of data to use (default: 1.0).
                       Values < 1.0 create random subsets for faster HP search.
                       Applied to train, val, and test sets.
        **dataset_kwargs: Additional arguments for CricketDataset
                         (min_history, train_ratio, val_ratio, seed, etc.)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = CricketDataset(root, split='train', raw_data_dir=raw_data_dir, **dataset_kwargs)
    val_dataset = CricketDataset(root, split='val', raw_data_dir=raw_data_dir, **dataset_kwargs)
    test_dataset = CricketDataset(root, split='test', raw_data_dir=raw_data_dir, **dataset_kwargs)

    # Apply fraction to all datasets for faster HP search
    if train_fraction < 1.0:
        seed = dataset_kwargs.get('seed', 42)

        # Subset training data
        n_train = int(len(train_dataset) * train_fraction)
        train_gen = torch.Generator().manual_seed(seed)
        train_indices = torch.randperm(len(train_dataset), generator=train_gen)[:n_train]
        train_dataset = Subset(train_dataset, train_indices.tolist())

        # Subset validation data
        n_val = int(len(val_dataset) * train_fraction)
        val_gen = torch.Generator().manual_seed(seed + 1)
        val_indices = torch.randperm(len(val_dataset), generator=val_gen)[:n_val]
        val_dataset = Subset(val_dataset, val_indices.tolist())

        # Subset test data
        n_test = int(len(test_dataset) * train_fraction)
        test_gen = torch.Generator().manual_seed(seed + 2)
        test_indices = torch.randperm(len(test_dataset), generator=test_gen)[:n_test]
        test_dataset = Subset(test_dataset, test_indices.tolist())

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


def create_dataloaders_distributed(
    root: str,
    batch_size: int = 64,
    num_workers: int = 4,
    raw_data_dir: Optional[str] = None,
    rank: int = 0,
    world_size: int = 1,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, Optional[DistributedSampler]]:
    """
    Create train, validation, and test dataloaders with DDP support.

    When world_size > 1, uses DistributedSampler to partition data
    across GPUs. Each GPU receives non-overlapping batches.

    Following Sebastian Raschka's DDP methodology:
    - DistributedSampler ensures each GPU gets unique data
    - shuffle=False in DataLoader when using sampler (sampler handles shuffling)
    - drop_last=True ensures all GPUs have same batch count
    - Returns train_sampler so set_epoch() can be called each epoch

    Args:
        root: Root directory for processed data cache
        batch_size: Batch size PER GPU (effective batch = batch_size * world_size)
        num_workers: Number of data loading workers per GPU
        raw_data_dir: Directory containing raw JSON match files
        rank: Global rank of current process (0 to world_size-1)
        world_size: Total number of processes
        **dataset_kwargs: Additional arguments for CricketDataset
                         (min_history, train_ratio, val_ratio, seed, etc.)

    Returns:
        Tuple of (train_loader, val_loader, test_loader, train_sampler)
        train_sampler is returned so set_epoch() can be called each epoch
        for proper shuffling in DDP mode.

    Example:
        # In train.py with DDP
        train_loader, val_loader, test_loader, train_sampler = create_dataloaders_distributed(
            root="data/processed",
            batch_size=64,
            rank=rank,
            world_size=world_size,
        )

        # In training loop
        for epoch in range(epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)  # CRITICAL for proper shuffling
            train_one_epoch()
    """
    train_dataset = CricketDataset(root, split='train', raw_data_dir=raw_data_dir, **dataset_kwargs)
    val_dataset = CricketDataset(root, split='val', raw_data_dir=raw_data_dir, **dataset_kwargs)
    test_dataset = CricketDataset(root, split='test', raw_data_dir=raw_data_dir, **dataset_kwargs)

    # Create distributed sampler for training data
    train_sampler = None
    shuffle_train = True

    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,  # Ensures all GPUs have same batch count
        )
        shuffle_train = False  # Sampler handles shuffling

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True if world_size > 1 else False,
    )

    # Validation and test don't need distributed sampling
    # (we could add it for faster eval, but metrics must be gathered)
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

    return train_loader, val_loader, test_loader, train_sampler


def compute_class_weights(
    dataset, cache: bool = True
) -> tuple[torch.Tensor, dict]:
    """
    Compute inverse frequency class weights for imbalanced data.

    Args:
        dataset: CricketDataset or Subset to compute weights from
        cache: If True, cache weights to disk and load from cache on subsequent calls
               (only works with full CricketDataset, not Subset)

    Returns:
        Tuple of (weights tensor [num_classes], distribution dict)
    """
    # Handle Subset wrapper (used when data_fraction < 1.0)
    base_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
    cache_path = os.path.join(base_dataset.processed_dir, 'class_weights.pt')
    class_names = ['Dot', 'Single', 'Two', 'Three', 'Four', 'Six', 'Wicket']

    # Always use cache if it exists (contains weights from full dataset)
    if cache and os.path.exists(cache_path):
        print(f"Loading cached class weights from {cache_path}")
        cached = torch.load(cache_path, weights_only=False)
        return cached['weights'], cached['distribution']

    # Compute class weights from the actual dataset (may be subset)
    from collections import Counter

    class_counts = Counter()
    for data in tqdm(dataset, desc="Computing class weights", unit="sample"):
        class_counts[data.y.item()] += 1

    total = sum(class_counts.values())
    num_classes = 7  # 0-6

    weights = []
    for c in range(num_classes):
        if class_counts[c] > 0:
            weights.append(total / (num_classes * class_counts[c]))
        else:
            weights.append(1.0)

    weights_tensor = torch.tensor(weights, dtype=torch.float)

    # Build distribution dict
    distribution = {}
    for c in range(num_classes):
        count = class_counts[c]
        percentage = (count / total * 100) if total > 0 else 0
        distribution[class_names[c]] = {'count': count, 'percentage': percentage}

    # Save to cache (only for full dataset, not subsets)
    is_subset = hasattr(dataset, 'dataset')
    if cache and not is_subset:
        torch.save({'weights': weights_tensor, 'distribution': distribution}, cache_path)
        print(f"Cached class weights to {cache_path}")

    return weights_tensor, distribution


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
