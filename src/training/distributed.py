"""
Distributed Data Parallel (DDP) utilities for Cricket GNN training.

Provides helper functions for initializing, detecting, and managing
distributed training across multiple GPUs following Sebastian Raschka's
methodology (https://sebastianraschka.com/teaching/pytorch-1h/).

Usage:
    # Single GPU (unchanged)
    python train.py

    # Multi-GPU with DDP
    torchrun --nproc_per_node=2 train.py
    torchrun --nproc_per_node=4 train.py --batch-size 32
"""

import os
from typing import Tuple

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    """
    Check if we're running in a distributed environment.

    DDP mode is auto-detected via environment variables set by torchrun:
    - WORLD_SIZE: Total number of processes (GPUs)

    Returns:
        True if WORLD_SIZE > 1 (set by torchrun)
    """
    return int(os.environ.get("WORLD_SIZE", 1)) > 1


def get_rank() -> int:
    """
    Get global rank of current process.

    Returns:
        Global rank (0 to WORLD_SIZE-1), or 0 if not distributed
    """
    return int(os.environ.get("RANK", 0))


def get_local_rank() -> int:
    """
    Get local rank (GPU index) on current node.

    Returns:
        Local GPU index, or 0 if not distributed
    """
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    """
    Get total number of processes.

    Returns:
        World size, or 1 if not distributed
    """
    return int(os.environ.get("WORLD_SIZE", 1))


def is_main_process() -> bool:
    """
    Check if this is the main process (rank 0).

    Only the main process should:
    - Save checkpoints
    - Print to console
    - Initialize WandB
    - Log metrics

    Returns:
        True if rank 0 or not distributed
    """
    return get_rank() == 0


def setup_distributed() -> Tuple[int, int, int]:
    """
    Initialize distributed process group.

    Called at the start of training when running with torchrun.
    Uses NCCL backend for optimal GPU-to-GPU communication.

    NCCL (NVIDIA Collective Communications Library):
    - Best performance for NVIDIA GPUs
    - Supports all-reduce, broadcast, gather operations
    - Handles gradient synchronization efficiently

    Returns:
        Tuple of (rank, local_rank, world_size)

    Example:
        rank, local_rank, world_size = setup_distributed()
        device = torch.device(f'cuda:{local_rank}')
    """
    if not is_distributed():
        return 0, 0, 1

    # Initialize process group with NCCL backend (optimal for GPU)
    # init_method="env://" reads MASTER_ADDR and MASTER_PORT from environment
    dist.init_process_group(backend="nccl", init_method="env://")

    rank = get_rank()
    local_rank = get_local_rank()
    world_size = get_world_size()

    # Set the device for this process
    # Each process is assigned to exactly one GPU
    torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    """
    Clean up distributed process group.

    Called at the end of training to properly shut down.
    Ensures all processes exit cleanly and release resources.
    """
    if is_distributed():
        dist.destroy_process_group()


def barrier() -> None:
    """
    Synchronization point across all processes.

    All processes wait until everyone reaches this point.
    Useful for:
    - Ensuring checkpoint is saved before other processes continue
    - Coordinating between training phases
    - Preventing race conditions

    Example:
        if is_main_process():
            save_checkpoint(model)
        barrier()  # All processes wait for rank 0 to finish saving
    """
    if is_distributed():
        dist.barrier()


def reduce_tensor(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    """
    Reduce a tensor across all processes.

    Used to aggregate metrics (loss, accuracy) from all GPUs.
    Each GPU computes metrics on its local batch; this function
    combines them to get the global metric.

    Args:
        tensor: Tensor to reduce (should be on GPU)
        average: If True, average the values; if False, sum them

    Returns:
        Reduced tensor (same value on all processes)

    Example:
        # Each process has local loss
        local_loss = criterion(logits, labels)

        # Aggregate to get global loss
        loss_tensor = torch.tensor([local_loss], device=device)
        global_loss = reduce_tensor(loss_tensor).item()
    """
    if not is_distributed():
        return tensor

    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)

    if average:
        rt /= get_world_size()

    return rt
