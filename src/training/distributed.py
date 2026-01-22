"""
Distributed Data Parallel (DDP) utilities for Cricket GNN training.

Provides helper functions for initializing, detecting, and managing
distributed training across multiple GPUs following Sebastian Raschka's
methodology (https://sebastianraschka.com/teaching/pytorch-1h/).

Supports multiple platforms:
- Linux with NVIDIA GPUs: NCCL backend (optimal)
- Windows: Gloo backend
- macOS with MPS: Single-device training (DDP not supported)

Usage:
    # Single GPU (unchanged)
    python train.py

    # Multi-GPU with DDP (Linux/CUDA)
    torchrun --nproc_per_node=2 train.py
    torchrun --nproc_per_node=4 train.py --batch-size 32
"""

import os
import sys
import warnings
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


def get_backend() -> str:
    """
    Auto-detect the appropriate distributed backend.

    Backend selection per Sebastian Raschka's methodology:
    - NCCL: Optimal for NVIDIA GPUs (Linux/macOS with CUDA)
    - Gloo: Required for Windows, fallback for non-CUDA systems

    Note: macOS with MPS only (no CUDA) will use Gloo, but DDP
    is not recommended on MPS as Apple Silicon has single GPU.

    Returns:
        Backend name: "nccl" or "gloo"
    """
    # Windows requires Gloo
    if sys.platform == "win32":
        return "gloo"

    # NCCL requires CUDA
    if torch.cuda.is_available():
        return "nccl"

    # Fallback to Gloo for CPU/MPS
    return "gloo"


def is_mps_available() -> bool:
    """
    Check if MPS (Metal Performance Shaders) is available.

    MPS is Apple Silicon's GPU acceleration framework.

    Returns:
        True if MPS is available
    """
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def setup_distributed() -> Tuple[int, int, int]:
    """
    Initialize distributed process group.

    Called at the start of training when running with torchrun.
    Auto-detects the appropriate backend based on platform and hardware.

    Backend selection per Sebastian Raschka's methodology:
    - NCCL: Best performance for NVIDIA GPUs (Linux with CUDA)
    - Gloo: Required for Windows, fallback for CPU/MPS systems

    Note: MPS (Apple Silicon) has a single GPU, so DDP provides no benefit.
    For Mac development, prefer single-device training without torchrun.

    Returns:
        Tuple of (rank, local_rank, world_size)

    Example:
        rank, local_rank, world_size = setup_distributed()
        ddp_enabled = world_size > 1
        if force_cpu:  # --force-cpu flag
            device = torch.device('cpu')
        elif ddp_enabled:
            if torch.cuda.is_available():
                device = torch.device(f'cuda:{local_rank}')
            else:
                # Gloo backend requires CPU tensors
                device = torch.device('cpu')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        elif is_mps_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    """
    if not is_distributed():
        return 0, 0, 1

    backend = get_backend()

    # Warn if attempting DDP on MPS - it won't provide multi-GPU benefits
    if is_mps_available() and not torch.cuda.is_available():
        warnings.warn(
            "DDP on MPS is not recommended. Apple Silicon Macs have a single GPU, "
            "so DDP provides no parallelism benefit. Consider running without "
            "torchrun for single-device training: python train.py"
        )

    # Initialize process group
    # init_method="env://" reads MASTER_ADDR and MASTER_PORT from environment
    # Increase timeout to 60 minutes for long operations like class weight computation
    from datetime import timedelta
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=timedelta(minutes=60)
    )

    rank = get_rank()
    local_rank = get_local_rank()
    world_size = get_world_size()

    # Set the CUDA device for this process (only if CUDA available)
    # Each process is assigned to exactly one GPU
    if torch.cuda.is_available():
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


def gather_predictions(
    labels: list,
    preds: list,
    probs,
    device: torch.device
) -> tuple:
    """
    Gather predictions from all processes for distributed evaluation.

    When using DistributedSampler for validation/test, each GPU processes
    a subset of the data. This function gathers all predictions to rank 0
    for computing global metrics.

    Args:
        labels: List of ground truth labels from this process
        preds: List of predictions from this process
        probs: Numpy array of probabilities from this process
        device: Device tensors are on (used for synchronization)

    Returns:
        Tuple of (all_labels, all_preds, all_probs) on rank 0,
        or (None, None, None) on other ranks.

    Example:
        # After local evaluation
        if is_distributed:
            all_labels, all_preds, all_probs = gather_predictions(
                labels, preds, probs, device
            )
            if is_main:
                metrics = compute_metrics(all_labels, all_preds, all_probs)
    """
    import numpy as np

    if not is_distributed():
        return labels, preds, probs

    world_size = get_world_size()

    # Gather all predictions to all processes using all_gather_object
    # This handles variable-length lists automatically
    gathered_labels = [None] * world_size
    gathered_preds = [None] * world_size
    gathered_probs = [None] * world_size

    dist.all_gather_object(gathered_labels, labels)
    dist.all_gather_object(gathered_preds, preds)
    dist.all_gather_object(gathered_probs, probs.tolist())  # Convert to list for gathering

    # Only rank 0 needs to aggregate and return the full results
    if get_rank() == 0:
        # Flatten the gathered lists
        all_labels = [label for sublist in gathered_labels for label in sublist]
        all_preds = [pred for sublist in gathered_preds for pred in sublist]
        # Reconstruct probability array
        all_probs_list = [prob for sublist in gathered_probs for prob in sublist]
        all_probs = np.array(all_probs_list)
        return all_labels, all_preds, all_probs

    return None, None, None
