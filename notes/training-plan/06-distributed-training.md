# Distributed Training Guide

This document covers multi-GPU training using PyTorch Distributed Data Parallel (DDP), following [Sebastian Raschka's methodology](https://sebastianraschka.com/teaching/pytorch-1h/).

---

## Platform Support

### Mac (Apple Silicon / MPS)

Apple Silicon Macs use MPS (Metal Performance Shaders) for GPU acceleration.

**Important**: MPS does not support multi-GPU DDP because Macs have a single integrated GPU. For Mac development, use single-device training:

```bash
# Recommended for Mac development
python train.py --epochs 10 --batch-size 32
```

If you accidentally run with `torchrun` on Mac, the code will still work but you'll see a warning that DDP provides no parallelism benefit on single-GPU systems.

**Q: Can I use multiple CPUs on Mac instead?**

While the Gloo backend technically supports multi-process CPU training, this provides **no parallelism benefit on a single machine**. DDP is designed for multi-device (GPU) or multi-node training. Running multiple CPU processes on the same machine just creates competition for the same resources. For single-machine Mac development, use standard single-process training.

### Linux with NVIDIA GPUs (RunPod, etc.)

Use NCCL backend for optimal multi-GPU performance:

```bash
# 2 GPUs
torchrun --nproc_per_node=2 train.py --epochs 100 --batch-size 64

# 4 GPUs
torchrun --nproc_per_node=4 train.py --epochs 100 --batch-size 32
```

### Windows

Windows uses the Gloo backend automatically:

```bash
torchrun --nproc_per_node=2 train.py
```

### Backend Selection

The backend is auto-detected based on platform and hardware:

| Platform | CUDA Available | Backend Used |
|----------|----------------|--------------|
| Linux    | Yes            | NCCL         |
| Linux    | No             | Gloo         |
| macOS    | Yes (rare)     | NCCL         |
| macOS    | No (MPS only)  | Gloo*        |
| Windows  | Any            | Gloo         |

*Note: DDP on MPS is not recommended due to single-GPU limitation.

---

## Overview

DDP enables training across multiple GPUs by:
1. Running one process per GPU
2. Keeping a model copy on each GPU
3. Splitting data across GPUs (non-overlapping batches)
4. Synchronizing gradients after each backward pass

**Key benefit**: Near-linear speedup with number of GPUs.

---

## Quick Start

### Single GPU (unchanged)
```bash
python train.py --epochs 100 --batch-size 64
```

### Multi-GPU with DDP
```bash
# 2 GPUs
torchrun --nproc_per_node=2 train.py --epochs 100 --batch-size 64

# 4 GPUs
torchrun --nproc_per_node=4 train.py --epochs 100 --batch-size 32

# All available GPUs
torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) train.py
```

---

## Effective Batch Size

With DDP, the effective batch size is:
```
effective_batch = batch_size_per_gpu * num_gpus
```

For example, `--batch-size 32` with 4 GPUs = 128 effective batch size.

**Recommendation**: When scaling to more GPUs, either:
- Keep per-GPU batch size constant (increases effective batch)
- Scale learning rate with `sqrt(num_gpus)` if increasing effective batch significantly

---

## Environment Variables

`torchrun` automatically sets these environment variables:

| Variable | Description |
|----------|-------------|
| `WORLD_SIZE` | Total number of processes |
| `RANK` | Global rank (0 to WORLD_SIZE-1) |
| `LOCAL_RANK` | GPU index on current node |
| `MASTER_ADDR` | Address of rank 0 process |
| `MASTER_PORT` | Port for communication |

---

## Multi-Node Training

For training across multiple machines:

```bash
# On node 0 (master)
torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --node_rank=0 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    train.py

# On node 1
torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --node_rank=1 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    train.py
```

---

## Performance Tuning

### Optimal Batch Sizes

| GPUs | Batch Size/GPU | Effective Batch | Notes |
|------|----------------|-----------------|-------|
| 1    | 64             | 64              | Baseline |
| 2    | 64             | 128             | Good speedup |
| 4    | 32-64          | 128-256         | May need LR scaling |
| 8    | 32             | 256             | Consider LR warmup |

### Learning Rate Scaling

When increasing effective batch size significantly:
```python
# Linear scaling rule
scaled_lr = base_lr * (effective_batch / base_batch)

# Square root scaling (more conservative)
scaled_lr = base_lr * sqrt(effective_batch / base_batch)
```

---

## WandB Integration with DDP

**Critical**: Only rank 0 should initialize WandB to avoid duplicate logging.

The training code already handles this:
- Only the main process (rank 0) prints to console
- Only the main process saves checkpoints
- Only the main process saves training history

When adding WandB logging, follow the same pattern:

```python
from src.training import is_main_process

if is_main_process():
    wandb.init(
        project="cricket-gnn",
        name=args.run_name,
        config=vars(args)
    )

# During training
if is_main_process():
    wandb.log({"train/loss": train_loss, "epoch": epoch})

# Cleanup
if is_main_process():
    wandb.finish()
```

**Multi-Node WandB Timeout**: For multi-node setups, if you encounter timeout errors during `wandb.init()`, increase the timeout:

```python
wandb.init(
    project="cricket-gnn",
    settings=wandb.Settings(init_timeout=120)  # 120 seconds
)
```

---

## Key Implementation Details

### 1. Process Initialization

The training script auto-detects DDP via environment variables:

```python
from src.training.distributed import setup_distributed

rank, local_rank, world_size = setup_distributed()
```

### 2. Model Wrapping

In `Trainer.__init__`, the model is wrapped with DDP:

```python
if self.is_distributed:
    from torch.nn.parallel import DistributedDataParallel as DDP
    self.model = DDP(model, device_ids=[rank])
```

### 3. Data Distribution

`DistributedSampler` ensures non-overlapping data:

```python
train_sampler = DistributedSampler(
    train_dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    drop_last=True,
)
```

### 4. Critical: Epoch Setting

**Must call `set_epoch()` each epoch** for proper shuffling:

```python
for epoch in range(epochs):
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    train_one_epoch()
```

### 5. Metric Aggregation

Training metrics are aggregated across processes:

```python
from src.training.distributed import reduce_tensor

avg_loss_tensor = torch.tensor([avg_loss], device=device)
global_avg_loss = reduce_tensor(avg_loss_tensor).item()
```

### 6. Synchronization with barrier()

`barrier()` ensures all processes reach a synchronization point before continuing. It is called after checkpoint saves:

```python
# In Trainer.train() after saving best model
if self.is_main:
    self.save_checkpoint('best_model.pt')

# All processes wait for checkpoint to be saved
if self.is_distributed:
    barrier()
```

**When to use barrier():**
- After saving checkpoints (prevents other ranks from continuing before save completes)
- Between distinct training phases

**When NOT to use barrier():**
- During forward/backward passes (DDP handles gradient sync automatically)
- Excessively (adds synchronization overhead)

### 7. Device Selection Architecture

Device selection happens in two places by design:

1. **`train.py`**: Primary control point - explicitly sets device based on platform and passes it to Trainer
2. **`trainer.py`**: Fallback logic if `device=None` - allows Trainer to work standalone

This defensive pattern ensures the Trainer can be used independently without requiring train.py's device logic.

---

## Common Issues

### Issue: NCCL Timeout

**Symptom**:
```
RuntimeError: NCCL error: unhandled system error
```

**Solutions**:
1. Increase timeout: `export NCCL_TIMEOUT=1800`
2. Check network connectivity between nodes
3. Ensure all GPUs are visible: `nvidia-smi`

### Issue: Out of Memory

**Symptom**: `CUDA out of memory` on some GPUs

**Solutions**:
1. Reduce `--batch-size`
2. Ensure uniform GPU memory across devices
3. Use `drop_last=True` in DataLoader (already enabled for DDP)

### Issue: Metrics Don't Match Single-GPU

**Cause**: Validation metrics not being synchronized across processes.

**Note**: The trainer already handles metric aggregation via `reduce_tensor()`.

### Issue: Notebooks Don't Work

**Expected**: DDP requires separate processes, which notebooks can't spawn correctly.

**Solution**: Always run DDP training from command line, not Jupyter:
```bash
torchrun --nproc_per_node=2 train.py
```

---

## Verification Checklist

- [ ] Single GPU training still works: `python train.py`
- [ ] DDP training works: `torchrun --nproc_per_node=2 train.py`
- [ ] Only rank 0 creates checkpoints
- [ ] Only rank 0 prints to console
- [ ] Checkpoints can be loaded for single-GPU inference
- [ ] Training metrics are consistent between single and multi-GPU

---

## Files Modified for DDP Support

| File | Changes |
|------|---------|
| `train.py` | DDP initialization, distributed dataloaders |
| `src/training/distributed.py` | NEW: DDP utility functions |
| `src/training/trainer.py` | DDP model wrapping, sampler handling, rank-gated ops |
| `src/data/dataset.py` | `create_dataloaders_distributed()` function |

---

## References

- [Sebastian Raschka's PyTorch DDP Guide](https://sebastianraschka.com/teaching/pytorch-1h/)
- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [PyTorch Distributed Documentation](https://docs.pytorch.org/docs/stable/distributed.html)
- [WandB Distributed Training Guide](https://docs.wandb.ai/models/track/log/distributed-training)
