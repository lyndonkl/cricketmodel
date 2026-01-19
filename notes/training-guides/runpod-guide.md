# RunPod Training Guide

This guide walks you through setting up RunPod for hyperparameter search and model training with the Cricket GNN.

---

## 1. Create a RunPod Account

1. Go to [runpod.io](https://runpod.io) and create an account
2. Add credits to your account (start with $10-20 for initial testing)
3. Note: You'll need a payment method on file

---

## 2. Create a Network Volume

Network volumes persist independently of GPU pods, so you only pay storage costs (~$0.07/GB/month) while uploading data.

1. Go to **Storage** > **Network Volumes** in the RunPod dashboard
2. Click **+ New Network Volume**
3. Configure:
   - **Name:** `cricket-data`
   - **Datacenter:** Choose one with S3 API support:
     - EUR-IS-1, EU-RO-1, EU-CZ-1, US-KS-2, or US-CA-2
   - **Size:** 150 GB (gives headroom for processed data + checkpoints)
4. Click **Create**
5. Note the **Volume ID** (you'll need this later)

**Cost:** ~$10.50/month for 150GB

---

## 3. Upload Data via S3 API (No GPU Charges)

This is the key to avoiding GPU charges during upload. You upload directly to the network volume without running a pod.

### Get S3 Credentials

1. Go to **Settings** > **API Keys** in RunPod dashboard
2. Create a new API key or use existing one
3. Note your:
   - **Access Key ID**
   - **Secret Access Key**
   - **Endpoint URL** (based on your datacenter, e.g., `https://us-ks-2.runpod.io`)

### Install AWS CLI

```bash
# macOS
brew install awscli

# Or via pip
pip install awscli
```

### Configure AWS CLI for RunPod

```bash
aws configure --profile runpod
# Enter your RunPod Access Key ID
# Enter your RunPod Secret Access Key
# Region: us-east-1 (or leave blank)
# Output format: json
```

### Upload Your Processed Data

**Option A: Upload folder directly (Recommended - avoids zip/unzip time)**

```bash
# Sync the processed folder directly to network volume
# This uploads files in parallel and is often faster than zipping
cd /path/to/cricketmodel/data

aws s3 sync processed/ s3://YOUR_VOLUME_ID/processed/ \
    --endpoint-url https://YOUR_DATACENTER.runpod.io \
    --profile runpod

# Example:
aws s3 sync processed/ s3://vol_abc123xyz/processed/ \
    --endpoint-url https://us-ks-2.runpod.io \
    --profile runpod
```

**Option B: Upload as zip file**

```bash
# Zip first (if you prefer)
zip -r processed.zip processed/

# Upload zip
aws s3 cp processed.zip s3://YOUR_VOLUME_ID/ \
    --endpoint-url https://YOUR_DATACENTER.runpod.io \
    --profile runpod
```

**Upload time:** ~2-4 hours for 97GB depending on your internet speed. You're only charged storage (~$7/month), not GPU time.

### Verify Upload

```bash
aws s3 ls s3://YOUR_VOLUME_ID/processed/ \
    --endpoint-url https://YOUR_DATACENTER.runpod.io \
    --profile runpod | head -20
```

---

## 4. Create a GPU Pod

Now that your data is uploaded, create a GPU pod and attach the network volume.

### Choose a Template

1. Go to **Pods** > **+ Deploy**
2. Select **GPU Pod**
3. Choose a template:
   - **PyTorch 2.1** (or closest to 2.8 available)
   - Or use **RunPod Pytorch** template

### Select GPU

For multi-GPU hyperparameter search (4 GPUs recommended):

| Configuration | VRAM | Price/hr | Best For |
|---------------|------|----------|----------|
| 4x RTX 3090 | 4x 24 GB | ~$1.20 | Budget multi-GPU |
| 4x RTX 4090 | 4x 24 GB | ~$1.80 | Good balance |
| 4x A100 40GB | 4x 40 GB | ~$3.20 | Maximum speed |

**Why 4 GPUs?** Run 4 HP trials in parallel (one per GPU), completing searches ~4x faster.

### Attach Network Volume

1. In the pod configuration, find **Volume**
2. Select your `cricket-data` network volume
3. Set **Mount Path:** `/workspace/data`

**Note on Storage Architecture:** The network volume is network-attached storage (NAS), separate from the GPU node. Data transfers at 200-400 MB/s typically. For most training workloads, PyTorch DataLoader prefetching handles this well. If you experience I/O bottlenecks, you can copy data to the pod's local SSD at the start of training (see Section 6).

### Configure Pod

- **Container Disk:** 20 GB (for OS, packages, code)
- **Volume Disk:** Already set via network volume
- Expose ports: 8888 (Jupyter), 22 (SSH)

### Deploy

Click **Deploy** and wait for the pod to start (~30 seconds).

---

## 5. Connect to Your Pod

### Option A: Jupyter Lab (Recommended for Interactive Work)

1. Click **Connect** on your pod
2. Click **Connect to Jupyter Lab**
3. Opens in browser with full IDE

### Option B: SSH (Recommended for Long Training)

1. Click **Connect** > **SSH over exposed TCP**
2. Copy the SSH command:
   ```bash
   ssh root@ssh.runpod.io -p 12345 -i ~/.ssh/your_key
   ```

### Option C: Web Terminal

1. Click **Connect** > **Start Web Terminal**
2. Opens terminal in browser

---

## 6. Set Up the Environment

Once connected to your pod:

### Verify Data

If you uploaded the folder directly (Option A):
```bash
ls /workspace/data/processed/*.pt | head -5  # Should show .pt files
```

If you uploaded a zip file (Option B):
```bash
cd /workspace/data
unzip processed.zip
ls processed/  # Verify extraction
```

### Optional: Copy Data to Local SSD (Faster I/O)

If you experience data loading bottlenecks, copy to the pod's local storage:
```bash
# This uses the pod's local NVMe SSD (faster than network volume)
cp -r /workspace/data/processed /workspace/processed_local

# Then symlink to this location instead (in the next step)
```

### Clone Repository

```bash
cd /workspace
git clone https://github.com/lyndonkl/cricketmodel.git
cd cricketmodel
```

**For private repo**, use a GitHub PAT:
```bash
git clone https://<YOUR_PAT>@github.com/lyndonkl/cricketmodel.git
```

### Create Symlink to Data

```bash
# Link the network volume data to where the code expects it
ln -s /workspace/data/processed data/processed

# Verify
ls -la data/processed/*.pt | head -5
```

### Install Dependencies

```bash
# Install matching versions
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric==2.7.0
pip install optuna optuna-integration[wandb] wandb
pip install tqdm pyyaml scikit-learn plotly

# Verify
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Login to WandB

```bash
wandb login
# Paste your API key from https://wandb.ai/authorize
```

---

## 7. Run Hyperparameter Search (All 4 Phases)

### Multi-GPU Strategy: Parallel Workers

With 4 GPUs, run 4 Optuna workers in parallel - each worker uses one GPU and runs trials independently. All workers share the same SQLite database and coordinate automatically via Optuna's distributed optimization.

**Why this works:**
- Each worker sets `CUDA_VISIBLE_DEVICES` to use only one GPU
- All workers connect to the same Optuna study (SQLite storage)
- Optuna's TPE sampler coordinates trial selection across workers
- ~4x faster than single-GPU (4 trials run simultaneously)

### Phase 1: Coarse Search (4 GPUs)

```bash
# Run 4 parallel workers, one per GPU
for gpu in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gpu python scripts/hp_search.py \
        --phase phase1_coarse \
        --n-trials 10 \
        --epochs 25 \
        --wandb \
        --device cuda \
        --n-jobs 1 &
done
wait
echo "Phase 1 complete!"
```

**Expected time:** ~30 min (vs ~2 hours with single GPU)

### Phase 2: Architecture Tuning (4 GPUs)

```bash
PHASE1_BEST=$(ls -t checkpoints/optuna/cricket_gnn_phase1_coarse_*/best_params.json | head -1)
echo "Using Phase 1 results: $PHASE1_BEST"

for gpu in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gpu python scripts/hp_search.py \
        --phase phase2_architecture \
        --n-trials 12 \
        --epochs 25 \
        --best-params "$PHASE1_BEST" \
        --wandb \
        --device cuda \
        --n-jobs 1 &
done
wait
echo "Phase 2 complete!"
```

### Phase 3: Training Dynamics (4 GPUs)

```bash
PHASE2_BEST=$(ls -t checkpoints/optuna/cricket_gnn_phase2_architecture_*/best_params.json | head -1)
echo "Using Phase 2 results: $PHASE2_BEST"

for gpu in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gpu python scripts/hp_search.py \
        --phase phase3_training \
        --n-trials 15 \
        --epochs 30 \
        --best-params "$PHASE2_BEST" \
        --wandb \
        --device cuda \
        --n-jobs 1 &
done
wait
echo "Phase 3 complete!"
```

### Phase 4: Loss Function (4 GPUs)

```bash
PHASE3_BEST=$(ls -t checkpoints/optuna/cricket_gnn_phase3_training_*/best_params.json | head -1)
echo "Using Phase 3 results: $PHASE3_BEST"

for gpu in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gpu python scripts/hp_search.py \
        --phase phase4_loss \
        --n-trials 10 \
        --epochs 30 \
        --best-params "$PHASE3_BEST" \
        --wandb \
        --device cuda \
        --n-jobs 1 &
done
wait
echo "Phase 4 complete!"
```

### Run All Phases in Sequence (Unattended)

Create a script to run all 4 phases with 4 GPUs:

```bash
cat > run_all_phases.sh << 'EOF'
#!/bin/bash
set -e

NUM_GPUS=4

run_phase() {
    local phase=$1
    local trials=$2
    local epochs=$3
    local best_params=$4

    echo "=== Running $phase with $NUM_GPUS GPUs ==="

    for gpu in $(seq 0 $((NUM_GPUS-1))); do
        if [ -z "$best_params" ]; then
            CUDA_VISIBLE_DEVICES=$gpu python scripts/hp_search.py \
                --phase $phase --n-trials $trials --epochs $epochs \
                --wandb --device cuda --n-jobs 1 &
        else
            CUDA_VISIBLE_DEVICES=$gpu python scripts/hp_search.py \
                --phase $phase --n-trials $trials --epochs $epochs \
                --best-params "$best_params" \
                --wandb --device cuda --n-jobs 1 &
        fi
    done
    wait
    echo "=== $phase complete! ==="
}

# Phase 1: Coarse Search
run_phase "phase1_coarse" 10 25 ""
PHASE1_BEST=$(ls -t checkpoints/optuna/cricket_gnn_phase1_coarse_*/best_params.json | head -1)
echo "Phase 1 best: $PHASE1_BEST"

# Phase 2: Architecture
run_phase "phase2_architecture" 12 25 "$PHASE1_BEST"
PHASE2_BEST=$(ls -t checkpoints/optuna/cricket_gnn_phase2_architecture_*/best_params.json | head -1)
echo "Phase 2 best: $PHASE2_BEST"

# Phase 3: Training Dynamics
run_phase "phase3_training" 15 30 "$PHASE2_BEST"
PHASE3_BEST=$(ls -t checkpoints/optuna/cricket_gnn_phase3_training_*/best_params.json | head -1)
echo "Phase 3 best: $PHASE3_BEST"

# Phase 4: Loss Function
run_phase "phase4_loss" 10 30 "$PHASE3_BEST"
PHASE4_BEST=$(ls -t checkpoints/optuna/cricket_gnn_phase4_loss_*/best_params.json | head -1)
echo "Phase 4 best: $PHASE4_BEST"

echo ""
echo "=== All HP search phases complete! ==="
echo "Final best params: $PHASE4_BEST"
cat "$PHASE4_BEST"

# Train final model with all 4 GPUs using DDP
echo ""
echo "=== Training final model with torchrun (4 GPUs) ==="
torchrun --standalone --nproc_per_node=4 train.py \
    --config "$PHASE4_BEST" \
    --epochs 100 \
    --wandb \
    --wandb-project cricket-gnn-final

echo "=== Done! ==="
EOF

chmod +x run_all_phases.sh
```

### Running Long Jobs (Survives Disconnection)

**Option A: Using nohup**

```bash
nohup ./run_all_phases.sh > hp_search.log 2>&1 &
echo $! > run.pid  # Save process ID
```

**Option B: Using screen (Recommended)**

```bash
# Start a named screen session
screen -S training

# Run your script
./run_all_phases.sh

# Detach: Press Ctrl+A, then D
# Reconnect later: screen -r training
```

**Option C: Using tmux**

```bash
# Start a named tmux session
tmux new -s training

# Run your script
./run_all_phases.sh

# Detach: Press Ctrl+B, then D
# Reconnect later: tmux attach -t training
```

### Monitoring Progress

```bash
# If using nohup, tail the log
tail -f hp_search.log

# Check if process is running
ps aux | grep hp_search

# Check GPU usage
nvidia-smi -l 1  # Updates every second
```

**Best monitoring: WandB Dashboard**

Since we use `--wandb`, you can monitor all trials in real-time at [wandb.ai](https://wandb.ai). You'll see:
- Live training curves
- Trial comparisons
- GPU utilization
- Hyperparameter importance (after enough trials)

---

## 8. Train Final Model with Best Parameters

After HP search completes, train the final model using `torchrun` for distributed training across all 4 GPUs:

```bash
# Get the final best params (from Phase 4)
BEST_PARAMS=$(ls -t checkpoints/optuna/cricket_gnn_phase4_loss_*/best_params.json | head -1)

# View best parameters
cat $BEST_PARAMS

# Train final model with torchrun on 4 GPUs
torchrun --standalone --nproc_per_node=4 train.py \
    --config $BEST_PARAMS \
    --epochs 100 \
    --wandb \
    --wandb-project cricket-gnn-final
```

**torchrun options:**
- `--nproc_per_node=4` - Use exactly 4 GPUs (explicit)
- `--nproc_per_node=gpu` - Use all available GPUs (auto-detect)
- `--standalone` - Single-node training (no multi-machine coordination)

**Why torchrun?**
- Proper DDP setup - all 4 GPUs train the same model in parallel
- ~4x faster training than single GPU
- Automatic gradient synchronization across GPUs
- Better error handling and process management

---

## 9. Download Results

### Option A: Download via S3 API

```bash
# On the pod, copy results to network volume
cp -r checkpoints /workspace/data/

# From your local machine
aws s3 cp s3://YOUR_VOLUME_ID/checkpoints ./checkpoints --recursive \
    --endpoint-url https://YOUR_DATACENTER.runpod.io \
    --profile runpod
```

### Option B: Download via SCP

```bash
# From your local machine
scp -P 12345 -r root@ssh.runpod.io:/workspace/cricketmodel/checkpoints ./
```

### Option C: Zip and Download via Jupyter

In Jupyter terminal:
```bash
cd /workspace/cricketmodel
zip -r results.zip checkpoints/optuna/ optuna_studies.db
```
Then download `results.zip` through Jupyter file browser.

---

## 10. Stop Pod to Save Money

**Important:** Stop your pod when not training to avoid charges.

1. Go to RunPod dashboard > **Pods**
2. Click **Stop** on your pod
3. Your network volume data persists (you only pay storage)

To resume later:
1. Start the pod again
2. Network volume is automatically re-mounted
3. Continue from where you left off

---

## Cost Estimate (4x RTX 4090)

| Item | Cost |
|------|------|
| Network Volume (150GB) | ~$10.50/month |
| HP Search (4x RTX 4090, ~2 hrs) | ~$3.60 |
| Final Training (4x RTX 4090, ~1 hr) | ~$1.80 |
| **Total** | **~$16** |

**Note:** Multi-GPU is ~4x faster but same total GPU-hours cost. You pay for wall-clock time, so 4 GPUs for 1 hour = 1 GPU for 4 hours in cost.

---

## Troubleshooting

### Pod won't start
- Check if the datacenter has available GPUs
- Try a different GPU type or datacenter

### Data not found
- Verify symlink: `ls -la data/processed/`
- Check network volume is mounted: `df -h`

### CUDA out of memory
- Reduce batch size: add `--batch-size 32` to commands
- Use a GPU with more VRAM

### Connection lost during training
- Use `nohup` or `screen`/`tmux` for long-running jobs
- Check pod logs in RunPod dashboard

### S3 upload fails
- Verify credentials: `aws s3 ls s3://YOUR_VOLUME_ID/ --endpoint-url ... --profile runpod`
- Check datacenter supports S3 API (EUR-IS-1, EU-RO-1, EU-CZ-1, US-KS-2, US-CA-2)
