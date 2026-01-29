# RunPod Training Guide

This guide walks you through setting up RunPod for hyperparameter search and model training with the `CricketHeteroGNNFull` model. The model uses binary prediction heads (boundary detection + wicket detection) with Binary Focal Loss for class imbalance handling.

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

For multi-GPU hyperparameter search (7 GPUs recommended):

| Configuration | VRAM | Price/hr | Best For |
|---------------|------|----------|----------|
| 7x RTX 3090 | 7x 24 GB | ~$2.10 | Budget multi-GPU |
| 7x RTX 4090 | 7x 24 GB | ~$3.15 | Good balance |
| 7x A100 40GB | 7x 40 GB | ~$5.60 | Maximum speed |

**Why 7 GPUs?** Run 7 HP trials in parallel (one per GPU), completing searches ~7x faster.

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
# Install PyTorch (use version matching your CUDA driver)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric
pip install optuna optuna-integration[wandb] wandb
pip install tqdm pyyaml scikit-learn plotly

# Verify
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

### Login to WandB

```bash
wandb login
# Paste your API key from https://wandb.ai/authorize
```

---

## 7. Run Hyperparameter Search

### Model and Search Strategy

The search uses the `full_model_only` phase which fixes the model to `CricketHeteroGNNFull` and searches the following hyperparameters using Optuna with Bayesian optimization (TPE sampler):

| Hyperparameter | Search Range |
|----------------|-------------|
| `hidden_dim` | 64-256, step 32 |
| `num_layers` | 2-5 |
| `num_heads` | 2, 4, 8 |
| `lr` | 1e-4 to 2e-3 (log scale) |
| `dropout` | 0.0-0.3 |
| `weight_decay` | 1e-5 to 0.1 (log scale) |
| `focal_gamma` | 0.0-3.0 |

The objective is to **minimize validation loss**. Trials are pruned early using Optuna's MedianPruner.

### Multi-GPU Strategy: Parallel Workers

With 7 GPUs, run 7 Optuna workers in parallel - each worker uses one GPU and runs trials independently. All workers share the same SQLite database and coordinate automatically via Optuna's distributed optimization.

**Why this works:**
- Each worker sets `CUDA_VISIBLE_DEVICES` to use only one GPU
- All workers connect to the same Optuna study (SQLite storage)
- Optuna's TPE sampler coordinates trial selection across workers
- ~7x faster than single-GPU (7 trials run simultaneously)

### Run the Search

The search script is included in the repo at `scripts/run_full_model_search.sh`.

```bash
cd /workspace/cricketmodel
git pull  # Get the latest version

# Run with 7 GPUs (default is 6, override with NUM_GPUS)
NUM_GPUS=7 ./scripts/run_full_model_search.sh
```

**Default configuration:**
- `NUM_GPUS=6` - Override with env var (e.g., `NUM_GPUS=7`)
- `TRIALS_PER_GPU=15` - Total trials = NUM_GPUS x 15 (105 with 7 GPUs)
- `EPOCHS=10` - Epochs per trial
- `DATA_FRACTION=0.02` - 2% of data for fast iteration
- `BATCH_SIZE=1024` - Large batch for fast epochs on 24GB VRAM GPUs
- `STAGGER_DELAY=15` - Seconds between GPU starts to avoid SQLite race conditions

The script automatically:
1. Cleans up previous Optuna studies and checkpoints
2. Creates a base params file fixing `model_class=CricketHeteroGNNFull`
3. Launches one Optuna worker per GPU with staggered starts
4. Logs to WandB for real-time monitoring
5. Saves best parameters to `checkpoints/optuna/cricket_gnn_full_model_*/best_params.json`

### Running Long Jobs (Survives Disconnection)

**Important:** Always run from the cricketmodel directory.

**Option A: Using nohup**

```bash
cd /workspace/cricketmodel
nohup bash -c 'NUM_GPUS=7 ./scripts/run_full_model_search.sh' > hp_search.log 2>&1 &
echo $! > run.pid  # Save process ID
```

**Option B: Using screen (Recommended)**

```bash
# Start a named screen session
screen -S training

# Navigate to project and run
cd /workspace/cricketmodel
NUM_GPUS=7 ./scripts/run_full_model_search.sh

# Detach: Press Ctrl+A, then D
# Reconnect later: screen -r training
```

**Option C: Using tmux**

```bash
# Start a named tmux session
tmux new -s training

# Navigate to project and run
cd /workspace/cricketmodel
NUM_GPUS=7 ./scripts/run_full_model_search.sh

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
- Live training curves (validation loss per epoch)
- Trial comparisons
- GPU utilization
- Hyperparameter importance (after enough trials)
- Optuna visualizations (optimization history, parameter importance, parallel coordinate, contour plots)

---

## 8. Train Final Model with Best Parameters

After HP search completes, train the final model using `torchrun` for distributed training across all 7 GPUs:

```bash
# Get the best params from the search
BEST_PARAMS=$(ls -t checkpoints/optuna/cricket_gnn_full_model_*/best_params.json | head -1)

# View best parameters
cat $BEST_PARAMS

# Train final model with torchrun on 7 GPUs
torchrun --standalone --nproc_per_node=7 train.py \
    --config $BEST_PARAMS \
    --epochs 100 \
    --batch-size 1024 \
    --wandb \
    --wandb-project cricket-gnn-final
```

**torchrun options:**
- `--nproc_per_node=7` - Use exactly 7 GPUs (explicit)
- `--nproc_per_node=gpu` - Use all available GPUs (auto-detect)
- `--standalone` - Single-node training (no multi-machine coordination)

**Why torchrun?**
- Proper DDP setup - all 7 GPUs train the same model in parallel
- ~7x faster training than single GPU
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

## Cost Estimate (7x RTX 4090)

| Item | Cost |
|------|------|
| Network Volume (150GB) | ~$10.50/month |
| HP Search (7x RTX 4090, ~2 hrs) | ~$6.30 |
| Final Training (7x RTX 4090, ~1 hr) | ~$3.15 |
| **Total** | **~$20** |

**Note:** Multi-GPU is ~7x faster but same total GPU-hours cost. You pay for wall-clock time, so 7 GPUs for 1 hour = 1 GPU for 7 hours in cost.

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
