# Google Colab Guide for Cricket GNN

This guide walks you through running hyperparameter search for the Cricket Ball Prediction GNN on Google Colab's free T4 GPU.

---

## 1. What is Google Colab?

Google Colab (Colaboratory) is a free Jupyter notebook environment that runs in your browser. Key features:

- **Free GPU access:** T4 GPU with 15 GB VRAM (sufficient for our model)
- **No setup required:** Python and common libraries pre-installed
- **Google Drive integration:** Easy file upload/download
- **Session limits:** 12 hours max runtime, ~90 minutes of browser inactivity triggers disconnect

**Access it at:** [colab.research.google.com](https://colab.research.google.com)

---

## 2. First-Time Setup

### Create a Google Account (if needed)

You need a Google account to use Colab. Create one at [accounts.google.com](https://accounts.google.com) if you don't have one.

### Get a WandB API Key

We use Weights & Biases (WandB) to track experiments:

1. Go to [wandb.ai](https://wandb.ai) and create a free account
2. Once logged in, go to [wandb.ai/authorize](https://wandb.ai/authorize)
3. Copy your API key (you'll paste this into Colab later)

### Prepare Your Data

You'll need the raw match data file:
- **File:** `data/t20s_male_json.zip` (216 MB)
- This contains ~2,300 T20 match JSON files

---

## 3. Enable GPU Runtime

Before running the notebook, ensure GPU is enabled:

1. Open the notebook in Colab
2. Click **Runtime** > **Change runtime type**
3. Under **Hardware accelerator**, select **T4 GPU**
4. Click **Save**

The first cell of the notebook verifies GPU availability.

---

## 4. Running the Notebook

### Cell-by-Cell Execution

Run cells in order by clicking the play button (▶) or pressing `Shift+Enter`:

| Cell | What It Does | Expected Output |
|------|--------------|-----------------|
| 1 | GPU check | Shows "T4" GPU with ~15 GB memory |
| 2 | Clone repo | Repository cloned, shows recent commits |
| 3 | Install deps | Packages install (may show warnings, that's OK) |
| 4 | WandB login | Prompts for API key, shows "Successfully logged in" |
| 5 | Upload data | File picker opens, then shows extraction count |
| 6 | Process data | Progress bars, takes ~30 minutes first time |
| 7 | Run HP search | Trial progress, F1 scores per trial |
| 8 | View results | Best hyperparameters found |
| 9 | Download | Downloads results.zip to your computer |

### Time Estimates

| Phase | Trials | Epochs | Approximate Time |
|-------|--------|--------|------------------|
| Quick test | 5 | 10 | ~30 minutes |
| Standard | 10 | 25 | 2-3 hours |
| Thorough | 20 | 30 | 5-6 hours |

### Monitoring Progress

During HP search, you'll see output like:
```
[I 2024-01-15 10:23:45,678] Trial 5 finished with value: 0.2341 and parameters: {...}
[I 2024-01-15 10:25:12,345] Trial 6 pruned.
```

- **finished with value:** Trial completed, shows F1 macro score
- **pruned:** Trial stopped early (underperforming)

You can also monitor in real-time on your [WandB dashboard](https://wandb.ai).

---

## 5. Troubleshooting

### Session Timeout / Disconnected

Colab has two timeout mechanisms:
- **Browser inactivity (~90 min):** If your browser tab is idle (no mouse/keyboard activity), Colab disconnects. Your code may keep running briefly but then stops.
- **Max runtime (12 hours):** Hard limit regardless of activity.

**To prevent browser inactivity disconnects:**
- Keep the Colab tab visible and occasionally interact with it (scroll, click)
- Don't close your laptop lid or let your computer sleep
- Some users run a simple JavaScript snippet in browser console to simulate activity (search "Colab anti-disconnect")

**To recover after disconnect:**
1. Click "Reconnect" in the toolbar
2. Re-run cells 1-5 (setup cells)
3. Cell 6 will be faster if data was cached
4. Your Optuna study is saved to SQLite and resumes automatically

### GPU Memory Errors

If you see `CUDA out of memory`:

1. Reduce batch size:
   ```python
   !python scripts/hp_search.py ... --batch-size 32
   ```

2. Restart runtime: **Runtime** > **Restart runtime**

3. Re-run cells from the beginning

### GPU Not Available

If cell 1 shows "No GPU detected":

1. Go to **Runtime** > **Change runtime type**
2. Select **T4 GPU** under Hardware accelerator
3. Click **Save**
4. Re-run cell 1

**Note:** Free GPUs have limited availability. If T4 is unavailable, try again later or during off-peak hours (early morning US time).

### Data Upload Issues

If the upload dialog doesn't appear or times out:

1. Refresh the page
2. Re-run the upload cell
3. Alternatively, upload to Google Drive and mount:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   !cp /content/drive/MyDrive/t20s_male_json.zip .
   ```

### WandB Login Fails

If WandB login doesn't work:

1. Get a fresh API key from [wandb.ai/authorize](https://wandb.ai/authorize)
2. Re-run the login cell
3. Paste the key when prompted

To run without WandB, remove `--wandb` from the command:
```python
!python scripts/hp_search.py --phase phase1_coarse --n-trials 10 --epochs 25 --device cuda
```

### Import Errors

If you see `ModuleNotFoundError`:

1. Re-run cell 3 (install dependencies)
2. Restart runtime if needed

---

## 6. Retrieving Results

### Download Checkpoints

Cell 9 creates and downloads `results.zip` containing:

- `checkpoints/optuna/*/best_params.json` - Best hyperparameters
- `checkpoints/optuna/*/study_results.json` - Full trial history
- `checkpoints/optuna/*/best_model.pt` - Model checkpoint
- `checkpoints/optuna/visualizations/*.html` - Interactive charts
- `optuna_studies.db` - SQLite database for resuming

### View WandB Dashboard

1. Go to [wandb.ai](https://wandb.ai)
2. Navigate to your project (default: `cricket-gnn-optuna`)
3. View:
   - **Runs table:** All trials with hyperparameters and metrics
   - **Charts:** F1 vs trial number, parameter importance
   - **Parallel coordinates:** Trace good configurations

### Using Results for Next Phase

To continue with Phase 2 locally:

```bash
# Extract results
unzip results.zip

# Run Phase 2 with best params from Phase 1
python scripts/hp_search.py \
    --phase phase2_architecture \
    --n-trials 12 \
    --epochs 25 \
    --best-params checkpoints/optuna/cricket_gnn_phase1_coarse_*/best_params.json \
    --wandb \
    --device cpu
```

---

## 7. Tips for Efficient Colab Usage

### Save Progress Frequently

- Download results after each phase completes
- Copy important files to Google Drive as backup

### Maximize GPU Time

- Prepare your data file before starting the session
- Have your WandB API key ready
- Run thorough searches only when you have 3+ hours

### Work in Phases

Complete one phase per session rather than running `full_with_model`:

1. Session 1: Phase 1 (coarse search)
2. Session 2: Phase 2 (architecture)
3. Session 3: Phase 3 (training dynamics)
4. Session 4: Phase 4 (loss function)

This approach is more resilient to disconnections.

### Use Off-Peak Hours

GPU availability is better during:
- Early morning US time (2-8 AM PST)
- Weekends
- Non-semester periods (summers, holidays)

---

## 8. Colab vs Local Training Comparison

| Aspect | Colab (T4 GPU) | Local (CPU) | Local (M1/M2 Mac) |
|--------|----------------|-------------|-------------------|
| Speed | ~3x faster | Baseline | ~1.5x faster |
| Cost | Free | Free | Free |
| Session limit | 12 hours | None | None |
| Setup | ~10 minutes | Environment setup | Environment setup |
| Data | Upload each session | Local disk | Local disk |
| Best for | Quick experiments | Long runs | Development |

**Recommendation:** Use Colab for initial exploration and Phase 1-2. Use local CPU/MPS for final training with best hyperparameters.
