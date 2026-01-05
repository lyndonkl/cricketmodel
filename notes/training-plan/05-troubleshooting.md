# Troubleshooting Guide

This document helps diagnose and fix common training issues with CricketHeteroGNN models.

---

## Quick Diagnosis Table

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Loss is NaN | Numerical instability | Gradient clipping, reduce LR |
| Val loss increases from epoch 1 | LR too high | Reduce LR 10x |
| Train loss stuck | LR too low or low capacity | Increase LR or hidden_dim |
| Only predicts majority class | Class imbalance | Focal loss + class weights |
| Training crashes with OOM | Batch size too large | Reduce batch_size |
| Gradients vanishing | Too many layers | Reduce num_layers, add skip connections |
| Gradients exploding | Unstable training | Gradient clipping |
| Model not improving | Poor architecture | Check data, increase capacity |

---

## Loss Issues

### NaN Loss

**Symptoms**:
```
Epoch 1: Loss = 2.345
Epoch 2: Loss = 5.678
Epoch 3: Loss = nan
```

**Causes**:
1. Learning rate too high
2. No gradient clipping
3. Division by zero in custom loss
4. Log of zero probability

**Solutions**:

```python
# 1. Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 2. Reduce learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)  # was 1e-3

# 3. Add epsilon to prevent log(0)
loss = -torch.log(probs + 1e-8)

# 4. Check for NaN in forward pass
def forward_with_nan_check(self, data):
    x_dict = self.encode(data)
    for name, x in x_dict.items():
        if torch.isnan(x).any():
            raise ValueError(f"NaN detected in {name} after encoding")
    # ... continue
```

### Loss Not Decreasing

**Symptoms**:
```
Epoch 1: Loss = 1.890
Epoch 10: Loss = 1.885
Epoch 50: Loss = 1.882
```

**Causes**:
1. Learning rate too low
2. Model capacity too small
3. Data issue (labels shuffled, features wrong)

**Solutions**:

```python
# 1. Try higher learning rates
for lr in [1e-3, 5e-3, 1e-2]:
    # train and see if loss moves

# 2. Increase model capacity
model = CricketHeteroGNN(
    hidden_dim=256,  # was 128
    num_layers=4,    # was 3
)

# 3. Sanity check: overfit a single batch
single_batch = next(iter(train_loader))
for i in range(100):
    loss = train_step(model, single_batch)
    print(f"Step {i}: {loss:.4f}")
# Should see loss → 0 if model can learn
```

### Loss Oscillating

**Symptoms**:
```
Epoch 1: Loss = 1.8
Epoch 2: Loss = 2.1
Epoch 3: Loss = 1.7
Epoch 4: Loss = 2.3
```

**Causes**:
1. Learning rate too high
2. Batch size too small
3. Data shuffling issue

**Solutions**:

```python
# 1. Reduce learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 2. Increase batch size
train_loader = DataLoader(train_dataset, batch_size=128)  # was 32

# 3. Use learning rate warmup
def lr_lambda(epoch):
    warmup_epochs = 5
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    return 1.0

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

---

## Class Imbalance Issues

### Only Predicts Majority Class

**Symptoms**:
```
Confusion matrix shows all predictions in one column:
     0   1   2   3   4   5   6
 0  [0   100 0   0   0   0   0]
 1  [0   200 0   0   0   0   0]
 2  [0   50  0   0   0   0   0]
 ...
```

**Cause**: Model learns to always predict the most common class (dots, typically class 1).

**Solutions**:

```python
# 1. Use class weights
class_counts = np.bincount(y_train)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * len(class_weights)
weights = torch.tensor(class_weights, dtype=torch.float)

criterion = nn.CrossEntropyLoss(weight=weights)

# 2. Use Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

criterion = FocalLoss(gamma=2.0, weight=weights)

# 3. Oversample minority classes
from torch.utils.data import WeightedRandomSampler

sample_weights = [class_weights[y] for y in y_train]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
```

### Wicket Recall Is Zero

**Symptoms**:
```
val/wicket_recall: 0.000
```

**Cause**: Wickets are very rare (~3-5% of balls). Model ignores them.

**Solutions**:

```python
# 1. Higher weight for wicket class
weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 1.0])  # class 5 = wicket

# 2. Higher focal gamma (focus more on hard examples)
criterion = FocalLoss(gamma=3.0)  # was 2.0

# 3. Separate binary head for wicket detection
# Add auxiliary loss:
#   main_loss + 0.5 * binary_wicket_loss
```

---

## Overfitting

### Symptoms

```
Epoch 20: Train Loss = 0.5, Val Loss = 1.0
Epoch 40: Train Loss = 0.2, Val Loss = 1.5  (Val getting worse)
Epoch 60: Train Loss = 0.1, Val Loss = 2.0
```

### Solutions

```python
# 1. Increase dropout
model = CricketHeteroGNN(dropout=0.3)  # was 0.1

# 2. Add weight decay
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.05  # was 0.01
)

# 3. Reduce model capacity
model = CricketHeteroGNN(
    hidden_dim=64,   # was 128
    num_layers=2,    # was 3
)

# 4. Early stopping
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None

    def __call__(self, val_score):
        if self.best_score is None or val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

early_stopping = EarlyStopping(patience=15)
for epoch in range(epochs):
    # ... train ...
    if early_stopping(val_macro_f1):
        print(f"Early stopping at epoch {epoch}")
        break
```

---

## Underfitting

### Symptoms

```
Epoch 100: Train Acc = 0.40, Val Acc = 0.38
# Both low, similar values
```

### Solutions

```python
# 1. Increase model capacity
model = CricketHeteroGNN(
    hidden_dim=256,   # was 128
    num_layers=4,     # was 3
    num_heads=8,      # was 4
)

# 2. Reduce regularization
model = CricketHeteroGNN(dropout=0.0)  # was 0.1
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

# 3. Train longer
# Increase epochs or reduce early stopping patience

# 4. Check data quality
# - Are features computed correctly?
# - Are labels correct?
# - Is there signal in the data?
```

---

## Graph-Specific Issues

### Over-Smoothing

**Symptoms**:
- Performance degrades with more layers
- All node embeddings become similar
- Smoothness metric drops below 0.3

**Diagnosis**:
```python
def measure_smoothness(node_embeddings):
    """1.0 = diverse, 0.0 = identical"""
    norms = node_embeddings.norm(dim=-1, keepdim=True)
    normalized = node_embeddings / (norms + 1e-8)
    sim_matrix = torch.mm(normalized, normalized.t())
    # Exclude diagonal
    n = sim_matrix.size(0)
    off_diag = sim_matrix.masked_select(~torch.eye(n, dtype=bool))
    return 1 - off_diag.mean().item()

# After forward pass:
smoothness = measure_smoothness(x_dict['player'])
if smoothness < 0.3:
    print("WARNING: Over-smoothing detected!")
```

**Solutions**:

```python
# 1. Reduce number of layers
model = CricketHeteroGNN(num_layers=2)  # was 4

# 2. Add skip connections (if not already)
class ConvBlockWithSkip(nn.Module):
    def forward(self, x, edge_index):
        out = self.conv(x, edge_index)
        return out + x  # Skip connection

# 3. Use JumpingKnowledge
from torch_geometric.nn import JumpingKnowledge

self.jump = JumpingKnowledge(mode='cat')  # or 'max', 'lstm'
# Concatenate all layer outputs
```

### Edge Index Errors

**Symptoms**:
```
RuntimeError: index out of bounds
```

**Cause**: Edge indices reference non-existent nodes.

**Diagnosis**:
```python
def check_edge_indices(data):
    for edge_type in data.edge_types:
        edge_index = data[edge_type].edge_index
        src_type, _, dst_type = edge_type

        num_src = data[src_type].x.size(0)
        num_dst = data[dst_type].x.size(0)

        src_max = edge_index[0].max().item()
        dst_max = edge_index[1].max().item()

        if src_max >= num_src:
            print(f"ERROR: {edge_type} src index {src_max} >= {num_src}")
        if dst_max >= num_dst:
            print(f"ERROR: {edge_type} dst index {dst_max} >= {num_dst}")

check_edge_indices(data)
```

---

## Memory Issues

### Out of Memory (OOM)

**Symptoms**:
```
RuntimeError: CUDA out of memory
```

**Solutions**:

```python
# 1. Reduce batch size
train_loader = DataLoader(train_dataset, batch_size=32)  # was 64

# 2. Use gradient accumulation
accumulation_steps = 4
optimizer.zero_grad()
for i, batch in enumerate(train_loader):
    loss = model(batch)
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 3. Reduce model size
model = CricketHeteroGNN(hidden_dim=64)  # was 128

# 4. Use mixed precision (FP16)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for batch in train_loader:
    with autocast():
        loss = model(batch)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# 5. Clear cache periodically
torch.cuda.empty_cache()
```

### Memory Leak

**Symptoms**:
- Memory usage grows each epoch
- Eventually crashes with OOM

**Cause**: Tensors not properly released.

**Solutions**:

```python
# 1. Detach tensors when logging
wandb.log({"loss": loss.item()})  # NOT loss (which keeps grad)

# 2. Delete large tensors explicitly
del outputs, loss
torch.cuda.empty_cache()

# 3. Don't store history
loss_history = []
for batch in train_loader:
    loss = model(batch)
    loss_history.append(loss.item())  # .item(), not loss

# 4. Check for retained gradients
for name, param in model.named_parameters():
    if param.grad is not None and param.grad.grad_fn is not None:
        print(f"WARNING: {name} grad has grad_fn")
```

---

## Debugging Checklist

### Before Training

- [ ] Data loads correctly
- [ ] Model forward pass works on single batch
- [ ] Loss computes without NaN
- [ ] Gradients flow to all parameters
- [ ] Learning rate schedule is correct

### During Training

- [ ] Loss is decreasing
- [ ] Validation loss follows train loss (initially)
- [ ] Gradient norms are stable
- [ ] All classes are being predicted (not just majority)

### After Training

- [ ] Final model loads correctly
- [ ] Predictions are diverse (not all same class)
- [ ] Metrics match expectations
- [ ] Model generalizes to test set

### Debugging Commands

```python
# Check gradients
for name, param in model.named_parameters():
    if param.grad is None:
        print(f"No gradient for {name}")
    elif param.grad.norm() == 0:
        print(f"Zero gradient for {name}")

# Check predictions
preds = model(batch)
print(f"Unique predictions: {torch.unique(preds.argmax(dim=-1))}")
print(f"Prediction distribution: {torch.bincount(preds.argmax(dim=-1))}")

# Check activations
def hook_fn(module, input, output):
    print(f"{module.__class__.__name__}: mean={output.mean():.4f}, std={output.std():.4f}")

for name, module in model.named_modules():
    module.register_forward_hook(hook_fn)
```

---

## Getting Help

If you've tried everything and are still stuck:

1. **Check WandB logs** for the exact epoch where things went wrong
2. **Compare to a working run** - what's different?
3. **Simplify** - can you reproduce with a simpler model?
4. **Isolate** - which component is causing the issue?
5. **Search** - has anyone else had this problem with PyG?

### Useful Resources

- PyTorch Geometric docs: https://pytorch-geometric.readthedocs.io/
- PyG GitHub issues: https://github.com/pyg-team/pytorch_geometric/issues
- Graph ML community: https://discord.gg/VnvuH7b
