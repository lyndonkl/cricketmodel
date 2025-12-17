# Training Pipeline

## Overview

The training pipeline follows standard PyTorch practices with PyG-specific data handling.

---

## 1. Training Loop

```python
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(loader, desc='Training'):
        batch = batch.to(device)

        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, batch.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.y.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            batch = batch.to(device)
            logits = model(batch)
            loss = criterion(logits, batch.y)

            total_loss += loss.item() * batch.y.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)

            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(batch.y.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels
```

---

## 2. Main Training Script

```python
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.dataset import CricketDataset, create_dataloaders
from src.model.hetero_gnn import CricketHeteroGNN

def main():
    # Config
    config = {
        'hidden_dim': 128,
        'num_layers': 3,
        'num_heads': 4,
        'dropout': 0.1,
        'batch_size': 64,
        'lr': 1e-3,
        'weight_decay': 0.01,
        'epochs': 100,
        'patience': 10,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    train_loader, val_loader, test_loader = create_dataloaders(
        root='data/processed',
        batch_size=config['batch_size']
    )

    # Model
    model = CricketHeteroGNN(
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
    ).to(device)

    # Loss with class weights for imbalanced data
    class_weights = compute_class_weights(train_loader)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])

    # Training
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'checkpoints/best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print("Early stopping triggered")
                break

    # Test
    model.load_state_dict(torch.load('checkpoints/best_model.pt'))
    test_loss, test_acc, preds, labels = evaluate(
        model, test_loader, criterion, device
    )
    print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # Detailed metrics
    print_classification_report(labels, preds)


if __name__ == '__main__':
    main()
```

---

## 3. Class Weights

Cricket outcomes are imbalanced (many dots, few wickets):

```python
from collections import Counter

def compute_class_weights(loader):
    """Compute inverse frequency class weights."""
    class_counts = Counter()

    for batch in loader:
        class_counts.update(batch.y.tolist())

    total = sum(class_counts.values())
    num_classes = 7

    weights = []
    for c in range(num_classes):
        if class_counts[c] > 0:
            weights.append(total / (num_classes * class_counts[c]))
        else:
            weights.append(1.0)

    return torch.tensor(weights, dtype=torch.float)
```

### Typical Class Distribution

| Class | Outcome | Typical % |
|-------|---------|-----------|
| 0 | Dot | ~40% |
| 1 | Single | ~30% |
| 2 | Two | ~8% |
| 3 | Three | ~2% |
| 4 | Four | ~12% |
| 5 | Six | ~5% |
| 6 | Wicket | ~3% |

---

## 4. Metrics

### Classification Report

```python
from sklearn.metrics import classification_report, confusion_matrix

def print_classification_report(labels, preds):
    target_names = ['Dot', 'Single', 'Two', 'Three', 'Four', 'Six', 'Wicket']
    print(classification_report(labels, preds, target_names=target_names))

    cm = confusion_matrix(labels, preds)
    print("\nConfusion Matrix:")
    print(cm)
```

### Additional Metrics

```python
def compute_metrics(labels, preds, probs):
    """Compute additional evaluation metrics."""
    from sklearn.metrics import (
        accuracy_score, f1_score, log_loss, top_k_accuracy_score
    )

    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'f1_macro': f1_score(labels, preds, average='macro'),
        'f1_weighted': f1_score(labels, preds, average='weighted'),
        'log_loss': log_loss(labels, probs),
        'top_2_accuracy': top_k_accuracy_score(labels, probs, k=2),
        'top_3_accuracy': top_k_accuracy_score(labels, probs, k=3),
    }

    return metrics
```

---

## 5. Distributed Training

For multi-GPU training with PyG:

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.loader import DataLoader

def setup_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def main_distributed():
    local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')

    # Model with DDP
    model = CricketHeteroGNN(...).to(device)
    model = DDP(model, device_ids=[local_rank])

    # Distributed sampler
    train_dataset = CricketDataset(root='data', split='train')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        sampler=train_sampler,
        num_workers=4
    )

    # Training loop (same as before)
    ...
```

---

## 6. Logging with Weights & Biases

```python
import wandb

def train_with_logging():
    wandb.init(project='cricket-prediction-v2', config=config)

    for epoch in range(config['epochs']):
        train_loss, train_acc = train_epoch(...)
        val_loss, val_acc, _, _ = evaluate(...)

        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': scheduler.get_last_lr()[0],
        })

    wandb.finish()
```

---

## 7. Checkpointing

```python
def save_checkpoint(model, optimizer, scheduler, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, path)

def load_checkpoint(model, optimizer, scheduler, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch']
```

---

## 8. Hyperparameter Search

```python
import optuna

def objective(trial):
    config = {
        'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256]),
        'num_layers': trial.suggest_int('num_layers', 2, 5),
        'num_heads': trial.suggest_categorical('num_heads', [2, 4, 8]),
        'dropout': trial.suggest_float('dropout', 0.0, 0.3),
        'lr': trial.suggest_loguniform('lr', 1e-4, 1e-2),
    }

    model = CricketHeteroGNN(**config)
    # ... train and evaluate

    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
```

---

## 9. Inference

```python
def predict(model, data, device):
    """Predict outcome for a single ball."""
    model.eval()

    with torch.no_grad():
        data = data.to(device)
        logits = model(data)
        probs = torch.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1)

    return pred.item(), probs.squeeze().cpu().numpy()

# Usage
outcome_names = ['Dot', 'Single', 'Two', 'Three', 'Four', 'Six', 'Wicket']
pred, probs = predict(model, sample_data, device)
print(f"Prediction: {outcome_names[pred]}")
print(f"Probabilities: {dict(zip(outcome_names, probs))}")
```
