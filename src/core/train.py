import os
import json
import time
import pickle
from typing import Tuple
from tqdm import tqdm


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix

# Support both package and direct script execution
try:
    from src.core.model import create_model
    from src.core.data_prep import BaccaratDataset
except ImportError:
    from model import create_model
    from data_prep import BaccaratDataset


def load_processed_data(data_dir: str, batch_size: int = 64) -> Tuple[DataLoader, DataLoader, DataLoader, object]:
    required = [
        "train_grids.npy", "train_labels.npy",
        "val_grids.npy", "val_labels.npy",
        "test_grids.npy", "test_labels.npy",
        "label_encoder.pkl",
    ]
    for fname in required:
        if not os.path.exists(os.path.join(data_dir, fname)):
            raise FileNotFoundError(f"Missing {fname} in {data_dir}. Generate with data_prep first.")

    train_grids = np.load(os.path.join(data_dir, "train_grids.npy"), mmap_mode='r')
    train_labels = np.load(os.path.join(data_dir, "train_labels.npy"), mmap_mode='r')
    val_grids = np.load(os.path.join(data_dir, "val_grids.npy"), mmap_mode='r')
    val_labels = np.load(os.path.join(data_dir, "val_labels.npy"), mmap_mode='r')
    test_grids = np.load(os.path.join(data_dir, "test_grids.npy"), mmap_mode='r')
    test_labels = np.load(os.path.join(data_dir, "test_labels.npy"), mmap_mode='r')

    with open(os.path.join(data_dir, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)

    train_set = BaccaratDataset(train_grids, train_labels)
    val_set = BaccaratDataset(val_grids, val_labels)
    test_set = BaccaratDataset(test_grids, test_labels)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    return train_loader, val_loader, test_loader, label_encoder


class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.count = 0
        self.state = None

    def step(self, val_loss: float, model: torch.nn.Module) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.count = 0
            self.state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.count += 1
        if self.count >= self.patience:
            if self.state is not None:
                model.load_state_dict(self.state)
            return True
        return False


def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for grids, labels in tqdm(loader, desc="Train", leave=False):
        grids = grids.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(grids)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(1, len(loader)), 100.0 * correct / max(1, total)


def validate(model, loader, device, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for grids, labels in tqdm(loader, desc="Val", leave=False):
            grids = grids.to(device)
            labels = labels.to(device)
            logits = model(grids)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds) if all_preds else np.array([])
    all_labels = np.concatenate(all_labels) if all_labels else np.array([])
    acc = 100.0 * correct / max(1, total)
    return total_loss / max(1, len(loader)), acc, all_preds, all_labels

def main(epochs: int = 50, save_dir: str = "./models"):
    os.makedirs(save_dir, exist_ok=True)
    # Hardcoded data path relative to this file
    data_dir = os.path.join(os.path.dirname(__file__), "processed_data")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader, label_encoder = load_processed_data(data_dir)

    model = create_model(device=device)
    model.get_model_info()

    # Use standard CrossEntropyLoss - let the model learn patterns naturally
    # No class weights or focal loss bias - equal treatment for all outcomes
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=7, factor=0.5, min_lr=1e-6)
    stopper = EarlyStopping(patience=20, min_delta=1e-3)

    print("=" * 60)
    print(f"Training for {epochs} epochs on {device}")
    print("=" * 60)
    
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    start = time.time()
    for epoch in range(epochs):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, optimizer, criterion)
        val_loss, val_acc, _, _ = validate(model, val_loader, device, criterion)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr)
        print(f"Epoch {epoch+1:03d}/{epochs} | train {tr_loss:.4f}/{tr_acc:.2f}% | val {val_loss:.4f}/{val_acc:.2f}% | lr {lr:.6f} | {time.time()-t0:.1f}s")
        if stopper.step(val_loss, model):
            print("Early stopping.")
            break
        
    train_time = time.time() - start
    print(f"Training done in {train_time/60:.2f} min. Evaluating...")

    # Evaluation
    test_loss, test_acc, preds, labels = validate(model, test_loader, device, criterion)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    report = classification_report(labels, preds, target_names=list(label_encoder.classes_), digits=4, zero_division=0)
    cm = confusion_matrix(labels, preds)

    # Save
    model_path = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_path, exist_ok=True)
    # Save full checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'label_encoder': label_encoder,
        'device': str(device),
    }, os.path.join(model_path, 'baccarat_cnn.pth'))

    # Additionally, save a weights-only file for safe inference
    torch.save(model.state_dict(), os.path.join(model_path, 'baccarat_cnn_weights.pth'))

    with open(os.path.join(save_dir, 'training_summary.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'epochs': len(history['train_loss']),
            'train_time_sec': train_time,
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'train_acc': history['train_acc'],
            'val_acc': history['val_acc'],
            'test': {
                'loss': float(test_loss),
                'accuracy': float(test_acc),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
            }
        }, f, indent=2)

    print(f"Test: loss={test_loss:.4f}, acc={test_acc:.2f}%, precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}")
    print("Classes:", list(label_encoder.classes_))
    print("\nClassification report:\n")
    print(report)
    print("Confusion matrix:\n", cm)
    print(f"Saved model and summary to {save_dir}")


if __name__ == "__main__":
    import sys
    epochs = 1
    for i, arg in enumerate(sys.argv[1:]):
        if arg.startswith("--epochs="):
            try:
                epochs = int(arg.split("=", 1)[1])
            except ValueError:
                pass
        elif arg in ("-e", "--epochs") and i + 2 <= len(sys.argv):
            try:
                epochs = int(sys.argv[i + 2])
            except ValueError:
                pass
    main(epochs=epochs)


