import os
from dataclasses import dataclass
from typing import Tuple, Dict, List

import random
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau  
import matplotlib.pyplot as plt

from data_loader import get_cifar10_loaders
from model import CIFAR10CNN


# Config <3
@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 0.003
    weight_decay: float = 0.0
    batch_size: int = 128
    num_workers: int = 0
    use_amp: bool = False  # mixed precision on GPU for speed
    save_dir: str = "artifacts"
    save_name: str = "best_cifar10_cnn.pt"
    seed: int = 67

# Using seeds to make results reproducible
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Compares logits to the truth and returns correct predictions and total predictions
def accuracy_from_logits(logits, targets):
    preds = logits.argmax(dim=1)  # Returns index of most confident class for each image
    correct = (preds == targets).sum().item()  # Sum and convert to int
    total = targets.size(0)
    return correct, total

# Train loop for a single epoch
def train_one_epoch(model, loader, criterion, optimizer, device):
    
    # Initialize important variables and set model to "training mode"
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Training loop (One pass)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Clear old gradients (for efficiency)
        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # Stats
        running_loss += loss.item() * images.size(0)
        c, t = accuracy_from_logits(logits, labels)
        correct += c
        total += t

    return {
        "loss": running_loss / total,
        "acc": correct / total
    }

# Evaluates performance based on loss and accuracy
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    
    # Initialize important eval variables and set model to "evaluate mode"
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # Evaluate based on ground truth
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        c, t = accuracy_from_logits(logits, labels)
        correct += c
        total += t

    return {
        "loss": running_loss / total,
        "acc": correct / total
    }

# Complete training loop
def train_loop(cfg):
    set_seed(cfg.seed)

    # Use CPU power if GPU is not available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_loader, val_loader = get_cifar10_loaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    # Initialize model, criteria, optimizer
    model = CIFAR10CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # This scheduler reduces LR upon hitting a valley
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    # Initialize storage directory if not yet exists
    os.makedirs(cfg.save_dir, exist_ok=True)
    best_path = os.path.join(cfg.save_dir, cfg.save_name)

    # Initialize history
    history: Dict[str, List[float]] = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    # Main Training loop
    for epoch in range(1, cfg.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Track metrics
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])

        # Updating the scheduler on how we're doing (using validation accuracy here)
        scheduler.step(val_metrics["acc"])

        # Using formatted strings to be able to track the performance visually in real time
        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"train_loss: {train_metrics['loss']:.4f}  train_acc: {train_metrics['acc']*100:5.2f}% | "
            f"val_loss: {val_metrics['loss']:.4f}  val_acc: {val_metrics['acc']*100:5.2f}%"
        )

        # Save the best weights (by validation accuracy)
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            torch.save({"model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "epoch": epoch,
                        "val_acc": best_val_acc},
                       best_path)
            print(f"Saved new best model to {best_path} (val_acc={best_val_acc*100:.2f}%)")

    # Plot training curves
    try:
        plt.figure()
        plt.plot(history["train_acc"], label="train_acc")
        plt.plot(history["val_acc"], label="val_acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy over time")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.plot(history["train_loss"], label="train_loss")
        plt.plot(history["val_loss"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss over time")
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"(Plotting skipped: {e})")

def main():
    cfg = TrainConfig()
    train_loop(cfg)

if __name__ == "__main__":
    main()