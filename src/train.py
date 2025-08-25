import os
from dataclasses import dataclass
from typing import Tuple, Dict, List

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


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 128
    num_workers: int = 2
    use_amp: bool = True                 # mixed precision on GPU for speed
    save_dir: str = "artifacts"
    save_name: str = "best_cifar10_cnn.pt"
    seed: int = 42

def set_seed(seed: int) -> None:
    """Make results a bit more reproducible."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[int, int]:
    """
    Convert model scores (logits) to predicted class IDs with argmax,
    compare to ground-truth labels, and count how many were correct.
    """
    preds = logits.argmax(dim=1)          # index of largest score along class dimension
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct, total

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None
) -> Dict[str, float]:
    """
    Runs one full pass over the training data:
      - model.train(): turn on layers like dropout, enable gradient tracking
      - forward -> loss -> backward -> optimizer.step()
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # clear old gradients efficiently

        # Forward pass (optionally in mixed precision)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            # Backward pass with gradient scaling to avoid underflow in float16
            scaler.scale(loss).backward()
            # (Optional) gradient clipping to stabilize training on some models
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
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

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluation:
      - model.eval(): turn off dropout, use running stats in BatchNorm
      - NO backward/optimizer steps; just measure loss & accuracy.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

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

def train_loop(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data: if your get_cifar10_loaders returns train & test,
    # we'll use test as "validation" during training.
    train_loader, val_loader = get_cifar10_loaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    # Model, loss, optimizer, scheduler
    model = CIFAR10CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # This scheduler reduces LR if validation accuracy stops improving ("plateaus")
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, verbose=True)

    # Mixed precision scaler (GPU only); speeds up training and uses less VRAM
    scaler = torch.cuda.amp.GradScaler() if (cfg.use_amp and device.type == "cuda") else None

    os.makedirs(cfg.save_dir, exist_ok=True)
    best_path = os.path.join(cfg.save_dir, cfg.save_name)

    history: Dict[str, List[float]] = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Track metrics
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])

        # Tell the scheduler how we're doing (using val accuracy here)
        scheduler.step(val_metrics["acc"])

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
            print(f"  ✔ Saved new best model to {best_path} (val_acc={best_val_acc*100:.2f}%)")

    # Optional: plot simple training curves for a quick look
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
    '''trainloader, testloader = get_cifar10_loaders(batch_size=128)
    model = CIFAR10CNN()

    # Define loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)'''

    cfg = TrainConfig()
    train_loop(cfg)

if __name__ == "__main__":
    main()