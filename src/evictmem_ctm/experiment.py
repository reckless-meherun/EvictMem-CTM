"""Shared training and evaluation utilities for sequence classifiers."""

import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from evictmem_ctm.data.system_logs import GAPS


def seed_everything(seed: int) -> None:
    # Required by CUDA matrix operations when deterministic algorithms are enabled.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def binary_metrics(predictions: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Compute accuracy and positive-class F1 without external metrics packages."""
    if labels.numel() == 0:
        raise ValueError("Cannot compute metrics for an empty group")
    correct = int((predictions == labels).sum())
    true_positive = int(((predictions == 1) & (labels == 1)).sum())
    false_positive = int(((predictions == 1) & (labels == 0)).sum())
    false_negative = int(((predictions == 0) & (labels == 1)).sum())
    denominator = 2 * true_positive + false_positive + false_negative
    return {"accuracy": correct / labels.numel(),
            "f1": 2 * true_positive / denominator if denominator else 0.0}


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    device: torch.device) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_examples = 0
    for sequences, labels, _gaps in loader:
        sequences, labels = sequences.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(sequences), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        total_examples += len(labels)
    if not total_examples:
        raise ValueError("Training loader is empty")
    return total_loss / total_examples


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             gaps: tuple[int, ...] = GAPS) -> dict:
    """Compute overall metrics and per-gap metrics; gaps never enter the model."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    predictions, targets, gaps = [], [], []
    total_loss = 0.0
    for sequences, labels, batch_gaps in loader:
        logits = model(sequences.to(device))
        total_loss += criterion(logits, labels.to(device)).item()
        predictions.append(logits.argmax(dim=1).cpu())
        targets.append(labels.cpu())
        gaps.append(batch_gaps.cpu())
    if not targets:
        raise ValueError("Evaluation loader is empty")
    predicted = torch.cat(predictions)
    actual = torch.cat(targets)
    gap_values = torch.cat(gaps)
    result = {"loss": total_loss / len(actual), **binary_metrics(predicted, actual)}
    result["per_gap"] = {}
    for gap in gaps:
        selected = gap_values == gap
        if not bool(selected.any()):
            raise ValueError(f"Evaluation data has no examples for gap {gap}")
        result["per_gap"][gap] = {"count": int(selected.sum()),
                                  **binary_metrics(predicted[selected], actual[selected])}
    return result


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, validation_f1: float, model_name: str,
                    model_config: dict, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "epoch": epoch, "validation_f1": validation_f1,
                "model_name": model_name, "model_config": model_config,
                "seed": seed}, path)


def load_checkpoint(path: Path, model: nn.Module, device: torch.device,
                    optimizer: torch.optim.Optimizer | None = None,
                    expected_model_name: str | None = None,
                    expected_model_config: dict | None = None) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if expected_model_name is not None and checkpoint["model_name"] != expected_model_name:
        raise ValueError("Checkpoint model name does not match requested model")
    if expected_model_config is not None and checkpoint["model_config"] != expected_model_config:
        raise ValueError("Checkpoint model configuration does not match requested model")
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint


def fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
        device: torch.device, checkpoint_path: Path, max_epochs: int = 20,
        learning_rate: float = 1e-3, weight_decay: float = 0.0,
        patience: int = 3, *, model_name: str, model_config: dict,
        seed: int = 42, gaps: tuple[int, ...] = GAPS) -> dict:
    """Train with validation-F1 early stopping and save the best epoch."""
    if max_epochs < 1 or patience < 1 or not model_name:
        raise ValueError("Positive epoch/patience and explicit model identity/config are required")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)
    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        validation = evaluate(model, val_loader, device, gaps)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_f1={validation['f1']:.4f}")
        if validation["f1"] > best_f1:
            best_f1 = validation["f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(checkpoint_path, model, optimizer, epoch, best_f1,
                            model_name, model_config, seed)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    return {"best_epoch": best_epoch, "best_validation_f1": best_f1,
            "training_time_seconds": time.perf_counter() - started}
