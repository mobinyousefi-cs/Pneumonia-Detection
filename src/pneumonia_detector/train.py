#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Artificial Intelligence in Pneumonia Detection
File: train.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-05
Updated: 2025-11-05
License: MIT License (see LICENSE file for details)
=

Description:
Training and evaluation loops for pneumonia detection model.

Usage:
from pneumonia_detector.config import TrainingConfig
from pneumonia_detector.train import train_model

cfg = TrainingConfig()
metrics = train_model(cfg)

Notes:
- The CLI in cli.py wraps this module for command-line usage.
=================================================================================================================
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainingConfig
from .data import create_dataloaders
from .model import build_model, move_to_device


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    train: bool = True,
) -> Tuple[float, float, float, float, float]:
    """Run a single epoch for training or evaluation.

    Returns
    -------
    loss, acc, precision, recall, f1
    """

    if train:
        model.train()
    else:
        model.eval()

    all_targets = []
    all_preds = []
    running_loss = 0.0

    pbar = tqdm(loader, desc="train" if train else "eval", leave=False)

    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(train):
            outputs = model(images)
            loss = criterion(outputs, targets)

            if train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        all_targets.extend(targets.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / len(loader.dataset)

    acc = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average="binary")
    recall = recall_score(all_targets, all_preds, average="binary")
    f1 = f1_score(all_targets, all_preds, average="binary")

    return epoch_loss, acc, precision, recall, f1


def train_model(cfg: TrainingConfig) -> Dict[str, float]:
    """Train the model using configuration and return best metrics."""

    set_seed(cfg.seed)
    cfg.ensure_dirs()

    device = cfg.device
    loaders = create_dataloaders(cfg)

    model, criterion = build_model(cfg)
    model = move_to_device(model, device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)

    best_f1 = 0.0
    best_metrics: Dict[str, float] = {}

    for epoch in range(1, cfg.num_epochs + 1):
        train_loss, train_acc, train_prec, train_rec, train_f1 = _run_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            train=True,
        )

        val_loss, val_acc, val_prec, val_rec, val_f1 = _run_epoch(
            model,
            loaders["val"],
            criterion,
            optimizer=None,
            device=device,
            train=False,
        )

        scheduler.step()

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_precision": train_prec,
            "train_recall": train_rec,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_precision": val_prec,
            "val_recall": val_rec,
            "val_f1": val_f1,
        }

        print(
            f"Epoch {epoch:02d} | "
            f"Train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, "
            f"val F1: {val_f1:.4f}",
        )

        # Save best model by validation F1
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_metrics = metrics
            torch.save(model.state_dict(), cfg.checkpoint_path)

    # Evaluate on test set using best checkpoint
    if cfg.checkpoint_path.exists():
        model.load_state_dict(torch.load(cfg.checkpoint_path, map_location=device))

    test_loss, test_acc, test_prec, test_rec, test_f1 = _run_epoch(
        model,
        loaders["test"],
        criterion,
        optimizer=None,
        device=device,
        train=False,
    )

    best_metrics.update(
        {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_precision": test_prec,
            "test_recall": test_rec,
            "test_f1": test_f1,
        }
    )

    # Persist metrics
    with cfg.metrics_path.open("w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2)

    return best_metrics