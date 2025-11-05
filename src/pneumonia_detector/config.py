#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Artificial Intelligence in Pneumonia Detection
File: config.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-05
Updated: 2025-11-05
License: MIT License (see LICENSE file for details)
=

Description:
Configuration dataclasses for training, evaluation, and paths.

Usage:
from pneumonia_detector.config import TrainingConfig
cfg = TrainingConfig()

Notes:
- Adjust default paths/hyperparameters as needed.
=================================================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import torch


@dataclass
class TrainingConfig:
    """Configuration for training and evaluation."""

    # Paths
    data_dir: Path = Path("data/chest_xray")
    output_dir: Path = Path("artifacts/exp1")

    # Dataloader params
    batch_size: int = 32
    num_workers: int = 4

    # Training params
    num_epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-4

    # Model params
    image_size: int = 224
    num_classes: int = 2
    class_names: List[str] = field(
        default_factory=lambda: ["NORMAL", "PNEUMONIA"],
    )

    # Misc
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def ensure_dirs(self) -> None:
        """Create output directory if it does not exist."""

        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def checkpoint_path(self) -> Path:
        """Default path for best model checkpoint."""

        return self.output_dir / "best_model.pt"

    @property
    def metrics_path(self) -> Path:
        """Default path for metrics JSON file."""

        return self.output_dir / "metrics.json"