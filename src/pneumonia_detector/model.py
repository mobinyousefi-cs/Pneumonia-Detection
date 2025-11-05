#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Artificial Intelligence in Pneumonia Detection
File: model.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-05
Updated: 2025-11-05
License: MIT License (see LICENSE file for details)
=

Description:
Model definitions and helpers (ResNet-18 transfer learning).

Usage:
from pneumonia_detector.config import TrainingConfig
from pneumonia_detector.model import build_model

cfg = TrainingConfig()
model, criterion = build_model(cfg)

Notes:
- Uses torchvision.models.resnet18 with ImageNet weights.
=================================================================================================================
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models

from .config import TrainingConfig


def build_model(cfg: TrainingConfig) -> Tuple[nn.Module, nn.Module]:
    """Build the CNN model and loss function.

    Parameters
    ----------
    cfg:
        Training configuration.

    Returns
    -------
    model, criterion
    """

    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    # Replace final layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, cfg.num_classes)

    criterion = nn.CrossEntropyLoss()

    return model, criterion


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def move_to_device(model: nn.Module, device: str) -> nn.Module:
    """Move model to target device."""

    return model.to(device)