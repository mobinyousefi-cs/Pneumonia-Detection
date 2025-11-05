#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Artificial Intelligence in Pneumonia Detection
File: infer.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-05
Updated: 2025-11-05
License: MIT License (see LICENSE file for details)
=

Description:
Inference utilities for single-image pneumonia prediction.

Usage:
from pneumonia_detector.config import TrainingConfig
from pneumonia_detector.infer import load_model_for_inference, predict_image

cfg = TrainingConfig()
model = load_model_for_inference(cfg, checkpoint_path)
label, prob = predict_image(cfg, model, image_path)

Notes:
- Used by CLI for the `predict` command.
=================================================================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torchvision import transforms

from .config import TrainingConfig
from .model import build_model, move_to_device


def load_model_for_inference(cfg: TrainingConfig, checkpoint_path: Path) -> torch.nn.Module:
    """Load model weights from checkpoint for inference."""

    device = cfg.device
    model, _ = build_model(cfg)
    model = move_to_device(model, device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _build_infer_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def predict_image(
    cfg: TrainingConfig,
    model: torch.nn.Module,
    image_path: Path,
) -> Tuple[str, float]:
    """Predict pneumonia vs normal class for a single image.

    Returns
    -------
    (label, probability_of_predicted_label)
    """

    device = cfg.device
    tf = _build_infer_transform(cfg.image_size)

    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

    prob, pred_idx = probs.max(dim=1)
    label = cfg.class_names[int(pred_idx.item())]

    return label, float(prob.item())