#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Artificial Intelligence in Pneumonia Detection
File: test_model.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-05
Updated: 2025-11-05
License: MIT License (see LICENSE file for details)
=

Description:
Unit tests for model construction.
=================================================================================================================
"""

import torch

from pneumonia_detector.config import TrainingConfig
from pneumonia_detector.model import build_model, count_parameters


def test_build_model_output_shape() -> None:
    cfg = TrainingConfig(batch_size=4)
    model, criterion = build_model(cfg)

    x = torch.randn(4, 3, cfg.image_size, cfg.image_size)
    out = model(x)

    assert out.shape == (4, cfg.num_classes)
    assert criterion(out, torch.randint(0, cfg.num_classes, (4,))).ndim == 0


def test_model_param_count_positive() -> None:
    cfg = TrainingConfig()
    model, _ = build_model(cfg)

    n_params = count_parameters(model)
    assert n_params > 0