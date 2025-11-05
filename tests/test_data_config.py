#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Artificial Intelligence in Pneumonia Detection
File: test_data_config.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-05
Updated: 2025-11-05
License: MIT License (see LICENSE file for details)
=

Description:
Lightweight tests for configuration and dataloader factories.
=================================================================================================================
"""

from pathlib import Path

from pneumonia_detector.config import TrainingConfig


def test_config_paths_properties() -> None:
    cfg = TrainingConfig(output_dir=Path("artifacts/test"))

    assert cfg.checkpoint_path.name == "best_model.pt"
    assert cfg.metrics_path.name == "metrics.json"