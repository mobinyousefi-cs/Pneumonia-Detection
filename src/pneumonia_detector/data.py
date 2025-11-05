#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Artificial Intelligence in Pneumonia Detection
File: data.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-05
Updated: 2025-11-05
License: MIT License (see LICENSE file for details)
=

Description:
Data loading and preprocessing utilities for chest X-ray images.

Usage:
from pneumonia_detector.config import TrainingConfig
from pneumonia_detector.data import create_dataloaders

cfg = TrainingConfig()
loaders = create_dataloaders(cfg)

Notes:
- Expects Kaggle chest_xray directory structure with train/val/test splits.
=================================================================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import TrainingConfig


def _build_transforms(image_size: int) -> Dict[str, transforms.Compose]:
    """Return training and evaluation transforms."""

    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return {"train": train_tf, "val": eval_tf, "test": eval_tf}


def create_dataloaders(cfg: TrainingConfig) -> Dict[str, DataLoader]:
    """Create DataLoader objects for train, val, and test splits.

    Parameters
    ----------
    cfg:
        Training configuration.

    Returns
    -------
    dict
        Mapping of split name to DataLoader.
    """

    data_dir: Path = cfg.data_dir
    transforms_dict = _build_transforms(cfg.image_size)

    splits = ["train", "val", "test"]
    loaders: Dict[str, DataLoader] = {}

    for split in splits:
        split_dir = data_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Expected split directory not found: {split_dir}")

        dataset = datasets.ImageFolder(root=str(split_dir), transform=transforms_dict[split])

        shuffle = split == "train"
        loaders[split] = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

    return loaders