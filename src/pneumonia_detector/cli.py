#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Artificial Intelligence in Pneumonia Detection
File: cli.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-05
Updated: 2025-11-05
License: MIT License (see LICENSE file for details)
=

Description:
Command-line interface (CLI) for training and inference.

Usage:
python -m pneumonia_detector.cli train --data-dir data/chest_xray
python -m pneumonia_detector.cli predict --checkpoint artifacts/exp1/best_model.pt --image path/to/xray.jpeg

Notes:
- Thin wrapper around train.py and infer.py modules.
=================================================================================================================
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import TrainingConfig
from .infer import load_model_for_inference, predict_image
from .train import train_model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pneumonia detection from chest X-ray images (CNN).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train subcommand
    train_p = subparsers.add_parser("train", help="Train the model")
    train_p.add_argument("--data-dir", type=Path, required=True, help="Root of chest_xray dataset")
    train_p.add_argument("--output-dir", type=Path, default=Path("artifacts/exp1"))
    train_p.add_argument("--epochs", type=int, default=10)
    train_p.add_argument("--batch-size", type=int, default=32)
    train_p.add_argument("--lr", type=float, default=1e-4)
    train_p.add_argument("--weight-decay", type=float, default=1e-4)

    # Predict subcommand
    pred_p = subparsers.add_parser("predict", help="Run inference on a single image")
    pred_p.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint (.pt)")
    pred_p.add_argument("--image", type=Path, required=True, help="Path to input chest X-ray image")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.command == "train":
        cfg = TrainingConfig(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        metrics = train_model(cfg)
        print("Best metrics:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    elif args.command == "predict":
        cfg = TrainingConfig()
        model = load_model_for_inference(cfg, args.checkpoint)

        label, prob = predict_image(cfg, model, args.image)
        print(f"Predicted class : {label}")
        print(f"Probability      : {prob:.4f}")


if __name__ == "__main__":  # pragma: no cover
    main()