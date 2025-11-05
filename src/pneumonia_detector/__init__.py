#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Artificial Intelligence in Pneumonia Detection
File: __init__.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-05
Updated: 2025-11-05
License: MIT License (see LICENSE file for details)
=

Description:
Package initialization and public API for the pneumonia_detector project.

Usage:
from pneumonia_detector import __version__, TrainingConfig, build_model

Notes:
- This file exposes key objects for convenient imports.
=================================================================================================================
"""

from .config import TrainingConfig
from .model import build_model

__all__ = ["__version__", "TrainingConfig", "build_model"]

__version__ = "0.1.0"