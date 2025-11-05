# Artificial Intelligence in Pneumonia Detection

Convolutional Neural Network (CNN) project for **automatic pneumonia detection** from chest X‑ray images.

The codebase is structured as a clean Python package (src layout) with:

- **Data pipeline** for Chest X-ray images (e.g. Kaggle *Chest X-Ray Images (Pneumonia)* dataset)
- **Transfer learning model** based on ResNet-18 (PyTorch)
- **Training & evaluation** utilities with accuracy/F1 metrics
- **CLI interface** for training and single-image inference
- **Tests + CI** (pytest, ruff, black)

> Author: **Mobin Yousefi** – [github.com/mobinyousefi-cs](https://github.com/mobinyousefi-cs)

---

## 1. Dataset

You can use the popular Kaggle dataset:

> **Chest X-Ray Images (Pneumonia)** by Paul Mooney  
> URL: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

After downloading and extracting, you should have a directory layout:

```text
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

Place this folder under `data/` or configure a custom path via CLI.

---

## 2. Installation

### 2.1. Create & activate virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# or
.venv\\Scripts\\activate      # Windows
```

### 2.2. Install package (editable) with dev tools

```bash
pip install --upgrade pip
pip install -e .[dev]
```

This installs:

- `torch`, `torchvision`, `numpy`, `pandas`, `scikit-learn`, `tqdm`, `Pillow`
- Dev tools: `pytest`, `black`, `ruff`

---

## 3. Quickstart

### 3.1. Train model

```bash
python -m pneumonia_detector.cli train \
  --data-dir data/chest_xray \
  --output-dir artifacts/exp1 \
  --epochs 10 \
  --batch-size 32 \
  --lr 1e-4
```

This will:

- Create `artifacts/exp1/` directory
- Save best model checkpoint as `best_model.pt`
- Log metrics to `metrics.json`

### 3.2. Run inference on a single image

```bash
python -m pneumonia_detector.cli predict \
  --checkpoint artifacts/exp1/best_model.pt \
  --image path/to/chest_xray/test/NORMAL/IM-0001-0001.jpeg
```

Example output:

```text
Predicted class : NORMAL
Probability      : 0.97
```

---

## 4. Project Structure

```text
src/pneumonia_detector/
├── __init__.py      # package metadata
├── config.py        # configuration dataclasses & defaults
├── data.py          # dataset & dataloader utilities
├── model.py         # CNN / ResNet model definition
├── train.py         # training + evaluation loops
├── infer.py         # single-image inference helpers
└── cli.py           # command-line interface
```

Tests and CI:

```text
tests/
├── test_imports.py
├── test_model.py
└── test_data_config.py

.github/workflows/ci.yml  # lint + tests on push/PR
```

---

## 5. Running tests & linters

```bash
pytest
ruff check src tests
black --check src tests
```

---

## 6. Notes

- Default model: **ResNet-18** with ImageNet weights and a custom final layer for 2 classes.
- This project is intentionally simple, readable, and easy to extend (e.g., mixup, focal loss, Grad-CAM, etc.).
- Always validate models carefully and **never** use them as a substitute for professional medical diagnosis.