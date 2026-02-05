# Multispectral Coastal Habitat Segmentation

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Semantic segmentation of coastal habitats from **8-band multispectral GeoTIFF imagery** into **7 habitat classes** using a U-Net architecture with VGG-style encoder.

---

## Overview

This project implements a deep learning pipeline for pixel-wise classification of coastal satellite imagery, addressing challenges like:

- **Multispectral input** (8 channels vs standard RGB)
- **Class imbalance** via weighted loss functions
- **Noisy boundaries** with morphological post-processing

---

## Model Architecture

```
Input (8-band GeoTIFF)
    │
    ▼
┌─────────────────┐
│  VGG-style      │
│  Encoder        │──────┐
│  (Downsampling) │      │ Skip Connections
└────────┬────────┘      │
         │               │
         ▼               │
┌─────────────────┐      │
│    Bottleneck   │      │
└────────┬────────┘      │
         │               │
         ▼               │
┌─────────────────┐      │
│  U-Net Decoder  │◄─────┘
│  (Upsampling)   │
└────────┬────────┘
         │
         ▼
Output (7-class segmentation mask)
```

---

## Quick Start

### 1. Environment Setup

```bash
# Using conda (recommended)
conda env create -f environment.yml
conda activate habitat-seg

# Or using pip
pip install -r requirements.txt
```

### 2. Data Preparation

```
data/
└── raw_labeled_data/
    ├── images/        # 8-band GeoTIFF tiles (.tif)
    └── annotations/   # Label masks (.tif)
```

### 3. Training

```bash
python train.py --data_dir data/raw_labeled_data \
                --epochs 50 \
                --batch_size 16 \
                --lr 1e-4
```

### 4. Inference

```bash
python infer.py --checkpoint checkpoints/best.pt \
                --image_dir data/raw_labeled_data/images \
                --out_dir outputs
```

---

## Techniques Used

| Challenge | Solution |
|-----------|----------|
| Class imbalance | Weighted cross-entropy loss |
| Limited data | Data augmentation (flip, rotate, scale) |
| Noisy predictions | Morphological post-processing |
| Boundary refinement | CRF post-processing |

---

## Evaluation Metrics

- **mIoU** (mean Intersection over Union)
- **F1 Score** (per-class and macro)
- **Confusion Matrix**
- **Boundary Accuracy**

---

## Repository Structure

```
├── src/
│   ├── dataset.py      # Data loading + augmentation
│   └── model.py        # U-Net architecture
├── train.py            # Training entrypoint
├── infer.py            # Inference entrypoint
├── notebooks/          # Exploration notebooks
├── report/             # Project report (PDF)
├── environment.yml     # Conda environment
└── requirements.txt    # Pip requirements
```

---

## Habitat Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | Background | Non-habitat areas |
| 1 | Seagrass | Submerged aquatic vegetation |
| 2 | Coral | Coral reef structures |
| 3 | Sand | Sandy substrate |
| 4 | Rock | Rocky substrate |
| 5 | Algae | Algal coverage |
| 6 | Deep Water | Deep water regions |

*Note: Edit `CLASS_COLORS` in `src/dataset.py` if your dataset uses different label mappings.*

---

## Documentation

- [`report/PROJECT-3_REPORT.pdf`](report/PROJECT-3_REPORT.pdf) — Full project report with methodology and results

---

## License

MIT
