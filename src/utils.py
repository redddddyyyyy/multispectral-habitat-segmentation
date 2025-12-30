from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import random_split


def split_dataset(dataset, train_frac: float = 0.8, seed: int = 42):
    n = len(dataset)
    n_train = int(train_frac * n)
    n_val = n - n_train
    g = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val], generator=g)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def device_from_arg(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
