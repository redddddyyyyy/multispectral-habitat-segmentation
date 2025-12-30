from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
from rasterio.errors import RasterioIOError
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


# If your masks are RGB-coded, map RGB -> class index here.
# Edit this to match your dataset.
CLASS_COLORS: Dict[Tuple[int, int, int], int] = {
    (30, 30, 30): 0,
    (34, 70, 34): 1,
    (120, 50, 20): 2,
    (100, 100, 40): 3,
    (200, 180, 130): 4,
    (210, 150, 100): 5,
    (150, 200, 255): 6,
}


def minmax_normalize(img: np.ndarray) -> np.ndarray:
    """Per-tile min-max normalization."""
    mn = img.min()
    mx = img.max()
    return (img - mn) / (mx - mn + 1e-8)


def rgb_mask_to_ids(mask_rgb: np.ndarray) -> np.ndarray:
    """Convert HxWx3 RGB mask to HxW class-id mask."""
    h, w, _ = mask_rgb.shape
    out = np.zeros((h, w), dtype=np.int64)
    for rgb, cls in CLASS_COLORS.items():
        rgb_arr = np.array(rgb, dtype=mask_rgb.dtype)
        out[np.all(mask_rgb == rgb_arr, axis=-1)] = cls
    return out


@dataclass
class SegSample:
    image: torch.Tensor  # [C,H,W]
    mask: torch.Tensor   # [H,W]


class GeoTiffSegmentationDataset(Dataset):
    """Reads multispectral GeoTIFF images + corresponding masks."""

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        size: Tuple[int, int] = (256, 256),
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.size = size

        self.images = sorted([p for p in self.image_dir.glob("*.tif")])
        self.masks = sorted([p for p in self.mask_dir.glob("*.tif")])

        if len(self.images) == 0:
            raise FileNotFoundError(f"No .tif images found in {self.image_dir}")
        if len(self.images) != len(self.masks):
            # Not fatal, but usually a problem.
            raise ValueError(f"Image/mask count mismatch: {len(self.images)} vs {len(self.masks)}")

    def __len__(self) -> int:
        return len(self.images)

    def _read_image(self, path: Path) -> np.ndarray:
        try:
            with rasterio.open(path) as src:
                arr = src.read().astype(np.float32)  # [C,H,W]
        except RasterioIOError as e:
            raise RuntimeError(f"Failed reading image: {path}") from e

        arr = np.transpose(arr, (1, 2, 0))  # -> [H,W,C]
        arr = minmax_normalize(arr)
        return arr

    def _read_mask(self, path: Path) -> np.ndarray:
        with rasterio.open(path) as src:
            arr = src.read()  # could be [1,H,W] label ids or [3,H,W] RGB

        if arr.shape[0] == 1:
            mask = arr[0].astype(np.int64)  # [H,W]
            return mask

        if arr.shape[0] == 3:
            rgb = np.transpose(arr, (1, 2, 0)).astype(np.uint8)  # [H,W,3]
            return rgb_mask_to_ids(rgb)

        # Fallback: take the first band
        return arr[0].astype(np.int64)

    def __getitem__(self, idx: int) -> SegSample:
        img = self._read_image(self.images[idx])
        mask = self._read_mask(self.masks[idx])

        img_t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)  # [C,H,W]
        mask_t = torch.from_numpy(mask.astype(np.int64))                   # [H,W]

        # Resize
        img_t = TF.resize(img_t, self.size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        # For masks use nearest; add channel then squeeze
        mask_t = TF.resize(mask_t.unsqueeze(0), self.size, interpolation=InterpolationMode.NEAREST).squeeze(0)

        return SegSample(image=img_t, mask=mask_t)
