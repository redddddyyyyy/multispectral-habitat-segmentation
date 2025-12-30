from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import rasterio
from rasterio.transform import Affine

from src.dataset import GeoTiffSegmentationDataset
from src.model import make_deeplabv3
from src.utils import ensure_dir, device_from_arg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--image_dir", type=str, required=True)
    p.add_argument("--mask_dir", type=str, default=None, help="Optional, only used to construct dataset. Not required for inference.")
    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--in_channels", type=int, default=8)
    p.add_argument("--num_classes", type=int, default=7)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    device = device_from_arg(args.device)
    out_dir = ensure_dir(args.out_dir)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = make_deeplabv3(in_channels=args.in_channels, num_classes=args.num_classes, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # If mask_dir is not provided, create a dummy dataset-like loader by reusing dataset with same folder.
    mask_dir = args.mask_dir or args.image_dir  # will be ignored if masks not used
    ds = GeoTiffSegmentationDataset(args.image_dir, mask_dir, size=(args.size, args.size))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    for i, batch in enumerate(tqdm(loader, desc="Infer")):
        imgs = batch.image.to(device)
        logits = model(imgs)["out"]
        preds = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)  # [B,H,W]

        for b in range(preds.shape[0]):
            out_path = Path(out_dir) / f"pred_{i*args.batch_size + b:05d}.tif"
            pred = preds[b]

            # Write as GeoTIFF with identity transform (no geo metadata available here).
            with rasterio.open(
                out_path,
                "w",
                driver="GTiff",
                height=pred.shape[0],
                width=pred.shape[1],
                count=1,
                dtype=pred.dtype,
                transform=Affine.identity(),
            ) as dst:
                dst.write(pred, 1)

    print(f"Saved predictions to {out_dir}")


if __name__ == "__main__":
    main()
