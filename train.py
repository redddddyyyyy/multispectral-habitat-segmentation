from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import GeoTiffSegmentationDataset
from src.model import make_deeplabv3
from src.utils import split_dataset, ensure_dir, device_from_arg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True, help="Folder containing images/ and annotations/")
    p.add_argument("--image_subdir", type=str, default="images")
    p.add_argument("--mask_subdir", type=str, default="annotations")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--in_channels", type=int, default=8)
    p.add_argument("--num_classes", type=int, default=7)
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        imgs = batch.image.to(device)
        masks = batch.mask.to(device)
        logits = model(imgs)["out"]
        preds = logits.argmax(dim=1)
        correct += (preds == masks).sum().item()
        total += masks.numel()
    return 100.0 * correct / max(total, 1)


def main() -> None:
    args = parse_args()
    device = device_from_arg(args.device)
    out_dir = ensure_dir(args.out_dir)

    data_dir = Path(args.data_dir)
    img_dir = data_dir / args.image_subdir
    mask_dir = data_dir / args.mask_subdir

    ds = GeoTiffSegmentationDataset(img_dir, mask_dir, size=(args.size, args.size))
    train_ds, val_ds = split_dataset(ds, train_frac=0.8, seed=args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type=="cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type=="cuda"))

    model = make_deeplabv3(in_channels=args.in_channels, num_classes=args.num_classes, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = -1.0
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        running_loss = 0.0

        for batch in pbar:
            imgs = batch.image.to(device)
            masks = batch.mask.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)["out"]
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / max(1, pbar.n))

        val_acc = evaluate(model, val_loader, device)
        print(f"Val pixel-acc: {val_acc:.2f}%")

        torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc, "args": vars(args)}, last_path)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc, "args": vars(args)}, best_path)
            print(f"Saved best -> {best_path} (acc {best_acc:.2f}%)")

    print("Done.")


if __name__ == "__main__":
    main()
