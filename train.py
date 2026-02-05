"""Training script for multispectral habitat segmentation."""
from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from src.dataset import GeoTiffSegmentationDataset
from src.model import make_deeplabv3
from src.metrics import SegmentationMetrics
from src.augmentations import get_train_augmentation, get_val_transform
from src.utils import split_dataset, ensure_dir, device_from_arg

# Optional TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


CLASS_NAMES = [
    "Background", "Seagrass", "Coral", "Sand", "Rock", "Algae", "Deep Water"
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train segmentation model")

    # Data
    p.add_argument("--data_dir", type=str, required=True, help="Data directory")
    p.add_argument("--image_subdir", type=str, default="images")
    p.add_argument("--mask_subdir", type=str, default="annotations")

    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=2)

    # Model
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--in_channels", type=int, default=8)
    p.add_argument("--num_classes", type=int, default=7)

    # Features
    p.add_argument("--use_class_weights", action="store_true", help="Use class weighting for imbalanced data")
    p.add_argument("--augment", action="store_true", help="Enable data augmentation")
    p.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")

    # Early stopping
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience (0 to disable)")

    # LR scheduler
    p.add_argument("--lr_scheduler", type=str, default="cosine", choices=["none", "cosine", "step"])
    p.add_argument("--T_0", type=int, default=10, help="Cosine annealing T_0")

    # Output
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--run_name", type=str, default=None, help="Name for this run")

    # Misc
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def compute_class_weights(dataset: GeoTiffSegmentationDataset, num_classes: int) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from dataset.

    Returns:
        Tensor of class weights [num_classes]
    """
    print("Computing class weights...")
    class_counts = torch.zeros(num_classes, dtype=torch.float64)

    for i in tqdm(range(len(dataset)), desc="Scanning dataset"):
        sample = dataset[i]
        mask = sample.mask
        for c in range(num_classes):
            class_counts[c] += (mask == c).sum().item()

    # Inverse frequency weighting
    total = class_counts.sum()
    weights = total / (class_counts + 1e-6)

    # Normalize so weights sum to num_classes
    weights = weights / weights.sum() * num_classes

    print("Class distribution:")
    for i, name in enumerate(CLASS_NAMES[:num_classes]):
        pct = 100.0 * class_counts[i] / total
        print(f"  {name}: {class_counts[i]:.0f} pixels ({pct:.2f}%) -> weight {weights[i]:.3f}")

    return weights.float()


class AugmentedDataset(torch.utils.data.Dataset):
    """Wrapper that applies augmentation to a dataset."""

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image, mask = self.transform(sample.image, sample.mask)
        # Return same structure
        from src.dataset import SegSample
        return SegSample(image=image, mask=mask)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict:
    """Evaluate model and return metrics."""
    model.eval()
    metrics = SegmentationMetrics(num_classes=num_classes, class_names=CLASS_NAMES)

    for batch in loader:
        imgs = batch.image.to(device)
        masks = batch.mask

        logits = model(imgs)["out"]
        preds = logits.argmax(dim=1).cpu()

        metrics.update(preds, masks)

    return metrics.compute()


def main() -> None:
    args = parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = device_from_arg(args.device)
    out_dir = ensure_dir(args.out_dir)

    # Run name for logging
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("MULTISPECTRAL HABITAT SEGMENTATION - TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Run name: {run_name}")
    print(f"Output dir: {out_dir}")
    print()

    # Data
    data_dir = Path(args.data_dir)
    img_dir = data_dir / args.image_subdir
    mask_dir = data_dir / args.mask_subdir

    base_ds = GeoTiffSegmentationDataset(img_dir, mask_dir, size=(args.size, args.size))
    train_ds_base, val_ds_base = split_dataset(base_ds, train_frac=0.8, seed=args.seed)

    # Augmentation
    if args.augment:
        train_transform = get_train_augmentation()
        print("Data augmentation: ENABLED")
    else:
        train_transform = get_val_transform()
        print("Data augmentation: DISABLED")

    train_ds = AugmentedDataset(train_ds_base, train_transform)
    val_ds = AugmentedDataset(val_ds_base, get_val_transform())

    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print()

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    # Model
    model = make_deeplabv3(
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        pretrained=True
    ).to(device)

    # Class weights
    if args.use_class_weights:
        weights = compute_class_weights(base_ds, args.num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print("\nClass weighting: ENABLED")
    else:
        criterion = nn.CrossEntropyLoss()
        print("\nClass weighting: DISABLED")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # LR Scheduler
    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=2)
        print(f"LR Scheduler: Cosine Annealing (T_0={args.T_0})")
    elif args.lr_scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        print("LR Scheduler: StepLR")
    else:
        print("LR Scheduler: None")

    # TensorBoard
    writer = None
    if args.tensorboard:
        if TENSORBOARD_AVAILABLE:
            log_dir = Path("runs") / run_name
            writer = SummaryWriter(log_dir)
            print(f"TensorBoard: {log_dir}")
        else:
            print("TensorBoard requested but not installed. Skipping.")

    print()
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)

    # Training state
    best_miou = -1.0
    best_epoch = 0
    patience_counter = 0

    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    for epoch in range(1, args.epochs + 1):
        # Training
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
            pbar.set_postfix(loss=f"{running_loss / max(1, pbar.n):.4f}")

        avg_loss = running_loss / len(train_loader)

        # LR scheduler step
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        # Validation
        val_results = evaluate(model, val_loader, device, args.num_classes)
        val_miou = val_results["miou"]
        val_acc = val_results["pixel_accuracy"]

        print(f"Epoch {epoch}: loss={avg_loss:.4f}, val_mIoU={val_miou:.4f}, val_acc={val_acc:.4f}, lr={current_lr:.6f}")

        # TensorBoard logging
        if writer is not None:
            writer.add_scalar("Loss/train", avg_loss, epoch)
            writer.add_scalar("mIoU/val", val_miou, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)
            writer.add_scalar("LR", current_lr, epoch)

            # Per-class IoU
            for i, name in enumerate(CLASS_NAMES[:args.num_classes]):
                writer.add_scalar(f"IoU/{name}", val_results["per_class_iou"][i], epoch)

        # Checkpointing
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_miou": val_miou,
            "val_acc": val_acc,
            "args": vars(args),
        }

        torch.save(checkpoint, last_path)

        if val_miou > best_miou:
            best_miou = val_miou
            best_epoch = epoch
            patience_counter = 0
            torch.save(checkpoint, best_path)
            print(f"  -> New best model! mIoU={best_miou:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {args.patience} epochs without improvement.")
            break

    # Summary
    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best mIoU: {best_miou:.4f} (epoch {best_epoch})")
    print(f"Best model saved to: {best_path}")
    print(f"Last model saved to: {last_path}")

    if writer is not None:
        writer.close()

    print("\nNext steps:")
    print(f"  1. Evaluate: python evaluate.py --checkpoint {best_path} --data_dir {args.data_dir}")
    print(f"  2. Visualize: python visualize.py --checkpoint {best_path} --data_dir {args.data_dir}")


if __name__ == "__main__":
    main()
