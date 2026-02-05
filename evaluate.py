"""Evaluate a trained segmentation model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import GeoTiffSegmentationDataset, CLASS_NAMES
from src.model import make_deeplabv3
from src.metrics import SegmentationMetrics
from src.utils import device_from_arg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate segmentation model")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    p.add_argument("--data_dir", type=str, required=True, help="Data directory with images/ and annotations/")
    p.add_argument("--image_subdir", type=str, default="images")
    p.add_argument("--mask_subdir", type=str, default="annotations")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--in_channels", type=int, default=8)
    p.add_argument("--num_classes", type=int, default=7)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    return p.parse_args()


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    class_names: list[str],
) -> dict:
    """
    Evaluate model on dataset.

    Returns:
        Dictionary with all metrics
    """
    model.eval()
    metrics = SegmentationMetrics(num_classes=num_classes, class_names=class_names)

    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch.image.to(device)
        masks = batch.mask  # Keep on CPU for metrics

        logits = model(images)["out"]
        preds = logits.argmax(dim=1).cpu()

        metrics.update(preds, masks)

    return metrics.compute(), metrics.summary()


def main():
    args = parse_args()
    device = device_from_arg(args.device)

    print(f"Device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = make_deeplabv3(
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        pretrained=False
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)

    # Print checkpoint info if available
    if "epoch" in ckpt:
        print(f"Checkpoint from epoch: {ckpt['epoch']}")
    if "val_acc" in ckpt:
        print(f"Checkpoint val accuracy: {ckpt['val_acc']:.2f}%")

    # Load dataset
    data_dir = Path(args.data_dir)
    img_dir = data_dir / args.image_subdir
    mask_dir = data_dir / args.mask_subdir

    dataset = GeoTiffSegmentationDataset(
        img_dir, mask_dir, size=(args.size, args.size)
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    print(f"Dataset size: {len(dataset)} samples")
    print()

    # Evaluate
    class_names = CLASS_NAMES if hasattr(dataset, "CLASS_NAMES") else [
        "Background", "Seagrass", "Coral", "Sand", "Rock", "Algae", "Deep Water"
    ]

    results, summary = evaluate(
        model, dataloader, device, args.num_classes, class_names
    )

    # Print results
    print()
    print(summary)

    # Save to JSON if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy arrays to lists for JSON serialization
        json_results = {
            "pixel_accuracy": results["pixel_accuracy"],
            "miou": results["miou"],
            "macro_f1": results["macro_f1"],
            "per_class_iou": results["per_class_iou"].tolist(),
            "per_class_f1": results["per_class_f1"].tolist(),
            "per_class_precision": results["per_class_precision"].tolist(),
            "per_class_recall": results["per_class_recall"].tolist(),
            "class_names": class_names,
            "checkpoint": args.checkpoint,
        }

        with open(output_path, "w") as f:
            json.dump(json_results, f, indent=2)

        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
