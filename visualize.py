"""Visualize segmentation predictions."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader

from src.dataset import GeoTiffSegmentationDataset
from src.model import make_deeplabv3
from src.utils import ensure_dir, device_from_arg


# Color palette for visualization (RGB, 0-1 range)
CLASS_COLORS_VIS = [
    [0.12, 0.12, 0.12],    # 0: Background - dark gray
    [0.13, 0.55, 0.13],    # 1: Seagrass - forest green
    [0.94, 0.50, 0.50],    # 2: Coral - light coral
    [0.96, 0.87, 0.70],    # 3: Sand - wheat
    [0.55, 0.27, 0.07],    # 4: Rock - saddle brown
    [0.60, 0.80, 0.20],    # 5: Algae - yellow green
    [0.25, 0.41, 0.88],    # 6: Deep Water - royal blue
]

CLASS_NAMES = [
    "Background",
    "Seagrass",
    "Coral",
    "Sand",
    "Rock",
    "Algae",
    "Deep Water",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize segmentation predictions")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    p.add_argument("--data_dir", type=str, required=True, help="Data directory")
    p.add_argument("--image_subdir", type=str, default="images")
    p.add_argument("--mask_subdir", type=str, default="annotations")
    p.add_argument("--out_dir", type=str, default="assets", help="Output directory for visualizations")
    p.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--in_channels", type=int, default=8)
    p.add_argument("--num_classes", type=int, default=7)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sample selection")
    return p.parse_args()


def mask_to_rgb(mask: np.ndarray, colors: list) -> np.ndarray:
    """Convert class ID mask to RGB image."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    for class_id, color in enumerate(colors):
        rgb[mask == class_id] = color

    return rgb


def multispectral_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert multispectral image to pseudo-RGB for visualization.

    Uses bands that approximate R, G, B if available,
    otherwise uses first 3 bands.
    """
    # image shape: [C, H, W]
    if image.shape[0] >= 3:
        # Use first 3 bands as RGB approximation
        rgb = image[:3].transpose(1, 2, 0)  # [H, W, 3]
    else:
        # Grayscale fallback
        rgb = np.stack([image[0]] * 3, axis=-1)

    # Normalize to 0-1
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

    return rgb


def create_comparison_figure(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    colors: list,
    class_names: list,
    sample_idx: int,
) -> plt.Figure:
    """Create a side-by-side comparison figure."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Input image (pseudo-RGB)
    rgb = multispectral_to_rgb(image)
    axes[0].imshow(rgb)
    axes[0].set_title("Input (Pseudo-RGB)", fontsize=12)
    axes[0].axis("off")

    # Ground truth
    gt_rgb = mask_to_rgb(gt_mask, colors)
    axes[1].imshow(gt_rgb)
    axes[1].set_title("Ground Truth", fontsize=12)
    axes[1].axis("off")

    # Prediction
    pred_rgb = mask_to_rgb(pred_mask, colors)
    axes[2].imshow(pred_rgb)
    axes[2].set_title("Prediction", fontsize=12)
    axes[2].axis("off")

    # Overlay: prediction on image
    overlay = rgb * 0.5 + pred_rgb * 0.5
    axes[3].imshow(overlay)
    axes[3].set_title("Overlay", fontsize=12)
    axes[3].axis("off")

    # Add legend
    cmap = ListedColormap(colors)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[i], edgecolor="black", linewidth=0.5)
        for i in range(len(class_names))
    ]
    fig.legend(
        legend_handles,
        class_names,
        loc="lower center",
        ncol=len(class_names),
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(f"Sample {sample_idx}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    return fig


def create_grid_figure(
    samples: list[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    colors: list,
    class_names: list,
) -> plt.Figure:
    """Create a grid of all samples for README."""
    n_samples = len(samples)
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i, (image, gt_mask, pred_mask) in enumerate(samples):
        rgb = multispectral_to_rgb(image)
        gt_rgb = mask_to_rgb(gt_mask, colors)
        pred_rgb = mask_to_rgb(pred_mask, colors)

        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title("Input" if i == 0 else "", fontsize=11)
        axes[i, 0].axis("off")
        axes[i, 0].set_ylabel(f"Sample {i+1}", fontsize=11, rotation=90, labelpad=10)

        axes[i, 1].imshow(gt_rgb)
        axes[i, 1].set_title("Ground Truth" if i == 0 else "", fontsize=11)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred_rgb)
        axes[i, 2].set_title("Prediction" if i == 0 else "", fontsize=11)
        axes[i, 2].axis("off")

    # Add legend at bottom
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[i], edgecolor="black", linewidth=0.5)
        for i in range(len(class_names))
    ]
    fig.legend(
        legend_handles,
        class_names,
        loc="lower center",
        ncol=len(class_names),
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, 0.02),
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)

    return fig


@torch.inference_mode()
def main():
    args = parse_args()
    device = device_from_arg(args.device)
    out_dir = ensure_dir(args.out_dir)

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
    model.eval()

    # Load dataset
    data_dir = Path(args.data_dir)
    dataset = GeoTiffSegmentationDataset(
        data_dir / args.image_subdir,
        data_dir / args.mask_subdir,
        size=(args.size, args.size)
    )

    # Select random samples
    torch.manual_seed(args.seed)
    indices = torch.randperm(len(dataset))[:args.num_samples].tolist()

    print(f"Visualizing {args.num_samples} samples: {indices}")

    # Generate predictions
    samples = []
    for idx in indices:
        sample = dataset[idx]
        image = sample.image  # [C, H, W]
        gt_mask = sample.mask.numpy()  # [H, W]

        # Predict
        logits = model(image.unsqueeze(0).to(device))["out"]
        pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        samples.append((image.numpy(), gt_mask, pred_mask))

        # Save individual comparison
        fig = create_comparison_figure(
            image.numpy(), gt_mask, pred_mask,
            CLASS_COLORS_VIS, CLASS_NAMES, idx
        )
        fig.savefig(out_dir / f"comparison_{idx:04d}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: comparison_{idx:04d}.png")

    # Save grid figure for README
    grid_fig = create_grid_figure(samples, CLASS_COLORS_VIS, CLASS_NAMES)
    grid_fig.savefig(out_dir / "sample_predictions.png", dpi=150, bbox_inches="tight")
    plt.close(grid_fig)
    print(f"  Saved: sample_predictions.png")

    print(f"\nAll visualizations saved to: {out_dir}")


if __name__ == "__main__":
    main()
