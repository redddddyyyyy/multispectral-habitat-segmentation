"""Data augmentation transforms for segmentation."""
from __future__ import annotations

import random
from typing import Tuple

import torch
import torchvision.transforms.functional as TF


class SegmentationAugmentation:
    """
    Joint augmentation for image-mask pairs.

    Applies the same spatial transforms to both image and mask,
    ensuring consistency for segmentation training.

    Args:
        h_flip_prob: Probability of horizontal flip
        v_flip_prob: Probability of vertical flip
        rotate_prob: Probability of rotation
        rotate_degrees: Max rotation angle (will rotate by random angle in [-degrees, degrees])
        scale_prob: Probability of random scaling
        scale_range: Tuple of (min_scale, max_scale)
        brightness_jitter: Max brightness adjustment (0 = no jitter)
        contrast_jitter: Max contrast adjustment (0 = no jitter)
    """

    def __init__(
        self,
        h_flip_prob: float = 0.5,
        v_flip_prob: float = 0.5,
        rotate_prob: float = 0.3,
        rotate_degrees: float = 15.0,
        scale_prob: float = 0.3,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        brightness_jitter: float = 0.2,
        contrast_jitter: float = 0.2,
    ):
        self.h_flip_prob = h_flip_prob
        self.v_flip_prob = v_flip_prob
        self.rotate_prob = rotate_prob
        self.rotate_degrees = rotate_degrees
        self.scale_prob = scale_prob
        self.scale_range = scale_range
        self.brightness_jitter = brightness_jitter
        self.contrast_jitter = contrast_jitter

    def __call__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply augmentations to image and mask.

        Args:
            image: Image tensor [C, H, W]
            mask: Mask tensor [H, W]

        Returns:
            Tuple of (augmented_image, augmented_mask)
        """
        # Horizontal flip
        if random.random() < self.h_flip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask.unsqueeze(0)).squeeze(0)

        # Vertical flip
        if random.random() < self.v_flip_prob:
            image = TF.vflip(image)
            mask = TF.vflip(mask.unsqueeze(0)).squeeze(0)

        # Random rotation
        if random.random() < self.rotate_prob:
            angle = random.uniform(-self.rotate_degrees, self.rotate_degrees)
            image = TF.rotate(image, angle, fill=0)
            # Use nearest interpolation for mask to preserve class labels
            mask = TF.rotate(
                mask.unsqueeze(0).float(),
                angle,
                interpolation=TF.InterpolationMode.NEAREST,
                fill=0
            ).squeeze(0).long()

        # Random scale (resize then crop/pad back to original size)
        if random.random() < self.scale_prob:
            _, h, w = image.shape
            scale = random.uniform(*self.scale_range)
            new_h, new_w = int(h * scale), int(w * scale)

            image = TF.resize(image, [new_h, new_w], antialias=True)
            mask = TF.resize(
                mask.unsqueeze(0),
                [new_h, new_w],
                interpolation=TF.InterpolationMode.NEAREST
            ).squeeze(0)

            # Crop or pad back to original size
            if scale > 1.0:
                # Crop center
                top = (new_h - h) // 2
                left = (new_w - w) // 2
                image = TF.crop(image, top, left, h, w)
                mask = TF.crop(mask.unsqueeze(0), top, left, h, w).squeeze(0)
            else:
                # Pad to original size
                pad_h = h - new_h
                pad_w = w - new_w
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                image = TF.pad(image, [pad_left, pad_top, pad_right, pad_bottom], fill=0)
                mask = TF.pad(
                    mask.unsqueeze(0),
                    [pad_left, pad_top, pad_right, pad_bottom],
                    fill=0
                ).squeeze(0)

        # Brightness and contrast jitter (image only, not mask)
        if self.brightness_jitter > 0:
            factor = 1.0 + random.uniform(-self.brightness_jitter, self.brightness_jitter)
            image = image * factor
            image = torch.clamp(image, 0, 1)

        if self.contrast_jitter > 0:
            factor = 1.0 + random.uniform(-self.contrast_jitter, self.contrast_jitter)
            mean = image.mean()
            image = (image - mean) * factor + mean
            image = torch.clamp(image, 0, 1)

        return image, mask


class ValidationTransform:
    """No-op transform for validation/testing."""

    def __call__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return image, mask


def get_train_augmentation(
    h_flip: bool = True,
    v_flip: bool = True,
    rotate: bool = True,
    scale: bool = True,
    color_jitter: bool = True,
) -> SegmentationAugmentation:
    """
    Get standard training augmentation.

    Args:
        h_flip: Enable horizontal flip
        v_flip: Enable vertical flip
        rotate: Enable random rotation
        scale: Enable random scaling
        color_jitter: Enable brightness/contrast jitter
    """
    return SegmentationAugmentation(
        h_flip_prob=0.5 if h_flip else 0.0,
        v_flip_prob=0.5 if v_flip else 0.0,
        rotate_prob=0.3 if rotate else 0.0,
        rotate_degrees=15.0,
        scale_prob=0.3 if scale else 0.0,
        scale_range=(0.8, 1.2),
        brightness_jitter=0.2 if color_jitter else 0.0,
        contrast_jitter=0.2 if color_jitter else 0.0,
    )


def get_val_transform() -> ValidationTransform:
    """Get validation transform (no augmentation)."""
    return ValidationTransform()
