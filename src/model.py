from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights


def make_deeplabv3(in_channels: int = 8, num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    """DeepLabV3-ResNet50 adapted for multispectral input + custom number of classes."""
    weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
    model = deeplabv3_resnet50(weights=weights)

    # --- Adapt first conv to in_channels ---
    old_conv: nn.Conv2d = model.backbone.conv1
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )

    with torch.no_grad():
        if old_conv.weight.shape[1] == 3 and in_channels >= 3:
            new_conv.weight[:, :3] = old_conv.weight
            if in_channels > 3:
                # Initialize extra channels with mean of RGB filters
                mean_w = old_conv.weight.mean(dim=1, keepdim=True)  # [out,1,kh,kw]
                new_conv.weight[:, 3:] = mean_w.repeat(1, in_channels - 3, 1, 1)
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
    model.backbone.conv1 = new_conv

    # --- Replace classifier head ---
    # torchvision classifier ends with a Conv2d to num_classes.
    model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)

    # If aux_classifier exists, match its output too.
    if getattr(model, "aux_classifier", None) is not None:
        model.aux_classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)

    return model


@torch.inference_mode()
def predict(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return predicted class IDs [B,H,W]."""
    out = model(x)["out"]
    return out.argmax(dim=1)
