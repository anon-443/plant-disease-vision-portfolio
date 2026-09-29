"""Transfer-learning model factory."""

from __future__ import annotations

from torch import nn
from torchvision import models

SUPPORTED_MODELS = ("mobilenet_v3_small", "efficientnet_b0")


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model {name!r}; choose one of {SUPPORTED_MODELS}")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")

    if name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
