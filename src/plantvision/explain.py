"""Grad-CAM implementation for torchvision feature extractors."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _last_conv_layer(model: torch.nn.Module) -> torch.nn.Module:
    layers = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
    if not layers:
        raise ValueError("Model has no convolutional layer for Grad-CAM")
    return layers[-1]


def gradcam_overlay(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    image: Image.Image,
    target_class: int | None = None,
    device: str = "cpu",
) -> Image.Image:
    """Return an RGB heatmap overlay for the selected output class."""
    model.eval()
    target_layer = _last_conv_layer(model)
    captured: dict[str, torch.Tensor] = {}

    def capture(_module, _inputs, output):
        captured["activations"] = output
        output.register_hook(lambda grad: captured.__setitem__("gradients", grad))

    handle = target_layer.register_forward_hook(capture)
    model.zero_grad(set_to_none=True)
    logits = model(input_tensor.to(device))
    if target_class is None:
        target_class = int(logits.argmax(dim=1).item())
    logits[0, target_class].backward()
    handle.remove()

    activations = captured["activations"].detach()
    gradients = captured["gradients"].detach()
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
    cam = F.interpolate(
        cam, size=(image.height, image.width), mode="bilinear", align_corners=False
    )[0, 0]
    cam = cam.cpu().numpy()
    cam -= cam.min()
    cam /= max(float(cam.max()), 1e-8)

    heat = np.zeros((*cam.shape, 3), dtype=np.uint8)
    heat[..., 0] = np.clip(255 * np.minimum(1.0, 2.0 * cam), 0, 255)
    heat[..., 1] = np.clip(255 * np.maximum(0.0, 2.0 * cam - 0.5), 0, 255)
    heat_img = Image.fromarray(heat, mode="RGB")
    base = image.convert("RGB").resize(heat_img.size)
    return Image.blend(base, heat_img, alpha=0.38)
