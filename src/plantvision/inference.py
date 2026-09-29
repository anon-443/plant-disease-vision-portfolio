"""Load a trained bundle and predict plant disease classes for images."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from plantvision.explain import gradcam_overlay
from plantvision.models import build_model


class PlantDiseasePredictor:
    def __init__(self, checkpoint_path: str | Path, device: str = "auto") -> None:
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if self.device == "auto":
            self.device = "cpu"
        self.bundle = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.class_names = self.bundle["class_names"]
        self.model = build_model(
            self.bundle["model_name"], self.bundle["num_classes"], pretrained=False
        )
        self.model.load_state_dict(self.bundle["state_dict"])
        self.model.to(self.device).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.bundle["image_size"], self.bundle["image_size"])),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.bundle["mean"], std=self.bundle["std"]),
            ]
        )

    def predict(self, image: Image.Image, top_k: int = 3, explain: bool = False) -> dict:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
        count = min(top_k, len(self.class_names))
        values, indices = torch.topk(probabilities, k=count)
        ranked = [
            {"class": self.class_names[int(index)], "confidence": float(value)}
            for value, index in zip(values.cpu(), indices.cpu())
        ]
        result = {"predictions": ranked, "explanation": None}
        if explain:
            result["explanation"] = gradcam_overlay(
                self.model, tensor, image, int(indices[0]), self.device
            )
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--save-gradcam", help="Optional path to save a Grad-CAM overlay")
    args = parser.parse_args()
    predictor = PlantDiseasePredictor(args.model_path, args.device)
    with Image.open(args.image) as image:
        result = predictor.predict(image, top_k=args.top_k, explain=bool(args.save_gradcam))
    for item in result["predictions"]:
        print(f"{item['class']}: {item['confidence']:.4f}")
    if args.save_gradcam and result["explanation"] is not None:
        result["explanation"].save(args.save_gradcam)
        print(f"Grad-CAM saved to {args.save_gradcam}")


if __name__ == "__main__":
    main()
