"""Train a transfer-learning classifier and evaluate its selected checkpoint."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from plantvision.data import LeafImageDataset, load_plantvillage, sample_per_class
from plantvision.evaluation import classification_metrics, save_evaluation, save_training_curves
from plantvision.models import SUPPORTED_MODELS, build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def transforms_for(image_size: int):
    train = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(18),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    evaluate = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return train, evaluate


def _loader(dataset, transform, batch_size, shuffle=False, workers=2, sampler=None):
    return DataLoader(
        LeafImageDataset(dataset, transform),
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def _run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_targets, all_preds = [], []
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * len(targets)
            all_targets.extend(targets.detach().cpu().tolist())
            all_preds.extend(logits.argmax(1).detach().cpu().tolist())
    return total_loss / len(loader.dataset), all_targets, all_preds


def train(args: argparse.Namespace) -> Path:
    seed_everything(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = load_plantvillage(args.val_fraction, args.seed)
    classes = splits.class_names
    train_split = sample_per_class(splits.train, args.max_train_per_class, args.seed)
    validation_split = sample_per_class(
        splits.validation, args.max_validation_per_class, args.seed + 1
    )
    test_split = sample_per_class(splits.test, args.max_test_per_class, args.seed + 2)
    train_transform, eval_transform = transforms_for(args.image_size)

    train_labels = np.asarray(train_split["label"], dtype=np.int64)
    class_weights = compute_class_weight(
        "balanced", classes=np.arange(len(classes)), y=train_labels
    )
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double), len(sample_weights), replacement=True
    )
    train_loader = _loader(
        train_split, train_transform, args.batch_size, workers=args.workers, sampler=sampler
    )
    val_loader = _loader(validation_split, eval_transform, args.batch_size, workers=args.workers)
    test_loader = _loader(test_split, eval_transform, args.batch_size, workers=args.workers)

    model = build_model(args.model, len(classes), pretrained=True).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )
    best_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    history = []
    bundle_path = output_dir / "model_bundle.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, y_val, pred_val = _run_epoch(model, val_loader, criterion, device)
        val_f1 = float(
            f1_score(
                y_val, pred_val, labels=np.arange(len(classes)), average="macro", zero_division=0
            )
        )
        scheduler.step(val_f1)
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_f1": val_f1,
        }
        history.append(entry)
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_macro_f1={val_f1:.4f}",
            flush=True,
        )
        if val_f1 > best_f1:
            best_f1, best_epoch, stale_epochs = val_f1, epoch, 0
            torch.save(
                {
                    "model_name": args.model,
                    "num_classes": len(classes),
                    "class_names": classes,
                    "image_size": args.image_size,
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                    "state_dict": model.state_dict(),
                    "best_validation_macro_f1": best_f1,
                    "best_epoch": best_epoch,
                },
                bundle_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early stopping after {epoch} epochs", flush=True)
                break

    checkpoint = torch.load(bundle_path, map_location=device, weights_only=False)
    model = build_model(checkpoint["model_name"], checkpoint["num_classes"], pretrained=False).to(
        device
    )
    model.load_state_dict(checkpoint["state_dict"])
    test_loss, y_test, pred_test = _run_epoch(model, test_loader, criterion, device)
    metrics = classification_metrics(y_test, pred_test, list(range(len(classes))))
    metrics.update(
        {
            "model_name": args.model,
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_f1,
            "test_loss": test_loss,
            "test_examples": len(y_test),
            "train_examples": len(train_split),
            "validation_examples": len(validation_split),
            "excluded_train_test_overlap_examples": splits.excluded_train_test_overlap_examples,
            "class_count": len(classes),
            "selection_protocol": "best validation macro-F1; test evaluated once after selection",
        }
    )
    metrics.pop("confusion_matrix")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    # Recompute the matrix for the image without duplicating it in the JSON file.
    from sklearn.metrics import confusion_matrix

    full_metrics = dict(metrics)
    full_metrics["confusion_matrix"] = confusion_matrix(
        y_test, pred_test, labels=list(range(len(classes)))
    ).tolist()
    save_evaluation(full_metrics, classes, output_dir)
    save_training_curves(history, output_dir / "training_curves.png")
    config = vars(args).copy()
    config.update(
        {
            "resolved_device": device,
            "split_sizes": {
                "train": len(train_split),
                "validation": len(validation_split),
                "test": len(test_split),
            },
            "per_class_caps": {
                "train": args.max_train_per_class,
                "validation": args.max_validation_per_class,
                "test": args.max_test_per_class,
            },
            "excluded_train_test_overlap_examples": splits.excluded_train_test_overlap_examples,
            "class_names": classes,
            "selection_metric": "validation macro-F1",
            "test_protocol": "official dataset test split; evaluated after checkpoint selection",
        }
    )
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_class"}, indent=2))
    return bundle_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default="mobilenet_v3_small")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-train-per-class", type=int, default=None)
    parser.add_argument("--max-validation-per-class", type=int, default=None)
    parser.add_argument("--max-test-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", default="artifacts/run")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 64:
        raise SystemExit("epochs/batch-size must be positive and image-size must be at least 64")
    if any(
        limit is not None and limit < 1
        for limit in (
            args.max_train_per_class,
            args.max_validation_per_class,
            args.max_test_per_class,
        )
    ):
        raise SystemExit("per-class sample caps must be positive")
    train(args)


if __name__ == "__main__":
    main()
