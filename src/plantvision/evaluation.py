"""Evaluation metrics and visual diagnostics."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[int]
) -> dict:
    labels = list(labels)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "per_class": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def save_evaluation(metrics: dict, class_names: list[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_class = metrics["per_class"]
    with (output_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["class", "precision", "recall", "f1-score", "support"]
        )
        writer.writeheader()
        for index, class_name in enumerate(class_names):
            row = per_class.get(str(index), {})
            writer.writerow(
                {
                    "class": class_name,
                    "precision": row.get("precision", 0.0),
                    "recall": row.get("recall", 0.0),
                    "f1-score": row.get("f1-score", 0.0),
                    "support": row.get("support", 0),
                }
            )

    matrix = np.asarray(metrics["confusion_matrix"])
    fig_size = max(12, min(28, len(class_names) * 0.48))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        title="PlantVillage test confusion matrix", xlabel="Predicted class", ylabel="True class"
    )
    ax.set_xticks(range(len(class_names)), class_names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(class_names)), class_names, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def save_training_curves(history: list[dict], output_path: Path) -> None:
    if not history:
        return
    epochs = [entry["epoch"] for entry in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [entry["train_loss"] for entry in history], label="train")
    axes[0].plot(epochs, [entry["val_loss"] for entry in history], label="validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend()
    axes[1].plot(epochs, [entry["val_macro_f1"] for entry in history], label="validation macro-F1")
    axes[1].set(title="Selection metric", xlabel="Epoch", ylabel="Macro-F1", ylim=(0, 1))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
