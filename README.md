# Plant Disease Vision

A reproducible, explainable computer-vision pipeline for classifying **38 PlantVillage leaf health/disease categories**. It compares two ImageNet transfer-learning backbones, selects checkpoints with validation macro-F1, reports held-out class-wise metrics, and produces Grad-CAM explanations in a Streamlit demo.

> **Research prototype, not an agronomic diagnostic.** PlantVillage images were captured under controlled conditions; field performance may be substantially lower. Confidence scores are not calibrated probabilities or treatment advice.

## Engineering scope

- **Leakage-aware partitions:** reads the maintainer's official train/test lists and leaf map. It removes training examples whose leaf IDs also occur in the held-out test set, then creates a leaf-group-disjoint validation partition from the remaining training examples. The verified split has zero leaf-ID intersections across all three partitions.
- **Model comparison:** MobileNetV3-Small and EfficientNet-B0 transfer learning with identical image transforms, sampler policy, seeds, epoch budget, and evaluation protocol.
- **Evaluation:** accuracy, balanced accuracy, macro/weighted precision-recall-F1, per-class CSV, confusion matrix, and training curves.
- **Explainability and inference:** Grad-CAM overlay, reusable predictor/CLI, and Streamlit upload/camera-photo demo.

## Data source and protocol

The source is [mohanty/PlantVillage on Hugging Face](https://huggingface.co/datasets/mohanty/PlantVillage), based on the original PlantVillage dataset. The card describes 54,306 images, 14 crop species, 26 diseases, 38 labels, and CC BY-SA 3.0. The official color split files contain 43,596 train paths and 10,709 test paths. The data archive is approximately 2.18 GB.

The Hub's current automatic conversion presents image paths as text and does not run its legacy Python dataset script. To avoid pretending those path strings are decoded images, this project downloads the published `data.zip`, color split lists, and `leaf-map.json` directly; image bytes are decoded lazily from the archive during training. Our actual parse found 290 train rows whose mapped leaf groups also occur in the test split, so those rows are excluded before making train/validation partitions. With seed 42, the uncapped partitions contain **39,011 train, 4,295 validation, and 10,709 test images**, with all 38 classes in each split and no shared leaf IDs. Run metadata records these counts and overlap exclusions.

The image archive is not included in this repository. Review the dataset card's terms and cite the original paper:

> Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). “Using deep learning for image-based plant disease detection.” *Frontiers in Plant Science*, 7, 1419. https://doi.org/10.3389/fpls.2016.01419

```bibtex
@article{Mohanty2016PlantDisease,
  title={Using deep learning for image-based plant disease detection},
  author={Mohanty, Sharada P. and Hughes, David P. and Salath{\'e}, Marcel},
  journal={Frontiers in Plant Science},
  volume={7},
  pages={1419},
  year={2016},
  doi={10.3389/fpls.2016.01419}
}
```

## Setup and training

Python 3.10+ is required. CUDA is optional; the code works on CPU. Install dependencies and development tools:

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

To run the full dataset, leave per-class caps unset. For a faster CPU comparison, the documented settings below give each model the same balanced, deterministic sample **after** leakage-safe split assignment:

```bash
plantvision-train --model mobilenet_v3_small --epochs 5 --patience 2 \\
  --image-size 160 --batch-size 32 --workers 2 \\
  --max-train-per-class 80 --max-validation-per-class 20 --max-test-per-class 50 \\
  --output-dir artifacts/mobilenet

plantvision-train --model efficientnet_b0 --epochs 5 --patience 2 \\
  --image-size 160 --batch-size 32 --workers 2 \\
  --max-train-per-class 80 --max-validation-per-class 20 --max-test-per-class 50 \\
  --output-dir artifacts/efficientnet
```

The first run downloads the official data archive/split metadata and ImageNet weights. Keep all experiment choices fixed between backbones. Checkpoint selection uses **validation** macro-F1; the test set is evaluated after selection. Optional `--max-*-per-class` arguments are deterministic caps, not a replacement for the full-data workflow; actual counts and choices are recorded in each run.

Each model run writes `model_bundle.pt`, `run_config.json`, `metrics.json`, `per_class_metrics.csv`, `confusion_matrix.png`, and `training_curves.png` under its output directory.

### Verified CPU comparison

Both models were trained with the documented settings and selected by validation macro-F1. MobileNetV3-Small reached validation macro-F1 **0.9292** and held-out sampled-test macro-F1 **0.9294** (accuracy **0.9293**). EfficientNet-B0 reached its best validation macro-F1 **0.9764** at epoch 4 and sampled-test macro-F1 **0.9602** (accuracy **0.9601**). The capped test subset contains **1,882 images** across all 38 classes (up to 50 per class; the smallest available class contributes 32). The live app initially selects MobileNetV3-Small and allows either backbone; EfficientNet-B0 scored higher on this capped split. These figures describe a seeded, balanced sample of the official test split—not a full-test benchmark—and should not be presented as field performance. The saved JSON and CSV contain the complete per-class results.

## Inference demo

```bash
plantvision-predict --model-path artifacts/efficientnet/model_bundle.pt --image path/to/leaf.jpg
streamlit run src/plantvision/app.py
```

The app lets you choose a trained backbone, upload an image or take a browser camera photo, and view top predictions plus a Grad-CAM overlay. It is an image-level research demo, not continuous video diagnosis.

**Temporary demo status:** The task-session demo stopped when its Sandbox was reset, so there is no active public demo link at present. Run the app locally with the command above, or deploy it with a trained model bundle for a persistent link.

## Tests and project layout

```bash
pytest
ruff check src tests
```

CI tests data parsing helpers, archive-backed image decoding, sampling, leaf-group leakage protection, metrics, and imports; it does not download the full image corpus or pretrained weights.

```text
src/plantvision/
  data.py         direct source-archive loader and leaf-group-safe splits
  models.py       transfer-learning model factory
  train.py        training, validation checkpointing, held-out evaluation
  evaluation.py   metrics and plots
  explain.py      Grad-CAM overlays
  inference.py    reusable predictor and CLI
  app.py          Streamlit inference demo
```

## Resume bullet

> Built a leakage-aware 38-class plant-disease vision pipeline; compared MobileNetV3-Small and EfficientNet-B0 transfer learning, selected EfficientNet-B0 on validation macro-F1, and achieved **0.960 macro-F1** on a 1,882-image leaf-disjoint sample of the official test split. Shipped an upload/camera-photo Streamlit demo with Grad-CAM explanations.
