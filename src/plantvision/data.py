"""PlantVillage source-file loading and leakage-aware image partitions.

The current Hugging Face Hub reader auto-converts this repository to image paths as text and no
longer executes its legacy custom dataset script. This module reads the maintainer's published
split lists and leaf map directly, while decoding RGB images lazily from the official data.zip.
"""

from __future__ import annotations

import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DatasetSplits:
    train: Any
    validation: Any
    test: Any
    class_names: list[str]
    excluded_train_test_overlap_examples: int = 0


def split_group_indices(
    labels: list[int] | np.ndarray,
    groups: list[str] | np.ndarray,
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices by group, rejecting invalid inputs and group leakage."""
    labels_arr = np.asarray(labels)
    groups_arr = np.asarray(groups)
    if labels_arr.ndim != 1 or groups_arr.ndim != 1 or len(labels_arr) != len(groups_arr):
        raise ValueError("labels and groups must be one-dimensional arrays of equal length")
    if len(labels_arr) < 2:
        raise ValueError("at least two samples are required")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    if len(np.unique(groups_arr)) < 2:
        raise ValueError("at least two distinct groups are required")

    splitter = GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(labels_arr)), labels_arr, groups_arr))
    overlap = set(groups_arr[train_idx]).intersection(groups_arr[val_idx])
    if overlap:
        raise RuntimeError(f"group leakage detected: {len(overlap)} overlapping groups")
    missing = set(np.unique(labels_arr)) - set(labels_arr[train_idx])
    if missing:
        raise ValueError(
            f"training partition lacks {len(missing)} class(es); change --seed or split"
        )
    return train_idx, val_idx


def _download_asset(root: Path, relative_path: str) -> Path:
    local = root / relative_path
    if local.is_file():
        return local
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install project dependencies first: pip install -e .") from exc
    downloaded = hf_hub_download(
        repo_id="mohanty/PlantVillage",
        filename=relative_path,
        repo_type="dataset",
        local_dir=str(root),
    )
    return Path(downloaded)


def _leaf_id_for_path(file_rel_path: str, leaf_map: dict[str, list[str]]) -> str:
    class_name = file_rel_path.split("/")[2]
    file_name = Path(file_rel_path).name.replace("_final_masked", "")
    if "___" in file_name:
        file_name = file_name.split("___")[-1]
    image_identifier = file_name.split("copy")[0]
    for suffix in (".jpg", ".png"):
        image_identifier = image_identifier.replace(suffix, "").replace(suffix.upper(), "")
    image_identifier = image_identifier.strip()
    suggestions = leaf_map.get(image_identifier.lower().strip(), [])
    if len(suggestions) == 1:
        return suggestions[0]
    for suggestion in suggestions:
        if class_name in suggestion:
            return suggestion
    return f"fallback_{image_identifier.lower().strip()}"


class PlantArchiveDataset(Dataset):
    """A compact row-indexed view over image paths stored in the official ZIP archive."""

    column_names: ClassVar[list[str]] = ["image", "image_path", "label", "leaf_id"]

    def __init__(self, rows: list[tuple[str, int, str]], archive_path: str | Path):
        self.rows = rows
        self.archive_path = str(archive_path)
        self._archive: zipfile.ZipFile | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def _get_archive(self) -> zipfile.ZipFile:
        if self._archive is None:
            self._archive = zipfile.ZipFile(self.archive_path)
        return self._archive

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_archive"] = None
        return state

    def __getitem__(self, index: int | str):
        if isinstance(index, str):
            positions = {"image_path": 0, "label": 1, "leaf_id": 2}
            if index == "image":
                raise TypeError("images are decoded lazily; request an integer row index")
            if index not in positions:
                raise KeyError(index)
            return [row[positions[index]] for row in self.rows]
        image_path, label, leaf_id = self.rows[int(index)]
        from PIL import Image

        with self._get_archive().open(image_path) as source:
            image = Image.open(source).convert("RGB")
        return {"image": image, "image_path": image_path, "label": label, "leaf_id": leaf_id}

    def select(self, indices: list[int] | np.ndarray) -> PlantArchiveDataset:
        return PlantArchiveDataset([self.rows[int(index)] for index in indices], self.archive_path)


def _read_split(
    path: Path, class_to_id: dict[str, int], leaf_map: dict[str, list[str]]
) -> list[tuple[str, int, str]]:
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        image_path = raw.strip()
        if not image_path:
            continue
        parts = image_path.split("/")
        if len(parts) < 4 or parts[1] != "color":
            continue
        class_name = parts[2]
        if class_name not in class_to_id:
            raise ValueError(f"split file contains an unexpected class: {class_name}")
        rows.append((image_path, class_to_id[class_name], _leaf_id_for_path(image_path, leaf_map)))
    if not rows:
        raise ValueError(f"no color images found in split file {path}")
    return rows


def exclude_overlapping_groups(
    dataset: PlantArchiveDataset, comparison: PlantArchiveDataset
) -> tuple[PlantArchiveDataset, int]:
    """Drop rows whose group occurs in another partition and return the row count removed."""
    comparison_groups = set(comparison["leaf_id"])
    selected = [
        index for index, group in enumerate(dataset["leaf_id"]) if group not in comparison_groups
    ]
    return dataset.select(selected), len(dataset) - len(selected)


def load_plantvillage(
    validation_fraction: float = 0.1, seed: int = 42, data_dir: str | Path | None = None
) -> DatasetSplits:
    """Load official color train/test lists and make a leaf-grouped validation holdout."""
    root = Path(data_dir or os.getenv("PLANTVISION_DATA_DIR", "data/plantvillage"))
    archive = _download_asset(root, "data.zip")
    train_list = _download_asset(root, "splits/color_train.txt")
    test_list = _download_asset(root, "splits/color_test.txt")
    leaf_map_path = _download_asset(root, "leaf_grouping/leaf-map.json")
    with leaf_map_path.open(encoding="utf-8") as file:
        leaf_map = json.load(file)
    train_paths = [
        row.strip() for row in train_list.read_text(encoding="utf-8").splitlines() if row.strip()
    ]
    test_paths = [
        row.strip() for row in test_list.read_text(encoding="utf-8").splitlines() if row.strip()
    ]
    class_names = sorted({path.split("/")[2] for path in train_paths + test_paths})
    if len(class_names) != 38:
        raise RuntimeError(f"expected 38 PlantVillage classes; found {len(class_names)}")
    class_to_id = {name: index for index, name in enumerate(class_names)}
    train_all = PlantArchiveDataset(_read_split(train_list, class_to_id, leaf_map), archive)
    test = PlantArchiveDataset(_read_split(test_list, class_to_id, leaf_map), archive)
    train_all, excluded_overlap_count = exclude_overlapping_groups(train_all, test)
    labels = np.asarray(train_all["label"], dtype=np.int64)
    groups = np.asarray(train_all["leaf_id"], dtype=str)
    train_idx, validation_idx = split_group_indices(labels, groups, validation_fraction, seed)
    return DatasetSplits(
        train=train_all.select(train_idx),
        validation=train_all.select(validation_idx),
        test=test,
        class_names=class_names,
        excluded_train_test_overlap_examples=excluded_overlap_count,
    )


def sample_per_class(
    dataset: PlantArchiveDataset, limit: int | None, seed: int
) -> PlantArchiveDataset:
    """Deterministically cap examples per label after leakage-safe split assignment."""
    if limit is None:
        return dataset
    if limit < 1:
        raise ValueError("per-class sample limits must be positive")
    labels = np.asarray(dataset["label"], dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected = []
    for label in np.unique(labels):
        candidates = np.flatnonzero(labels == label)
        count = min(limit, len(candidates))
        selected.extend(rng.choice(candidates, size=count, replace=False).tolist())
    selected.sort()
    return dataset.select(selected)


class LeafImageDataset(Dataset):
    """Adapt a row-indexed image dataset to PyTorch transforms."""

    def __init__(self, dataset: Any, transform: Any | None = None) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        row = self.dataset[int(index)]
        image = row["image"]
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"])
