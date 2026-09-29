from io import BytesIO
from zipfile import ZipFile

import numpy as np
from PIL import Image

from plantvision.data import (
    PlantArchiveDataset,
    _leaf_id_for_path,
    exclude_overlapping_groups,
    sample_per_class,
    split_group_indices,
)
from plantvision.evaluation import classification_metrics


def test_group_split_keeps_groups_disjoint_and_classes_in_training():
    labels = [0, 0, 1, 1, 0, 1, 0, 1]
    groups = ["a", "b", "c", "d", "e", "f", "g", "h"]
    train, validation = split_group_indices(labels, groups, validation_fraction=0.25, seed=7)
    assert not {groups[i] for i in train} & {groups[i] for i in validation}
    assert {labels[i] for i in train} == {0, 1}
    assert len(train) + len(validation) == len(labels)


def test_group_split_rejects_invalid_fraction():
    try:
        split_group_indices([0, 1], ["a", "b"], validation_fraction=1.0)
    except ValueError as exc:
        assert "strictly between" in str(exc)
    else:
        raise AssertionError("invalid validation fraction should fail")


def test_metrics_include_macro_and_weighted_scores():
    result = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], [0, 1])
    assert result["accuracy"] == 0.75
    assert 0 <= result["macro_f1"] <= 1
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]
    assert "per_class" in result


def test_archive_dataset_decodes_image_and_samples_each_class(tmp_path):
    archive = tmp_path / "images.zip"
    with ZipFile(archive, "w") as zipped:
        for index in range(8):
            buffer = BytesIO()
            Image.new("RGB", (12, 12), color=(index * 20, 0, 0)).save(buffer, format="JPEG")
            zipped.writestr(f"raw/color/class_{index % 2}/image_{index}.jpg", buffer.getvalue())
    rows = [
        (f"raw/color/class_{index % 2}/image_{index}.jpg", index % 2, f"leaf_{index}")
        for index in range(8)
    ]
    dataset = PlantArchiveDataset(rows, archive)
    assert dataset[0]["image"].size == (12, 12)
    assert dataset["label"] == [0, 1, 0, 1, 0, 1, 0, 1]
    capped = sample_per_class(dataset, limit=2, seed=42)
    assert len(capped) == 4
    assert np.bincount(capped["label"]).tolist() == [2, 2]


def test_leaf_id_mapping_matches_published_leaf_map_rules():
    path = "raw/color/Apple___healthy/uuid___Mary_HL 9155.JPG"
    leaf_map = {"mary_hl 9155": ["Apple___healthy:::1.0"]}
    assert _leaf_id_for_path(path, leaf_map) == "Apple___healthy:::1.0"
    assert _leaf_id_for_path(path, {}) == "fallback_mary_hl 9155"


def test_train_rows_overlapping_the_test_leaf_group_are_removed(tmp_path):
    archive = tmp_path / "empty.zip"
    with ZipFile(archive, "w"):
        pass
    train = PlantArchiveDataset(
        [("a.jpg", 0, "leaf_a"), ("b.jpg", 0, "leaf_b"), ("c.jpg", 1, "leaf_c")],
        archive,
    )
    test = PlantArchiveDataset([("test.jpg", 0, "leaf_b")], archive)
    clean, removed = exclude_overlapping_groups(train, test)
    assert removed == 1
    assert clean["leaf_id"] == ["leaf_a", "leaf_c"]
