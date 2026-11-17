import pytest
from PIL import Image
from parseq_ocr.data import scan_pairs, split_holdout, prepare_from_labeled, audit_splits
from parseq_ocr.metrics import score_pairs


def test_grouped_split_and_unchanged_pixels(tmp_path):
    source = tmp_path/"source"
    source.mkdir()
    for i, label in enumerate(["AA11", "AA11", "BB22", "CC33"]):
        Image.new("RGB", (48,24), (i*40,20,100)).save(source/f"{i}.png")
        (source/f"{i}.txt").write_text(label)
    records = scan_pairs(source)
    training, held = split_holdout(records, .5, 42)
    assert not {r["label"] for r in training} & {r["label"] for r in held}
    destination = tmp_path/"dataset"
    result = prepare_from_labeled(source, destination, .5)
    assert result["train_images"]+result["val_images"] == 4
    assert not result["pixel_duplicates_across_splits"]
    with pytest.raises(FileExistsError):
        prepare_from_labeled(source, destination)


def test_missing_label_is_not_silently_skipped(tmp_path):
    Image.new("RGB", (10,10)).save(tmp_path/"missing.png")
    with pytest.raises(ValueError, match="Missing label"):
        scan_pairs(tmp_path)


def test_only_literal_exact_is_reported():
    result = score_pairs(["ab 12", "CD34"], ["AB-12", "CD34"])
    assert result["exact_accuracy"] == .5
    assert set(result) == {"n", "correct", "exact_accuracy", "metric_policy"}


def test_pixel_overlap_is_found_even_with_different_filenames(tmp_path):
    for name in ["a", "b"]:
        Image.new("RGB", (12,12), "white").save(tmp_path/f"{name}.png")
    rows = [{"path": str(tmp_path/f"{name}.png"), "image": name, "label": "AB12"} for name in ["a", "b"]]
    assert audit_splits(rows[:1], rows[1:])["pixel_duplicates_across_splits"]
