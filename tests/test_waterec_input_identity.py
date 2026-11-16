"""Actual train orchestration rejects changed inputs before reusing old runs."""
import json
from pathlib import Path
from PIL import Image
import pytest
from parseq_ocr import waterec_training as training


@pytest.fixture
def completed_run(tmp_path, monkeypatch):
    source = tmp_path / "dataset" / "train"
    source.mkdir(parents=True)
    names = []
    for i, label in enumerate(["AA", "BB", "CC", "DD"]):
        path = source / f"{i}.png"
        Image.new("RGB", (8, 4), (20 + i * 40, 10, 20)).save(path)
        path.with_suffix(".txt").write_text(label)
        names.append(path.name)
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": names[:2], "holdout": names[2:]}))
    cfg = {
        "dataset_root": tmp_path / "dataset",
        "output_dir": tmp_path / "run",
        "split_manifest": split,
        "task": "plate",
        "normalization": "upper_no_whitespace",
        "synthetic_plate_count": 0,
        "epochs": 1,
        "device": "cpu",
        "precision": "fp32",
        "refit_all_training": True,
    }
    calls = []

    def fake_fit(records, held, config, folder, epochs, resume):
        calls.append((len(records), len(held)))
        folder.mkdir(parents=True, exist_ok=True)
        result = {"selected_epoch": 1}
        (folder / "summary.json").write_text(json.dumps(result))
        (folder / "model.pt").write_bytes(b"not-a-real-model")
        return result

    monkeypatch.setattr(training, "_fit", fake_fit)
    training.train(cfg)
    assert len(calls) == 2
    calls.clear()
    return cfg, source, split, calls


@pytest.mark.parametrize("change", ["label", "pixels", "split"])
def test_changed_inplace_inputs_rejected_without_overwriting_split(
    completed_run, change
):
    cfg, source, split, calls = completed_run
    output = cfg["output_dir"]
    saved_split = (output / "split.json").read_bytes()
    saved_fingerprint = (output / "input_fingerprint.json").read_bytes()
    if change == "label":
        (source / "0.txt").write_text("ZZ")
    elif change == "pixels":
        Image.new("RGB", (8, 4), (1, 2, 3)).save(source / "0.png")
    else:
        split.write_text(
            json.dumps({"train": ["0.png", "2.png"], "holdout": ["1.png", "3.png"]})
        )
    with pytest.raises(ValueError, match="changed in-place.*new output_dir"):
        training.train(cfg)
    assert not calls
    assert (output / "split.json").read_bytes() == saved_split
    assert (output / "input_fingerprint.json").read_bytes() == saved_fingerprint


def test_identical_inputs_reuse_completed_run(completed_run):
    cfg, _, _, calls = completed_run
    assert training.train(cfg)["selected_epoch"] == 1
    assert not calls


def test_legacy_completed_run_cannot_be_blessed(completed_run):
    cfg, _, _, calls = completed_run
    output = cfg["output_dir"]
    (output / "input_fingerprint.json").unlink()
    old = (output / "split.json").read_bytes()
    with pytest.raises(ValueError, match="lacks an input fingerprint.*new output_dir"):
        training.train(cfg)
    assert not calls and not (output / "input_fingerprint.json").exists()
    assert (output / "split.json").read_bytes() == old
