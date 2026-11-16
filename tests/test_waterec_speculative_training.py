"""Regression coverage for frozen-copy gradients and safe completed-run reuse."""
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from parseq_ocr import waterec_speculative_training as training
from parseq_ocr.waterec_models import NativeWATERec
from parseq_ocr.waterec_speculative_export import DraftHead


def test_copied_frozen_decoder_is_independently_trainable():
    torch.set_num_threads(2)
    target = NativeWATERec({"task": "plate", "variant": "RS", "tokens": 512}).eval()
    target.requires_grad_(False)
    original = (
        target.decoder.decoder.layers[0].multihead_attn.out_proj.weight.detach().clone()
    )
    head = DraftHead(target.decoder)
    assert all(p.requires_grad for p in head.parameters())
    keys, values = torch.randn(1, 12, 4, 32), torch.randn(1, 12, 4, 32)
    logits = head(keys, values, torch.zeros(1, 4, dtype=torch.bool))
    logits.square().mean().backward()
    for name, parameter in head.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.norm() > 0, name
    torch.optim.SGD(head.parameters(), lr=0.01).step()
    assert not torch.equal(head.output.weight, original)
    assert torch.equal(
        target.decoder.decoder.layers[0].multihead_attn.out_proj.weight, original
    )
    assert all(p.grad is None for p in target.parameters())


def test_distillation_ignores_tokens_after_first_eos():
    probabilities = torch.zeros(1, 25, 95)
    probabilities[:, :, 7] = 1
    probabilities[:, 1] = 0
    probabilities[:, 1, 0] = 1
    hard, valid = training._targets(probabilities.numpy())
    assert valid.tolist() == [[True, True] + [False] * 23]
    logits = torch.randn(1, 25, 95, requires_grad=True)
    first = training.distillation_loss(logits, hard, probabilities, valid)
    changed = logits.detach().clone()
    changed[:, 2:] *= 100
    second = training.distillation_loss(changed, hard, probabilities, valid)
    assert torch.equal(first.detach(), second)
    first.backward()
    assert torch.count_nonzero(logits.grad[:, 2:]) == 0


@pytest.mark.parametrize("mutation", ["label", "pixels"])
def test_inplace_input_changes_rejected_before_fit(tmp_path, monkeypatch, mutation):
    dataset = tmp_path / "dataset" / "train"
    dataset.mkdir(parents=True)
    image, text = dataset / "sample.png", dataset / "sample.txt"
    Image.new("RGB", (16, 8), "white").save(image)
    text.write_text("AB123")
    target = tmp_path / "target.pt"
    target.write_bytes(b"target identity; heavy model loading is deliberately mocked")
    calls = []

    def fit(*args):
        calls.append(True)
        return args[5] / "draft.pt", {"selected_epoch": 1}

    monkeypatch.setattr(training, "_fit_phase", fit)
    config = {
        "dataset_root": str(dataset.parent),
        "output_dir": str(tmp_path / "run"),
        "task": "plate",
        "tokens": 512,
        "max_label_length": 25,
        "synthetic_plate_count": 0,
        "device": "cpu",
        "speculative": {"fixed_epoch": 1},
    }
    training.train_speculative(config, target)
    identity = tmp_path / "run" / "speculative" / "input_identity.json"
    before = identity.read_bytes()
    if mutation == "label":
        text.write_text("CD456")
    else:
        Image.new("RGB", (16, 8), "black").save(image)
    with pytest.raises(ValueError, match="training inputs changed"):
        training.train_speculative(config, target)
    assert len(calls) == 1
    assert identity.read_bytes() == before


def test_staged_selection_cannot_reuse_full_teacher(tmp_path, monkeypatch):
    target = tmp_path / "target.pt"
    target.write_bytes(b"same selected and full target")
    monkeypatch.setattr(training, "_training_rows", lambda *_: ([], [], []))
    config = {
        "output_dir": str(tmp_path / "run"),
        "task": "plate",
        "device": "cpu",
        "refit_all_training": True,
        "speculative": {"selection_target_checkpoint": str(target)},
    }
    with pytest.raises(ValueError, match="teacher hashes must differ"):
        training.train_speculative(config, target)
    assert not (tmp_path / "run" / "speculative").exists()


def test_public_teacher_provenance_must_match_fold(tmp_path):
    folder = tmp_path / "selection"
    folder.mkdir()
    target = folder / "best.pt"
    fingerprint = {"sha256": "expected"}
    (tmp_path / "input_fingerprint.json").write_text(
        json.dumps({"sha256": "different"})
    )
    (tmp_path / "split.json").write_text(json.dumps({"train": [], "holdout": []}))
    with pytest.raises(ValueError, match="content/folds differ"):
        training._selection_provenance(target, fingerprint, [], [])


def test_legacy_teacher_provenance_is_not_certified(tmp_path):
    result = training._selection_provenance(tmp_path / "selected.pt", {}, [], [])
    assert "Unverified" in result["status"]


def test_teacher_cache_disables_outer_cpu_autocast():
    x = torch.randn(3, 3)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert (x @ x).dtype == torch.bfloat16
        with training._teacher_math(torch.device("cpu")):
            assert (x @ x).dtype == torch.float32
        assert (x @ x).dtype == torch.bfloat16
