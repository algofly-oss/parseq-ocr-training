"""Deployment identity guards must fail before any GPU allocation."""
import hashlib
import json
from pathlib import Path
import pytest


def deployment(tmp_path):
    base = tmp_path / "base"
    head = tmp_path / "speculative"
    base.mkdir()
    head.mkdir()
    characters = ["</s>"] + [str(i) for i in range(94)] + ["<s>", "<pad>"]
    target = {
        "task": "plate",
        "max_tokens": 512,
        "checkpoint_sha256": "a" * 64,
        "characters": characters,
        "bos": 95,
        "eos": 0,
    }
    metadata = {
        "family": "waterec_speculative",
        "max_tokens": 512,
        "max_len": 25,
        "vocab_size": 95,
        "bos": 95,
        "eos": 0,
        "target_characters": characters,
        "target_checkpoint_sha256": "a" * 64,
        "experimental": True,
    }
    (base / "metadata.json").write_text(json.dumps(target))
    (head / "metadata.json").write_text(json.dumps(metadata))
    metadata_digest = hashlib.sha256((head / "metadata.json").read_bytes()).hexdigest()
    for name in ["draft", "verifier"]:
        payload = name.encode()
        (head / f"{name}.engine").write_bytes(payload)
        (head / f"{name}.build.json").write_text(
            json.dumps(
                {
                    "identity": {"metadata_sha256": metadata_digest},
                    "engine_sha256": hashlib.sha256(payload).hexdigest(),
                    "attention_output_barriers": ["attention"]
                    if name == "draft"
                    else [],
                }
            )
        )
    return base, head, metadata


@pytest.mark.parametrize(
    "change,message",
    [
        ({"eos": 1}, "EOS0"),
        ({"bos": 94}, "BOS95"),
        ({"target_checkpoint_sha256": "b" * 64}, "checkpoint SHA256"),
        ({"experimental": False}, "metadata digest"),
    ],
)
def test_identity_rejected_before_gpu(tmp_path, monkeypatch, change, message):
    from parseq_ocr import waterec_speculative as module

    base, head, metadata = deployment(tmp_path)
    metadata.update(change)
    (head / "metadata.json").write_text(json.dumps(metadata))

    def unexpected_gpu(*args, **kwargs):
        raise AssertionError("GPU initialization preceded metadata validation")

    monkeypatch.setattr(module, "TensorRTRecognizer", unexpected_gpu)
    monkeypatch.setattr(module, "GpuInputRecognizer", unexpected_gpu)
    with pytest.raises(ValueError, match=message):
        module.SpeculativeRecognizer(base, head)


def test_changed_engine_rejected_before_gpu(tmp_path):
    from parseq_ocr.waterec_speculative import SpeculativeRecognizer

    base, head, _ = deployment(tmp_path)
    (head / "draft.engine").write_bytes(b"different engine")
    with pytest.raises(ValueError, match="engine/build digest"):
        SpeculativeRecognizer(base, head)


def test_torch_backend_cannot_silently_ignore_speculation():
    from parseq_ocr.waterec_cli import make_recognizer

    with pytest.raises(ValueError, match="requires --backend tensorrt"):
        make_recognizer({"speculative": True}, "torch")
