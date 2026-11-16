import json
import pytest
from parseq_ocr.waterec_cli import make_recognizer, deployment_metadata


def fixture_config(tmp_path):
    folder = tmp_path / "deployment"
    folder.mkdir()
    metadata = {
        "family": "waterec",
        "task": "plate",
        "variant": "RS",
        "max_tokens": 512,
        "max_len": 25,
        "normalization": "upper_no_whitespace",
    }
    (folder / "metadata.json").write_text(json.dumps(metadata))
    return {
        "output_dir": tmp_path,
        "task": "plate",
        "variant": "RS",
        "tokens": 512,
        "normalization": "upper_no_whitespace",
    }, metadata


@pytest.mark.parametrize(
    "change", [{"tokens": 384}, {"variant": "R"}, {"task": "price"}]
)
def test_stale_engine_rejected_before_loading_gpu_runtime(tmp_path, change):
    config, _ = fixture_config(tmp_path)
    config.update(change)
    with pytest.raises(ValueError, match="Deployment/config mismatch"):
        make_recognizer(config, "tensorrt")


def test_requested_orientation_must_exist_in_deployment(tmp_path):
    config, metadata = fixture_config(tmp_path)
    folder = tmp_path / "deployment_orientation"
    folder.mkdir()
    (folder / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="has no orientation head"):
        deployment_metadata(config, orientation=True)


def test_explicit_checkpoint_is_not_silently_ignored_by_tensorrt(tmp_path):
    config, _ = fixture_config(tmp_path)
    checkpoint = tmp_path / "new_checkpoint.pt"
    checkpoint.write_bytes(b"different training result")
    with pytest.raises(ValueError, match="differs from the deployed engine"):
        make_recognizer(config, "tensorrt", checkpoint)
