"""CPU/CLI regressions for the opt-in path; no CUDA execution in these tests."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import types

import numpy as np
from PIL import Image
import pytest
from parseq_ocr import waterec_cli


def test_gpu_preprocessing_rejects_torch_before_loading_model():
    with pytest.raises(ValueError, match="requires --backend tensorrt"):
        waterec_cli.make_recognizer({"gpu_preprocess": True}, "torch")


def test_gpu_option_rejects_training_command(monkeypatch, capsys):
    monkeypatch.setattr(waterec_cli, "load_config", lambda _: {})
    monkeypatch.setattr(sys, "argv", ["waterec-ocr", "train", "--gpu-preprocess"])
    with pytest.raises(SystemExit) as error:
        waterec_cli.main()
    assert error.value.code == 2
    assert "applies only to infer, evaluate and benchmark" in capsys.readouterr().err


def test_native_assets_are_present_and_declared_for_packaging():
    import parseq_ocr

    package = Path(parseq_ocr.__file__).parent
    assets = package / "_native" / "waterec"
    for name in ["coefficients_native.c", "kernels.cu", "PILLOW_LICENSE"]:
        assert (assets / name).is_file()
    assert "MIT-CMU" in (assets / "PILLOW_LICENSE").read_text()
    config = (package.parent / "pyproject.toml").read_text()
    for entry in [
        "_native/waterec/*.c",
        "_native/waterec/*.cu",
        "_native/waterec/PILLOW_LICENSE",
    ]:
        assert entry in config
    assert not list(assets.glob("*.so")) and not list(assets.glob("*.cubin"))


def gpu_dependencies():
    pytest.importorskip("tensorrt")
    pytest.importorskip("pycuda.driver")
    pytest.importorskip("cuda.bindings.runtime")


def test_gpu_frontend_import_never_loads_torch():
    gpu_dependencies()
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import parseq_ocr.waterec_gpu_preprocessing; assert 'torch' not in sys.modules",
        ],
        check=True,
    )


@pytest.mark.parametrize("insize,outsize", [(37, 13), (5, 19)])
def test_cpu_coefficients_reproduce_pillow_rounding(
    tmp_path, monkeypatch, insize, outsize
):
    gpu_dependencies()
    if not shutil.which("cc"):
        pytest.skip("A C compiler is required for the optional coefficient helper")
    from parseq_ocr.waterec_gpu_preprocessing import coefficients, _coefficient_library

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv("CC", "cc")
    coefficients.cache_clear()
    _coefficient_library.cache_clear()
    rng = np.random.default_rng(19)
    pixels = rng.integers(0, 256, (1, insize, 3), dtype=np.uint8)
    bounds, weights = coefficients(insize, outsize)
    result = np.empty((1, outsize, 3), np.uint8)
    for i, (start, count) in enumerate(bounds):
        accum = (1 << 21) + (
            pixels[0, start : start + count].astype(np.int64) * weights[i, :count, None]
        ).sum(0)
        result[0, i] = np.clip(accum >> 22, 0, 255)
    expected = np.asarray(
        Image.fromarray(pixels).resize((outsize, 1), Image.Resampling.BICUBIC)
    )
    np.testing.assert_array_equal(result, expected)
    assert not bounds.flags.writeable and not weights.flags.writeable
    assert list((tmp_path / "parseq-ocr" / "resize").glob("*.so"))
    coefficients.cache_clear()
    _coefficient_library.cache_clear()


@pytest.mark.parametrize("gpu", [False, True])
def test_benchmark_records_preprocessing_identity_without_gpu_execution(
    tmp_path, monkeypatch, gpu
):
    if gpu:
        gpu_dependencies()
    from parseq_ocr.waterec_benchmark import benchmark

    folder = tmp_path / "images"
    folder.mkdir()
    Image.new("RGB", (8, 4)).save(folder / "one.png")
    (folder / "one.txt").write_text("AB12")
    deploy = tmp_path / "deployment"
    deploy.mkdir()
    metadata = {
        "family": "waterec",
        "task": "plate",
        "variant": "RS",
        "max_tokens": 512,
        "max_len": 25,
        "normalization": "none",
        "checkpoint_sha256": "selected-model-identity",
    }
    (deploy / "metadata.json").write_text(json.dumps(metadata))
    for filename in ["encoder_cross_kv.engine", "decoder_cached_step.engine"]:
        (deploy / filename).write_bytes(filename.encode())

    class FakeRecognizer:
        def infer(self, images):
            return ["AB12"] * len(images)

        def close(self):
            pass

    monkeypatch.setattr(waterec_cli, "make_recognizer", lambda *args: FakeRecognizer())
    nv = types.SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda _: 0,
        nvmlDeviceGetComputeRunningProcesses=lambda _: [],
    )
    monkeypatch.setitem(sys.modules, "pynvml", nv)
    cfg = {
        "task": "plate",
        "variant": "RS",
        "tokens": 512,
        "normalization": "none",
        "output_dir": tmp_path,
        "gpu_preprocess": gpu,
    }
    result = benchmark(cfg, "tensorrt", folder, repeats=1, predecoded=True)
    assert result["pipeline"]["preprocessing"] == (
        "gpu_exact_bicubic" if gpu else "cpu_pillow_bicubic"
    )
    assert result["pipeline"]["checkpoint_sha256"] == "selected-model-identity"
    if gpu:
        assert set(result["pipeline"]["preprocessing_source_sha256"]) >= {
            "python",
            "coefficients_native.c",
            "kernels.cu",
        }
        assert all(
            len(v) == 64
            for v in result["pipeline"]["preprocessing_source_sha256"].values()
        )
    assert "Predecoded" in result["scope"] and result["warmup"] == 5
