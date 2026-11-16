import json
import numpy as np
import pytest
from PIL import Image
from parseq_ocr.waterec_orientation import (
    spatial_weights,
    create_head,
    _rotate,
    export_orientation,
)


def test_valid_spatial_weights_and_empty_quadrants():
    w = spatial_weights(1, 1)
    assert w.shape == (1, 5, 512)
    np.testing.assert_array_equal(w.sum(-1), [[1, 1, 0, 0, 0]])
    w = spatial_weights(3, 5)
    np.testing.assert_allclose(w.sum(-1), np.ones((1, 5)), atol=1e-7)
    assert not w[:, :, 15:].any()
    grid = np.arange(15, dtype=np.float32)
    np.testing.assert_allclose((w[0, 0, :15] * grid).sum(), grid.mean())
    np.testing.assert_allclose(
        (w[0, 4, :15] * grid).sum(),
        np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 13, 14]]).sum() / 2,
    )


def test_rotation_class_inverse_and_head_shape():
    import torch

    im = Image.new("RGB", (3, 2))
    im.putpixel((0, 0), (255, 0, 0))
    assert _rotate(im, 1).getpixel((1, 0)) == (255, 0, 0)
    assert np.array_equal(np.asarray(_rotate(_rotate(im, 1), 3)), np.asarray(im))
    head = create_head()
    assert sum(p.numel() for p in head.parameters()) == 250244
    head(torch.randn(2, 1920)).sum().backward()
    assert head[1].weight.grad.isfinite().all()


def test_export_rejects_wrong_encoder_before_writing(tmp_path):
    import torch

    pytest.importorskip("onnx")
    deploy = tmp_path / "base"
    deploy.mkdir()
    (deploy / "metadata.json").write_text(json.dumps({"checkpoint_sha256": "correct"}))
    torch.save(
        {
            "encoder_checkpoint_sha256": "wrong",
            "state_dict": create_head().state_dict(),
        },
        tmp_path / "head.pt",
    )
    with pytest.raises(ValueError, match="SHA differ"):
        export_orientation(deploy, tmp_path / "head.pt", tmp_path / "optional")
    assert not (tmp_path / "optional").exists()
