from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from parseq_ocr.ppocr_models import decode, preprocess, restore_bn, VOCAB
from parseq_ocr.ppocr_data import scan_labeled, split_price_holdout


def test_ctc_repeat_collapse_happens_before_blank_removal():
    a = VOCAB.index("1")
    b = VOCAB.index(".")
    assert decode([[a, a, 0, a, b, b, 0, b]]) == ["11.."]


def test_price_group_split_rounds_preserves_spaces_and_excludes_blanks(tmp_path):
    for i, text in enumerate([" 10 DH ", "10 DH", "10.50", "  ", "$2.00"]):
        Image.new("RGB", (3, 2)).save(tmp_path / f"{i}.png")
        (tmp_path / f"{i}.txt").write_text(text)
    rows = scan_labeled(tmp_path)
    assert len(rows) == 4 and rows[0]["label"] == "10 DH"
    fit, held = split_price_holdout(rows, 0.5, 20261114)
    assert len({r["label"] for r in held}) == 2  # round(3*.5), not floor
    assert not {r["label"] for r in fit} & {r["label"] for r in held}


def test_bn_reconstruction_keeps_initial_math():
    import torch

    torch.manual_seed(7)
    m = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1))
    x = torch.randn(2, 3, 9, 13)
    with torch.no_grad():
        expected = m(x)
    restore_bn(m, [x])
    m.eval()
    with torch.no_grad():
        actual = m(x)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    m.train()
    m(x).square().mean().backward()
    assert m[0][0].weight.grad.isfinite().all()


def test_bgr_linear_resize_keeps_color_channels():
    pytest.importorskip("cv2")
    image = Image.new("RGB", (5, 3), (255, 0, 0))
    x = preprocess(image)
    assert x.shape == (3, 128, 320)
    np.testing.assert_array_equal(x[:, 0, 0], [-1, -1, 1])
