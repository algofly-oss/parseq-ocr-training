import pytest
from parseq_ocr.metrics import score_pairs, score_price_values

def test_numeric_equivalence_is_not_a_reported_metric():
    with pytest.raises(ValueError, match="disabled"):
        score_price_values(["10.50"], ["10,50"])
    assert score_pairs(["10.50"], ["10,50"])["correct"] == 0
