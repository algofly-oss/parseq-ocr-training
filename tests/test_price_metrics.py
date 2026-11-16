from parseq_ocr.metrics import score_price_values


def test_decimal_position_and_whole_string_are_preserved():
    result = score_price_values(
        ["4.50 DH", "4.50", "4.50", "4.50", "4.50"],
        ["4,50dh", "450", "45.0", "offer 4.50", "4.50 9.99"],
    )
    assert result["numeric_correct"] == 1
    assert result["numeric_accuracy"] == .2


def test_unparseable_reference_is_not_a_match():
    result = score_price_values(["unreadable", "£4.50"], ["unreadable", "£4.50"])
    assert result["numeric_correct"] == 0
    assert result["numeric_reference_parseable"] == 0
