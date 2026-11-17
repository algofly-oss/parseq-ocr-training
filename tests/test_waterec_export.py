"""CPU checks for deployment geometry and exact cached decoder semantics."""
import pytest
import torch
from parseq_ocr.waterec_export import (
    CachedDecoderStep, _validate_checkpoint_config, project_cross_kv, rotary_real,
)
from parseq_ocr._vendor.waterec.encoder import apply_rotary_emb, compute_axial_cis
from parseq_ocr._vendor.waterec.decoder import NaARDecoder


def test_checkpoint_deployment_contract_rejects_changed_geometry_or_task():
    source = dict(tokens=512, max_label_length=25, task="plate", variant="RS", normalization="upper_no_whitespace")
    assert _validate_checkpoint_config(source, source) == "upper_no_whitespace"
    assert _validate_checkpoint_config({**source, "normalization": "none"}, source) == "upper_no_whitespace"
    missing = {k:v for k,v in source.items() if k != "normalization"}
    assert _validate_checkpoint_config(missing, missing) == "none"
    for key, value in [("tokens", 384), ("max_label_length", 24), ("task", "price"), ("variant", "R")]:
        changed = {**source, key: value}
        with pytest.raises(ValueError):
            _validate_checkpoint_config(changed, source)
    for key, value in [("tokens", 384), ("max_label_length", 24)]:
        with pytest.raises(ValueError):
            _validate_checkpoint_config(source, {**source, key: value})


def test_real_rope_preserves_original_complex_pairs():
    torch.manual_seed(17)
    query = torch.randn(2, 6, 14, 64)
    key = torch.randn_like(query)
    frequencies = compute_axial_cis(64, 7, 2).unsqueeze(0).expand(2, -1, -1)
    expected_q, expected_k = apply_rotary_emb(query, key, frequencies)
    assert torch.allclose(rotary_real(query, frequencies.real, frequencies.imag), expected_q, atol=1e-6, rtol=1e-6)
    assert torch.allclose(rotary_real(key, frequencies.real, frequencies.imag), expected_k, atol=1e-6, rtol=1e-6)


def test_cached_decoder_matches_native_full_prefix_at_all_positions():
    torch.set_num_threads(2)
    torch.manual_seed(19)
    decoder = NaARDecoder(384, 99, nhead=12, num_decoder_layers=2, max_len=25).eval()
    step = CachedDecoderStep(decoder).eval()
    memory = torch.randn(2, 13, 384)
    padding = torch.zeros(2, 13, dtype=torch.bool)
    padding[1, 8:] = True
    tokens = torch.randint(1, 97, (2, 25))
    tokens[:, 0] = decoder.bos
    past_k = torch.zeros(2, 2, 12, 25, 32)
    past_v = torch.zeros_like(past_k)
    with torch.inference_mode():
        cross_k, cross_v = project_cross_kv(memory, decoder.decoder.layers)
        for position in range(25):
            prefix = tokens[:, :position+1]
            embedded = decoder.positional_encoding(decoder.embedding(prefix))
            causal = torch.triu(torch.full((position+1, position+1), -torch.inf), 1)
            expected = decoder.tgt_word_prj(decoder.decoder(embedded, memory, tgt_mask=causal, memory_key_padding_mask=padding)[:, -1:])
            actual, past_k, past_v = step(tokens[:, position:position+1], torch.tensor([position]), cross_k, cross_v, padding, past_k, past_v)
            assert torch.allclose(actual, expected, atol=3e-5, rtol=3e-5)
