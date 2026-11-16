import pytest
import torch
from parseq_ocr.config import load_config
from parseq_ocr.models import create_system
from parseq_ocr.parallel import parallel_loss, stage_logits
from parseq_ocr.parallel_export import ParallelGraph
from parseq_ocr.export import DynamicAttention


def config():
    cfg = load_config('configs/alpr_fast_tiny.yaml')
    cfg.update(device='cpu', precision='fp32')
    return cfg


def test_parallel_training_has_gradients_and_rejects_truncation():
    torch.set_num_threads(2)
    system = create_system(config(), pretrained=False)
    loss = parallel_loss(system, torch.randn(2,3,48,192), ['AB12','CD34'])
    assert torch.isfinite(loss)
    loss.backward()
    assert system.model.encoder.pos_embed.grad is not None
    assert system.model.head.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='never truncated'):
        parallel_loss(system, torch.randn(1,3,48,192), ['A'*13])


def test_parallel_graph_matches_native_with_dynamic_batch_and_eos():
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    system = create_system(config(), pretrained=False).eval()
    for block in system.model.encoder.blocks:
        block.attn.fused_attn = False
    # Exercise EOS masks as well as the no-EOS path.
    with torch.no_grad():
        system.model.head.bias[system.tokenizer.eos_id] = 3
    for batch in [1,3]:
        x = torch.rand(batch,3,48,192)*255
        with torch.inference_mode():
            expected = system(x/127.5-1)
            stages = stage_logits(system, x/127.5-1)
        torch.testing.assert_close(stages[-1], expected, atol=.002, rtol=.001)
        if batch == 1:
            for layer in system.model.decoder.layers:
                layer.self_attn = DynamicAttention(layer.self_attn)
                layer.cross_attn = DynamicAttention(layer.cross_attn)
            graph = ParallelGraph(system).eval()
        with torch.inference_mode():
            actual = graph.logits(x)
        torch.testing.assert_close(actual, expected, atol=.002, rtol=.001)
        assert torch.equal(actual.argmax(-1), expected.argmax(-1))


def test_warm_start_resizes_positions_without_changing_other_weights(tmp_path):
    cfg = config()
    before = create_system(dict(cfg, image_size=[64,256]), pretrained=False)
    checkpoint = tmp_path/'warm.pt'
    torch.save(before.model.state_dict(), checkpoint)
    after = create_system(dict(cfg, initial_checkpoint=checkpoint, initial_image_size=[64,256]), pretrained=False)
    assert after.model.encoder.pos_embed.shape[1] == 12*24
    torch.testing.assert_close(after.model.head.weight, before.model.head.weight)
    assert after.model.max_label_length == 12


def test_stage_selection_never_initializes_from_full_data_fit(tmp_path, monkeypatch):
    from PIL import Image
    from parseq_ocr import fast_training
    folder = tmp_path/'dataset/train'
    folder.mkdir(parents=True)
    for i, label in enumerate(['AA12','BB34','CC56','DD78']):
        Image.new('RGB',(40,20),(i*50,0,0)).save(folder/f'{i}.png')
        (folder/f'{i}.txt').write_text(label)
    calls = []
    def fit(records, held, cfg, out, epochs, resume):
        calls.append((cfg.copy(), [r['label'] for r in records], [r['label'] for r in held]))
        out.mkdir(parents=True, exist_ok=True)
        (out/('best.pt' if held else 'model.pt')).touch()
        return {'selected_epoch': 1}
    monkeypatch.setattr(fast_training, '_fit', fit)
    cfg = config()
    cfg.update(dataset_root=tmp_path/'dataset', output_dir=tmp_path/'runs',
               warm_start_epochs=1, adaptation_epochs=1, epochs=1)
    fast_training.train_fast_tiny(cfg)
    assert len(calls) == 6
    for index in [0,2,4]:
        assert not set(calls[index][1]) & set(calls[index][2])
        assert calls[index][2] == calls[0][2]
    for index in [2,4]:
        assert calls[index][0]['initial_checkpoint'].parts[-2:] == ('selection','best.pt')
        assert calls[index+1][0]['initial_checkpoint'].parts[-2:] == ('full_fit','model.pt')


def _cached_recipe(tmp_path, monkeypatch, refit=False):
    import json
    from PIL import Image
    from parseq_ocr import fast_training
    folder = tmp_path/'dataset/train'
    folder.mkdir(parents=True)
    for i, label in enumerate(['AA12', 'BB34', 'CC56', 'DD78']):
        Image.new('RGB', (40, 20), (i*50, 0, 0)).save(folder/f'{i}.png')
        (folder/f'{i}.txt').write_text(label)
    calls = []
    def fit(records, held, cfg, out, epochs, resume):
        calls.append(cfg.copy())
        out.mkdir(parents=True, exist_ok=True)
        summary = {'selected_epoch': 1}
        (out/'summary.json').write_text(json.dumps(summary))
        (out/('best.pt' if held else 'model.pt')).touch()
        return summary
    monkeypatch.setattr(fast_training, '_fit', fit)
    cfg = config()
    cfg.update(dataset_root=tmp_path/'dataset', output_dir=tmp_path/'runs',
               warm_start_epochs=1, adaptation_epochs=1, epochs=1,
               refit_all_training=refit)
    fast_training.train_fast_tiny(cfg)
    return fast_training, cfg, calls, folder


def test_fast_tiny_selection_only_skips_all_refits(tmp_path, monkeypatch):
    fast_training, cfg, calls, _ = _cached_recipe(tmp_path, monkeypatch)
    assert len(calls) == 3
    assert not list(cfg['output_dir'].rglob('full_fit'))
    for call in calls[1:]:
        assert call['initial_checkpoint'].parts[-2:] == ('selection', 'best.pt')
    fast_training.train_fast_tiny(cfg)
    assert len(calls) == 3  # unchanged completed stages are reused


@pytest.mark.parametrize('change', ['label', 'pixels', 'membership', 'legacy'])
def test_fast_tiny_rejects_changed_input_before_cache_or_split_write(tmp_path, monkeypatch, change):
    import json
    from PIL import Image
    fast_training, cfg, calls, folder = _cached_recipe(tmp_path, monkeypatch)
    split = cfg['output_dir']/'split.json'
    before = split.read_bytes()
    if change == 'label':
        (folder/'0.txt').write_text('ZZ99')
    elif change == 'pixels':
        Image.new('RGB', (40, 20), (255, 255, 255)).save(folder/'0.png')
    elif change == 'membership':
        original = fast_training.split_holdout
        monkeypatch.setattr(fast_training, 'split_holdout', lambda *a: tuple(reversed(original(*a))))
    else:
        (cfg['output_dir']/'input_fingerprint.json').unlink()
    with pytest.raises(ValueError, match='new output_dir'):
        fast_training.train_fast_tiny(cfg)
    assert len(calls) == 3
    assert split.read_bytes() == before
