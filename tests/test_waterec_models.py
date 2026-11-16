"""Portable profile, token-remap and split checks without downloads or a GPU."""
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from PIL import Image
from parseq_ocr.waterec_models import NativeWATERec, create_native, preprocess_image
from parseq_ocr.waterec_training import split_training_records, ImageDataset


def toy():
    model = NativeWATERec.__new__(NativeWATERec)
    nn.Module.__init__(model)
    model.config = {'include_space': False}
    model.characters = ['</s>', 'A', 'B', '<s>', '<pad>']
    decoder = nn.Module()
    decoder.d_model, decoder.max_len = 4, 2
    decoder.bos, decoder.eos, decoder.ignore_index = 3, 0, 4
    decoder.embedding = nn.Module()
    decoder.embedding.embedding = nn.Embedding(5, 4)
    decoder.tgt_word_prj = nn.Linear(4, 3, bias=False)
    model.decoder = decoder
    return model


def test_space_preserves_characters_and_bos_eos_pad():
    model = toy()
    embedding = model.decoder.embedding.embedding.weight.detach().clone()
    head = model.decoder.tgt_word_prj.weight.detach().clone()
    model.add_space()
    assert model.characters == ['</s>', 'A', 'B', ' ', '<s>', '<pad>']
    assert torch.equal(model.decoder.embedding.embedding.weight[:3], embedding[:3])
    assert torch.equal(model.decoder.embedding.embedding.weight[4:], embedding[3:])
    assert torch.equal(model.decoder.tgt_word_prj.weight[:3], head)
    assert model.decoder.bos == 4 and model.decoder.eos == 0 and model.decoder.ignore_index == 5
    model.add_space()
    assert len(model.characters) == 6


def test_label_targets_and_no_silent_truncation():
    model = toy()
    assert model.encode_labels(['A', 'AB']).tolist() == [[3, 1, 0, 4], [3, 1, 2, 0]]
    for invalid in ['', 'AX', 'ABA']:
        with pytest.raises(ValueError):
            model.encode_labels([invalid])


@pytest.mark.parametrize('field,value', [('task', 'price'), ('variant', 'R'), ('tokens', 384),
                                        ('include_space', True), ('max_label_length', 12)])
def test_checkpoint_profile_mismatch_rejected_before_model_build(tmp_path, field, value):
    config = dict(task='plate', variant='RS', tokens=512, include_space=False, max_label_length=25)
    checkpoint = tmp_path / 'model.pt'
    torch.save({'config': config, 'state_dict': {}}, checkpoint)
    with pytest.raises(ValueError, match='profile mismatch'):
        create_native({**config, field: value}, checkpoint)


def test_patch_geometry_and_dataset_workers_hold_no_model():
    image = Image.new('RGB', (384, 192), (255, 0, 255))
    pixels = preprocess_image(image, 512)
    assert pixels.shape == (3, 64, 128)
    assert torch.all(pixels[0] == 1) and torch.all(pixels[1] == -1)
    dataset = ImageDataset([], SimpleNamespace(config={'tokens': 512}))
    assert dataset.tokens == 512 and not hasattr(dataset, 'model')


def test_price_rounding_is_separate_from_plate_floor():
    records = [{'image': str(i), 'label': str(i)} for i in range(7)]
    _, price = split_training_records(records, dict(task='price', holdout_fraction=.5, split_seed=1))
    _, plate = split_training_records(records, dict(task='plate', holdout_fraction=.5, split_seed=1))
    assert len(price) == 4 and len(plate) == 3
