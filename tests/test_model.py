import torch
from parseq_ocr.config import load_config
from parseq_ocr.models import create_system
from parseq_ocr.data import validate_labels


def test_vendored_tiny_model_loads_without_network():
    torch.set_num_threads(2)
    config = load_config()
    config.update(model="parseq-tiny", image_size=[32,128], autoregressive=False,
                  refinements=0, device="cpu", precision="fp32")
    model = create_system(config, pretrained=False).eval()
    validate_labels([{"image": "sample", "label": 'AB12"'}], model.tokenizer)
    with torch.inference_mode():
        logits = model(torch.zeros(1,3,32,128))
    assert logits.shape == (1,26,95)
    assert torch.isfinite(logits).all()
