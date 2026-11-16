"""Parallel OCR supervision and generic synthetic replay for the Fast Tiny recipe."""
import random
import string
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def stage_logits(system, pixels):
    model, tok = system.model, system.tokenizer
    n, b = model.max_label_length+1, pixels.shape[0]
    memory = model.encode(pixels)
    queries = model.pos_queries[:, :n].expand(b, -1, -1)
    bos = torch.full((b, 1), tok.bos_id, device=pixels.device, dtype=torch.int64)
    logits = model.head(model.decode(bos, memory, tgt_query=queries))
    outputs = [logits]
    mask = torch.diag(torch.ones(n-1, device=pixels.device, dtype=torch.bool), diagonal=1)
    for _ in range(model.refine_iters):
        prefix = torch.cat([bos, logits[:, :-1].argmax(-1)], 1)
        padding = (prefix == tok.eos_id).int().cumsum(-1) > 0
        logits = model.head(model.decode(prefix, memory, tgt_padding_mask=padding,
                                        tgt_query=queries, tgt_query_mask=mask))
        outputs.append(logits)
    return outputs


def parallel_loss(system, pixels, labels):
    """Supervise initial prediction and refinements; prefixes come from predictions."""
    target = system.tokenizer.encode(labels, pixels.device)[:, 1:]
    steps = system.model.max_label_length+1
    if target.shape[1] > steps:
        raise ValueError("Label exceeds Fast Tiny output capacity; labels are never truncated")
    pad = system.tokenizer.pad_id
    target = torch.nn.functional.pad(target, (0, steps-target.shape[1]), value=pad)
    stages = stage_logits(system, pixels)
    return sum(torch.nn.functional.cross_entropy(z.float().flatten(0, 1), target.flatten(),
                                                 ignore_index=pad) for z in stages)/len(stages)


def synthesize(count, height, width, max_length=12):
    """Random uppercase/digit strings, generated without evaluation images or labels."""
    pixels, labels = [], []
    fonts = ["DejaVuSans.ttf", "DejaVuSansMono.ttf", "DejaVuSerif.ttf"]
    for _ in range(count):
        text = "".join(random.choices(string.ascii_uppercase+string.digits,
                                       k=random.randint(min(5, max_length), max_length)))
        background = random.choice([(245,245,240), (235,220,90), (35,50,80), (210,210,210)])
        ink = (5,5,5) if sum(background) > 400 else (245,245,245)
        image = Image.new("RGB", (width, height), background)
        draw = ImageDraw.Draw(image)
        size = max(10, int(height*random.uniform(.5,.8)))
        font_name = random.choice(fonts)
        def font_at(size):
            try:
                return ImageFont.truetype(font_name, size)
            except OSError:
                return ImageFont.load_default(size=size)
        font = font_at(size)
        while draw.textbbox((0,0), text, font=font)[2] > width*.94 and size > 8:
            size -= 1
            font = font_at(size)
        box = draw.textbbox((0,0), text, font=font)
        draw.text((max(1,(width-box[2])//2), (height-(box[3]-box[1]))//2-box[1]),
                  text, font=font, fill=ink)
        image = image.rotate(random.uniform(-4,4), Image.Resampling.BICUBIC, fillcolor=background)
        pixels.append(np.asarray(image, dtype=np.float32).transpose(2,0,1)/127.5-1)
        labels.append(text)
    return torch.from_numpy(np.stack(pixels)), labels
