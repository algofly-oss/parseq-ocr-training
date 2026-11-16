"""Common image-to-text API for PyTorch, ONNX Runtime and TensorRT."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import torch
from .data import normalize
from .models import create_system
from .training import autocast


def raw_image_tensor(images, size):
    h, w = size
    return np.stack([np.asarray(im.convert("RGB").resize((w, h), Image.Resampling.BICUBIC),
                                dtype=np.float32).transpose(2, 0, 1) for im in images])


def image_tensor(images, size):
    return raw_image_tensor(images, size)/127.5-1


class TorchRecognizer:
    def __init__(self, config, checkpoint=None):
        self.config = config
        checkpoint = Path(checkpoint) if checkpoint else config["output_dir"]/"full_fit/model.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. Run notebook2 first.")
        self.system = create_system(config, checkpoint).eval().to(config["device"])

    @torch.inference_mode()
    def infer(self, images):
        tensor = torch.from_numpy(image_tensor(images, self.config["image_size"])).to(self.config["device"])
        with autocast(self.config):
            logits = self.system(tensor)
        texts = self.system.tokenizer.decode(logits.float().softmax(-1))[0]
        return [normalize(text, self.config["normalization"]) for text in texts]


class ONNXRecognizer:
    def __init__(self, folder, providers=None):
        import onnxruntime as ort
        folder = Path(folder)
        self.meta = json.loads((folder/"metadata.json").read_text())
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        def session(name):
            return ort.InferenceSession(str(folder/f"{name}.onnx"), options,
                                        providers=providers or ["CPUExecutionProvider"])
        self.single = session("model") if self.meta.get("graph_type") == "single_graph" else None
        if self.single:
            return
        self.encoder, self.decoder = session("encoder"), session("decoder")
        self.parallel = session("parallel") if not self.meta["decoding"]["autoregressive"] else None
        self.refine = session("refine") if self.meta["decoding"]["refinements"] else None

    def infer(self, images):
        m, b = self.meta, len(images)
        if self.single:
            rows = self.single.run(None, {"pixels": raw_image_tensor(images, (m["height"],m["width"]))})[0]
            texts = []
            for ids in rows.tolist():
                if m["eos_id"] in ids:
                    ids = ids[:ids.index(m["eos_id"])]
                texts.append(normalize("".join(m["vocabulary"][i] for i in ids), m["normalization"]))
            return texts
        pixels = image_tensor(images, (m["height"], m["width"]))
        memory = self.encoder.run(None, {"pixels": pixels})[0]
        tokens = np.full((b, 1), m["bos_id"], dtype=np.int64)
        if self.parallel:
            logits = self.parallel.run(None, {"memory": memory, "tokens": tokens})[0]
        else:
            emitted, finished = [], np.zeros(b, dtype=bool)
            for _ in range(m["max_steps"]):
                z = self.decoder.run(None, {"memory": memory, "tokens": tokens})[0]
                emitted.append(z)
                ids = z[:, 0].argmax(-1)
                tokens = np.concatenate([tokens, ids[:, None]], 1)
                finished |= ids == m["eos_id"]
                if finished.all():
                    break
            logits = np.concatenate(emitted, 1)
        for _ in range(m["decoding"]["refinements"]):
            prefix = np.concatenate([np.full((b, 1), m["bos_id"], np.int64),
                                     logits[:, :-1].argmax(-1)], 1)
            logits = self.refine.run(None, {"memory": memory, "tokens": prefix})[0]
        if not np.isfinite(logits).all():
            raise RuntimeError("Nonfinite ONNX logits")
        texts = []
        for ids in logits.argmax(-1).tolist():
            if m["eos_id"] in ids:
                ids = ids[:ids.index(m["eos_id"])]
            texts.append(normalize("".join(m["vocabulary"][i] for i in ids), m["normalization"]))
        return texts
