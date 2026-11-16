"""A single graph for image encoding, parallel character prediction and refinement."""
import json
from pathlib import Path
import torch
from .models import create_system
from .export import DynamicAttention


class ParallelGraph(torch.nn.Module):
    def __init__(self, system):
        super().__init__()
        m, tok = system.model, system.tokenizer
        self.encoder, self.embedding, self.decoder, self.head = m.encoder, m.text_embed, m.decoder, m.head
        n = m.max_label_length+1
        self.queries = torch.nn.Parameter(m.pos_queries[:, :n].detach().clone())
        self.bos, self.eos, self.refinements = tok.bos_id, tok.eos_id, m.refine_iters
        self.register_buffer("cloze", torch.diag(torch.ones(n-1, dtype=torch.bool, device=m.pos_queries.device), diagonal=1))
        self.register_buffer("positions", torch.arange(n, device=m.pos_queries.device, dtype=torch.float32)[None])

    def logits(self, pixels):
        memory = self.encoder(pixels/127.5-1)
        b, n = pixels.shape[0], self.queries.shape[1]
        query = self.queries.expand(b, -1, -1)
        bos = torch.full((b,1), self.bos, device=pixels.device, dtype=torch.int64)
        z = self.head(self.decoder(query, self.embedding(bos), memory))
        for _ in range(self.refinements):
            tokens = torch.cat([bos, z[:, :-1].argmax(-1)], 1)
            context = torch.cat([self.embedding(tokens[:, :1]),
                                 self.queries[:, :n-1]+self.embedding(tokens[:, 1:])], 1)
            first = torch.where(tokens == self.eos, self.positions,
                                self.positions.new_full((), float(n))).amin(-1, keepdim=True)
            padding = self.positions >= first
            z = self.head(self.decoder(query, context, memory, self.cloze, None, padding))
        return z

    def forward(self, pixels):
        return self.logits(pixels).argmax(-1).to(torch.int32)


def export_single_graph(config, checkpoint=None, output=None):
    checkpoint = Path(checkpoint) if checkpoint else config["output_dir"]/"full_fit/model.pt"
    output = Path(output) if output else config["output_dir"]/"deployment"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. Train first.")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(config["seed"])
    system = create_system(config, checkpoint).to(config["device"]).eval()
    for block in system.model.encoder.blocks:
        block.attn.fused_attn = False
    h, w = config["image_size"]
    probe = torch.rand(2,3,h,w, device=config["device"])*255
    with torch.no_grad():
        expected = system(probe/127.5-1)
    for layer in system.model.decoder.layers:
        layer.self_attn = DynamicAttention(layer.self_attn)
        layer.cross_attn = DynamicAttention(layer.cross_attn)
    graph = ParallelGraph(system).eval()
    with torch.no_grad():
        actual = graph.logits(probe)
    error = float((expected-actual).abs().max())
    agreement = float((expected.argmax(-1) == actual.argmax(-1)).float().mean())
    if error >= .002 or agreement != 1:
        raise RuntimeError(f"Single-graph math parity failed: {error=}, {agreement=}")
    (output/"export_parity.json").write_text(json.dumps({"max_logit_error": error,
        "argmax_agreement": agreement, "scope": "FP32 synthetic forward; native vs single-graph math"}, indent=2))
    tok = system.tokenizer
    (output/"metadata.json").write_text(json.dumps({"graph_type": "single_graph", "height": h,
        "width": w, "max_steps": system.model.max_label_length+1, "vocabulary": tok._itos,
        "eos_id": tok.eos_id, "normalization": config["normalization"],
        "input": "RGB float32 NCHW pixels in [0,255]; normalization is embedded",
        "resize": "Pillow bicubic", "decoding": {"autoregressive": False, "refinements": 2},
        "cuda_graph": config.get("cuda_graph", True),
        "trt_precision": config.get("trt_precision", "fp16")}, indent=2))
    torch.onnx.export(graph, (probe,), str(output/"model.onnx"), input_names=["pixels"],
        output_names=["ids"], dynamic_axes={"pixels": {0: "batch"}, "ids": {0: "batch"}},
        opset_version=17, dynamo=False)
    return output
