"""Equivalent dynamic-prefix PARSeq graphs for ONNX and TensorRT."""
from pathlib import Path
import json
import torch
from .models import create_system


def export_onnx(config, checkpoint=None, output=None):
    if config.get("recipe") == "fast_tiny":
        from .parallel_export import export_single_graph
        return export_single_graph(config, checkpoint, output)
    checkpoint = Path(checkpoint) if checkpoint else config["output_dir"]/"full_fit/model.pt"
    output = Path(output) if output else config["output_dir"]/"deployment"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}. Run training first.")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    device = config["device"]
    system = create_system(config, checkpoint).to(device).eval()
    for block in system.model.encoder.blocks:
        block.attn.fused_attn = False
    model, tok = system.model, system.tokenizer
    if len(model.decoder.layers) != 1:
        raise ValueError("Export adapter supports the pinned single-layer reference decoder")
    h, w = config["image_size"]
    torch.manual_seed(config["seed"])
    pixels = torch.randn(2, 3, h, w, device=device)
    with torch.no_grad():
        reference = system(pixels)
    for layer in model.decoder.layers:
        layer.self_attn = DynamicAttention(layer.self_attn)
        layer.cross_attn = DynamicAttention(layer.cross_attn)
    with torch.no_grad():
        memory = model.encode(pixels)
        prefix = torch.full((2, 1), tok.bos_id, device=device, dtype=torch.int64)
        if config["autoregressive"]:
            emitted, finished = [], torch.zeros(2, device=device, dtype=torch.bool)
            for _ in range(model.max_label_length+1):
                z = NextToken(model)(memory, prefix)
                emitted.append(z)
                ids = z[:, 0].argmax(-1)
                prefix = torch.cat([prefix, ids[:, None]], 1)
                finished |= ids == tok.eos_id
                if finished.all().item():
                    break
            converted = torch.cat(emitted, 1)
        else:
            converted = AllQueries(model, tok.eos_id)(memory, prefix)
        for _ in range(config["refinements"]):
            prefix = torch.cat([prefix.new_full((2, 1), tok.bos_id),
                                converted[:, :-1].argmax(-1)], 1)
            converted = AllQueries(model, tok.eos_id, True)(memory, prefix)
    error = float((reference-converted).abs().max())
    agreement = float((reference.argmax(-1) == converted.argmax(-1)).float().mean())
    if error >= .002 or agreement != 1:
        raise RuntimeError(f"Export adapter parity failed: error={error}, agreement={agreement}")
    (output/"export_parity.json").write_text(json.dumps({"max_logit_error": error,
        "argmax_agreement": agreement, "scope": "FP32 synthetic-image forward vs split-graph math"}, indent=2))
    metadata = {"height": h, "width": w, "embedding_dim": model.pos_queries.shape[-1],
                "image_tokens": (h//4)*(w//8), "bos_id": tok.bos_id, "eos_id": tok.eos_id,
                "pad_id": tok.pad_id, "max_steps": model.max_label_length+1,
                "vocabulary": tok._itos, "normalization": config["normalization"],
                "decoding": {"autoregressive": config["autoregressive"],
                             "refinements": config["refinements"]}}
    (output/"metadata.json").write_text(json.dumps(metadata, indent=2))
    torch.onnx.export(Encoder(model), (pixels,), str(output/"encoder.onnx"),
        input_names=["pixels"], output_names=["memory"], opset_version=17, dynamo=False,
        dynamic_axes={"pixels": {0: "batch"}, "memory": {0: "batch"}})
    tokens = torch.tensor([[tok.bos_id, 1], [tok.bos_id, 2]], device=device)
    graphs = {"decoder": NextToken(model)}
    if not config["autoregressive"]:
        graphs["parallel"] = AllQueries(model, tok.eos_id)
    if config["refinements"]:
        graphs["refine"] = AllQueries(model, tok.eos_id, True)
    for name, graph in graphs.items():
        torch.onnx.export(graph, (memory, tokens), str(output/f"{name}.onnx"),
            input_names=["memory", "tokens"], output_names=["logits"],
            opset_version=17, dynamo=False,
            dynamic_axes={"memory": {0: "batch"}, "tokens": {0: "batch", 1: "prefix_length"},
                          "logits": {0: "batch"}})
    return output

class DynamicAttention(torch.nn.Module):
    """Equivalent MHA expressed with explicit dynamic batch/sequence reshapes."""
    def __init__(self,source):
        super().__init__();self.in_proj_weight=source.in_proj_weight;self.in_proj_bias=source.in_proj_bias;self.out_proj=source.out_proj;self.heads=source.num_heads;self.dim=source.embed_dim;self.head_dim=self.dim//self.heads
    def forward(self,query,key,value,attn_mask=None,key_padding_mask=None,**kwargs):
        b,l,_=query.shape;s=key.shape[1];d=self.dim
        def project(x,index):
            z=torch.nn.functional.linear(x,self.in_proj_weight[index*d:(index+1)*d],self.in_proj_bias[index*d:(index+1)*d])
            return z.reshape(b,x.shape[1],self.heads,self.head_dim).transpose(1,2)
        q,k,v=project(query,0),project(key,1),project(value,2)
        scores=(q@k.transpose(-1,-2))*(self.head_dim**-.5)
        if attn_mask is not None:
            if attn_mask.dtype==torch.bool:scores=scores.masked_fill(attn_mask[None,None],float('-inf'))
            else:scores=scores+attn_mask[None,None]
        if key_padding_mask is not None:scores=scores.masked_fill(key_padding_mask[:,None,None,:],float('-inf'))
        probabilities=scores.softmax(-1);result=(probabilities@v).transpose(1,2).reshape(b,l,d)
        return self.out_proj(result),probabilities.mean(1)

class Encoder(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,pixels):return self.model.encode(pixels)

class NextToken(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,memory,tokens):
        b,t=tokens.shape
        mask=torch.triu(torch.ones(t,t,device=tokens.device,dtype=torch.bool),1)
        query=self.model.pos_queries[:,t-1:t].expand(b,-1,-1)
        return self.model.head(self.model.decode(tokens,memory,mask,tgt_query=query,tgt_query_mask=mask[-1:,:]))

class AllQueries(torch.nn.Module):
    def __init__(self,model,eos,refine=False):
        super().__init__();self.model=model;self.eos=eos;self.refine=refine
        n=model.max_label_length+1;self.steps=n;self.register_buffer('cloze_mask',torch.diag(torch.ones(n-1,dtype=torch.bool),diagonal=1).to(model.pos_queries.device))
    def forward(self,memory,tokens):
        b,t=tokens.shape;queries=self.model.pos_queries[:,:self.steps].expand(b,-1,-1)
        if not self.refine:return self.model.head(self.model.decode(tokens,memory,tgt_query=queries))
        n=self.steps
        positions=torch.arange(n,device=tokens.device,dtype=torch.float32)[None,:t]
        first_eos=torch.where(tokens==self.eos,positions,positions.new_full((),float(n))).amin(-1,keepdim=True)
        padding=positions>=first_eos
        return self.model.head(self.model.decode(tokens,memory,tgt_padding_mask=padding,tgt_query=queries,tgt_query_mask=self.cloze_mask[:,:t]))
