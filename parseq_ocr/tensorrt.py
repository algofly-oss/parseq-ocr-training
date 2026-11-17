"""GPU-resident TensorRT inference; training and ONNX Runtime are not required."""
from pathlib import Path
import pathlib
import json
import hashlib
import numpy as np
import torch
import tensorrt as trt
from PIL import Image
from .data import normalize

LOGGER = trt.Logger(trt.Logger.ERROR)


def build_engines(folder, max_batch=32, precision=None, force=False):
    """Build for this GPU; refinement remains FP32 on the tested TensorRT release."""
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    folder = Path(folder)
    meta = json.loads((folder/"metadata.json").read_text())
    if meta.get("graph_type") == "single_graph":
        from .single_graph import build_single_graph
        return build_single_graph(folder, max_batch, precision or meta.get("trt_precision", "fp16"), force)
    precision = precision or "fp16"
    names = ["encoder", "decoder"]
    if not meta["decoding"]["autoregressive"]:
        names.append("parallel")
    if meta["decoding"]["refinements"]:
        names.append("refine")
    if precision not in ("fp16", "bf16", "fp32") or max_batch < 1:
        raise ValueError("Invalid precision or max_batch")
    descriptions = {}
    for name in names:
        source, target = folder/f"{name}.onnx", folder/f"{name}.engine"
        sidecar = folder/f"{name}.build.json"
        info = {"onnx_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "precision_enabled": "fp32" if name == "refine" else precision,
                "max_batch": max_batch, "gpu": torch.cuda.get_device_name(),
                "tensorrt_version": trt.__version__}
        if target.exists() and not force:
            if not sidecar.exists() or json.loads(sidecar.read_text()) != info:
                raise ValueError(f"Cached {name} engine differs from its source/target. Rebuild with force=True.")
            descriptions[name] = info
            continue
        builder = trt.Builder(LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, LOGGER)
        if not parser.parse(source.read_bytes()):
            raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
        if info["precision_enabled"] != "fp32":
            config.set_flag(trt.BuilderFlag.FP16 if info["precision_enabled"] == "fp16" else trt.BuilderFlag.BF16)
        profile = builder.create_optimization_profile()
        optimal = min(8, max_batch)
        if name == "encoder":
            h, w = meta["height"], meta["width"]
            profile.set_shape("pixels", (1,3,h,w), (optimal,3,h,w), (max_batch,3,h,w))
        else:
            n, d = meta["image_tokens"], meta["embedding_dim"]
            profile.set_shape("memory", (1,n,d), (optimal,n,d), (max_batch,n,d))
            profile.set_shape("tokens", (1,1), (optimal,8), (max_batch,meta["max_steps"]))
        config.add_optimization_profile(profile)
        blob = builder.build_serialized_network(network, config)
        if blob is None:
            raise RuntimeError(f"TensorRT failed to build {name}")
        target.write_bytes(blob)
        sidecar.write_text(json.dumps(info, indent=2))
        descriptions[name] = info
        print(f"Built {target.name} ({info['precision_enabled']})", flush=True)
    (folder/"engine_build.json").write_text(json.dumps(descriptions, indent=2))
    return descriptions

class Engine:
    def __init__(self,path):
        self.runtime=trt.Runtime(LOGGER)
        self.engine=self.runtime.deserialize_cuda_engine(pathlib.Path(path).read_bytes())
        if self.engine is None:raise RuntimeError('TensorRT deserialization failed')
        self.context=self.engine.create_execution_context()
        self.names=[self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.outputs=[n for n in self.names if self.engine.get_tensor_mode(n)==trt.TensorIOMode.OUTPUT]

    def __call__(self,inputs):
        buffers={}
        for name,value in inputs.items():
            dtype=torch.from_numpy(np.empty((),dtype=trt.nptype(self.engine.get_tensor_dtype(name)))).dtype
            buffers[name]=value.to(dtype).contiguous()
            if not self.context.set_input_shape(name,tuple(value.shape)):
                raise RuntimeError(f'Invalid input shape {name}: {value.shape}')
        for name in self.outputs:
            shape=tuple(self.context.get_tensor_shape(name))
            dtype=torch.from_numpy(np.empty((),dtype=trt.nptype(self.engine.get_tensor_dtype(name)))).dtype
            buffers[name]=torch.empty(shape,device='cuda',dtype=dtype)
        for name,value in buffers.items():self.context.set_tensor_address(name,value.data_ptr())
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError('TensorRT execution failed')
        return {name:buffers[name] for name in self.outputs}

class TensorRTRecognizer:
    def __init__(self,folder):
        folder=pathlib.Path(folder);self.meta=json.loads((folder/'metadata.json').read_text())
        self.single = None
        if self.meta.get("graph_type") == "single_graph":
            from .single_graph import SingleGraphTensorRTRecognizer
            self.single = SingleGraphTensorRTRecognizer(folder)
            return
        self.encoder=Engine(folder/'encoder.engine');self.decoder=Engine(folder/'decoder.engine')
        self.parallel=Engine(folder/'parallel.engine') if not self.meta['decoding']['autoregressive'] else None
        self.refine=Engine(folder/'refine.engine') if self.meta['decoding']['refinements'] else None
    @torch.inference_mode()
    def infer(self,images):
        if self.single is not None:
            return self.single.infer(images)
        m=self.meta;b=len(images);x=np.stack([np.asarray(im.resize((m['width'],m['height']),Image.Resampling.BICUBIC),dtype=np.float32).transpose(2,0,1) for im in images]);pixels=torch.from_numpy(x).cuda()/127.5-1
        memory=self.encoder({'pixels':pixels})['memory'];tokens=torch.full((b,1),m['bos_id'],device='cuda',dtype=torch.int64)
        if self.parallel:logits=self.parallel({'memory':memory,'tokens':tokens})['logits']
        else:
            emitted=[];finished=torch.zeros(b,device='cuda',dtype=torch.bool)
            for _ in range(m['max_steps']):
                z=self.decoder({'memory':memory,'tokens':tokens})['logits'];emitted.append(z);ids=z[:,0].argmax(-1);tokens=torch.cat([tokens,ids[:,None]],1);finished|=ids==m['eos_id']
                if finished.all().item():break
            logits=torch.cat(emitted,1)
        for _ in range(m['decoding']['refinements']):
            prefix=torch.cat([torch.full((b,1),m['bos_id'],device='cuda',dtype=torch.int64),logits[:,:-1].argmax(-1)],1)
            logits=self.refine({'memory':memory,'tokens':prefix})['logits']
        if not torch.isfinite(logits).all().item():raise RuntimeError('Nonfinite TensorRT logits')
        texts=[]
        for row in logits.argmax(-1).cpu().tolist():
            if m['eos_id'] in row:row=row[:row.index(m['eos_id'])]
            texts.append(''.join(m['vocabulary'][i] for i in row))
        return texts

    def close(self):
        if self.single is not None:
            self.single.close()
