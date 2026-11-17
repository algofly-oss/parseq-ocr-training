"""TensorRT single-graph build and optional CUDA graph replay, using PyTorch CUDA."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import tensorrt as trt
from .data import normalize
from .inference import raw_image_tensor

LOGGER = trt.Logger(trt.Logger.ERROR)


def build_single_graph(folder, max_batch=32, precision="fp16", force=False):
    folder = Path(folder)
    if max_batch < 1 or precision not in ("fp16", "fp32", "bf16"):
        raise ValueError("Invalid max_batch or precision")
    source, target = folder/"model.onnx", folder/"model.engine"
    meta = json.loads((folder/"metadata.json").read_text())
    info = {"onnx_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "precision_enabled": precision, "decoder": "FP32 constrained" if precision != "fp32" else "FP32",
            "max_batch": max_batch, "gpu": torch.cuda.get_device_name(), "tensorrt_version": trt.__version__}
    resize_preview = getattr(trt.PreviewFeature, "RUNTIME_ACTIVATION_RESIZE_10_10", None)
    info["activation_resize"] = resize_preview is not None
    sidecar = folder/"model.build.json"
    if target.exists() and not force:
        if not sidecar.exists() or json.loads(sidecar.read_text()) != info:
            raise ValueError("Cached engine differs from source/target. Rebuild with force=True.")
        return {"model": info}
    builder = trt.Builder(LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, LOGGER)
    if not parser.parse(source.read_bytes()):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    if resize_preview is not None:
        config.set_preview_feature(resize_preview, True)
    if precision != "fp32":
        config.set_flag(trt.BuilderFlag.FP16 if precision == "fp16" else trt.BuilderFlag.BF16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        for layer in network:
            if "/encoder/" not in layer.name and layer.type != trt.LayerType.CONSTANT:
                floating = [i for i in range(layer.num_outputs) if layer.get_output(i).dtype in (trt.float32, trt.float16, trt.bfloat16)]
                if floating:
                    layer.precision = trt.float32
                    for i in floating:
                        layer.set_output_type(i, trt.float32)
    h, w = meta["height"], meta["width"]
    profile = builder.create_optimization_profile()
    profile.set_shape("pixels", (1,3,h,w), (min(8,max_batch),3,h,w), (max_batch,3,h,w))
    config.add_optimization_profile(profile)
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError("Single-graph TensorRT build failed")
    target.write_bytes(blob)
    sidecar.write_text(json.dumps(info, indent=2))
    (folder/"engine_build.json").write_text(json.dumps({"model": info}, indent=2))
    return {"model": info}


class _Execution:
    def __init__(self, engine, batch, meta, capture):
        self.context = engine.create_execution_context(trt.ExecutionContextAllocationStrategy.USER_MANAGED)
        if not self.context.set_input_shape("pixels", (batch,3,meta["height"],meta["width"])):
            raise ValueError("Unsupported execution shape")
        self.workspace_bytes = self.context.update_device_memory_size_for_shapes()
        if self.workspace_bytes < 0:
            raise RuntimeError("Could not determine TensorRT activation memory")
        # Keep this allocation alive for the execution context and captured graph.
        # CUDA allocations must provide at least 256-byte alignment.
        allocated = max(256, self.workspace_bytes)
        self.workspace = torch.empty(allocated + 255, device="cuda", dtype=torch.uint8)
        address = (self.workspace.data_ptr() + 255) // 256 * 256
        self.context.set_device_memory(address, allocated)
        self.buffers = {}
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            dtype = torch.from_numpy(np.empty((), dtype=trt.nptype(engine.get_tensor_dtype(name)))).dtype
            shape = tuple(self.context.get_tensor_shape(name))
            self.buffers[name] = torch.zeros(shape, device="cuda", dtype=dtype)
            self.context.set_tensor_address(name, self.buffers[name].data_ptr())
        self.graph = None
        for _ in range(2):
            self.enqueue()
        torch.cuda.synchronize()
        if capture:
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.enqueue()

    def enqueue(self):
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT enqueue failed")

    def run(self, pixels):
        self.buffers["pixels"].copy_(torch.from_numpy(pixels))
        if self.graph is None:
            self.enqueue()
        else:
            self.graph.replay()
        return self.buffers["ids"].cpu().tolist()


class SingleGraphTensorRTRecognizer:
    """One model engine; separate persistent execution/capture per used batch size.

    CUDA capture skips Python engine dispatch on replay. Input copying and text
    decoding remain outside the graph. Activation memory is allocated per used
    shape; enabling runtime activation resize at engine build avoids reserving
    maximum-batch scratch for every small batch. Captured shapes retain their
    buffers until close(). Instances are not thread safe.
    """
    def __init__(self, folder, cuda_graph=None):
        folder = Path(folder)
        self.meta = json.loads((folder/"metadata.json").read_text())
        self.runtime = trt.Runtime(LOGGER)
        self.engine = self.runtime.deserialize_cuda_engine((folder/"model.engine").read_bytes())
        if self.engine is None:
            raise RuntimeError("TensorRT deserialization failed")
        self.max_batch = self.engine.get_tensor_profile_shape("pixels", 0)[2][0]
        self.capture = self.meta.get("cuda_graph", True) if cuda_graph is None else cuda_graph
        self.executions = {}

    @torch.inference_mode()
    def infer(self, images):
        n, m = len(images), self.meta
        if not 1 <= n <= self.max_batch:
            raise ValueError(f"Batch must be 1..{self.max_batch}")
        pixels = raw_image_tensor(images, (m["height"], m["width"]))
        if n not in self.executions:
            self.executions[n] = _Execution(self.engine, n, m, self.capture)
        rows = self.executions[n].run(pixels)
        texts = []
        for ids in rows:
            if m["eos_id"] in ids:
                ids = ids[:ids.index(m["eos_id"])]
            texts.append("".join(m["vocabulary"][i] for i in ids))
        return texts

    def close(self):
        torch.cuda.synchronize()
        self.executions.clear()
