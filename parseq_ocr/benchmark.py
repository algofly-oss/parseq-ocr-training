"""Use the CLI in a fresh process for meaningful whole-process memory figures."""
import os
import threading
import warnings
import time
import resource
import numpy as np
import torch
from PIL import Image
from .data import scan_pairs


def benchmark(config, backend, folder, batch_size=1, repeats=3, limit=256):
    from .cli import make_recognizer
    if batch_size < 1 or repeats < 1:
        raise ValueError("batch_size and repeats must be positive")
    torch.set_num_threads(4)
    records = scan_pairs(folder, config["normalization"])[:limit]
    sampler = GPUMemorySampler() if backend == "tensorrt" or config["device"] == "cuda" and backend == "torch" else None
    started = time.perf_counter()
    recognizer = make_recognizer(config, backend)
    cuda = backend == "tensorrt" or backend == "torch" and config["device"] == "cuda"
    def sync():
        if cuda:
            torch.cuda.synchronize()
    def images(rows):
        result = []
        for row in rows:
            with Image.open(row["path"]) as im:
                result.append(im.convert("RGB"))
        return result
    sync()
    load_seconds = time.perf_counter()-started
    warmup = images(records[:batch_size])
    first = time.perf_counter()
    recognizer.infer(warmup)
    sync()
    first_seconds = time.perf_counter()-first
    for _ in range(5):
        recognizer.infer(warmup)
    sync()
    latencies, passes = [], []
    for repeat in range(repeats):
        elapsed = 0.
        for start in range(0, len(records), batch_size):
            sync()
            t = time.perf_counter()
            recognizer.infer(images(records[start:start+batch_size]))
            sync()
            duration = time.perf_counter()-t
            elapsed += duration
            latencies.append(duration*1000)
        passes.append({"repeat": repeat, "seconds": elapsed,
                       "images_per_second": len(records)/elapsed})
    result = {"backend": backend, "batch_size": batch_size, "unique_images": len(records),
              "timed_images": len(records)*repeats, "load_seconds": load_seconds,
              "first_inference_seconds": first_seconds, "median_batch_ms": float(np.median(latencies)),
              "p95_batch_ms": float(np.percentile(latencies,95)),
              "images_per_second": len(records)*repeats/(sum(latencies)/1000),
              "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
              "passes": passes, "scope": "Image reading, preprocessing, inference and decoding included; model load excluded from steady-state timing."}
    if sampler:
        result.update(sampler.close())
    return result

class GPUMemorySampler:
    def __init__(self,interval=.02):
        self.interval=interval;self.peak=0;self.latest=0;self.samples=0;self.error=None
        self.stop=threading.Event();self.thread=None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');import pynvml
            self.nv=pynvml;self.nv.nvmlInit();self.handle=self.nv.nvmlDeviceGetHandleByIndex(0)
            self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
        except Exception as e:self.error=repr(e)

    def sample(self):
        if self.error:return None
        try:
            entries=self.nv.nvmlDeviceGetComputeRunningProcesses(self.handle)
            value=sum(int(p.usedGpuMemory) for p in entries if p.pid==os.getpid() and 0<=int(p.usedGpuMemory)<(1<<50))
            self.latest=value;self.peak=max(self.peak,value);self.samples+=1
            return value
        except Exception as e:self.error=repr(e);return None

    def run(self):
        while not self.stop.is_set():self.sample();self.stop.wait(self.interval)

    def close(self):
        self.sample();self.stop.set()
        if self.thread:self.thread.join(timeout=1)
        result={'nvml_process_peak_bytes':self.peak if self.samples else None,'nvml_samples':self.samples,
                'nvml_sample_interval_seconds':self.interval,'nvml_error':self.error,
                'nvml_memory_note':'Driver-accounted memory for this PID, including CUDA context; sampled peak, not an instantaneous allocation trace.'}
        if self.thread:
            try:self.nv.nvmlShutdown()
            except Exception:pass
        return result
