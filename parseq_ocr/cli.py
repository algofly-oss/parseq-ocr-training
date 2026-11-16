"""CLI equivalents of the numbered notebook workflow."""
import argparse
import json
from pathlib import Path
from .config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "infer", "export", "build", "evaluate", "benchmark"])
    parser.add_argument("--config", default="configs/alpr.yaml")
    parser.add_argument("--backend", choices=["torch", "onnx", "tensorrt"], default="torch")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--folder", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    deployment = config["output_dir"]/"deployment"
    if args.command == "train":
        from .training import train
        print(json.dumps(train(config), indent=2))
        return
    if args.command == "export":
        from .export import export_onnx
        print(export_onnx(config, args.checkpoint))
        return
    if args.command == "build":
        from .tensorrt import build_engines
        print(json.dumps(build_engines(deployment, args.max_batch, force=args.force), indent=2))
        return
    if args.command == "benchmark":
        if args.folder is None:
            parser.error("benchmark requires --folder")
        from .benchmark import benchmark
        result = benchmark(config, args.backend, args.folder, args.batch_size, args.repeats)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return
    recognizer = make_recognizer(config, args.backend, args.checkpoint)
    if args.command == "infer":
        if args.image is None:
            parser.error("infer requires --image")
        from PIL import Image
        with Image.open(args.image) as image:
            text = recognizer.infer([image.convert("RGB")])[0]
        print(json.dumps({"text": text, "backend": args.backend}))
    else:
        if args.folder is None:
            parser.error("evaluate requires --folder")
        from .evaluation import evaluate_folder
        print(json.dumps(evaluate_folder(recognizer, args.folder, config["normalization"],
                                         args.batch_size, args.output), indent=2))


def make_recognizer(config, backend, checkpoint=None):
    if backend == "tensorrt":
        from .tensorrt import TensorRTRecognizer
        return TensorRTRecognizer(config["output_dir"]/"deployment")
    from .inference import TorchRecognizer, ONNXRecognizer
    if backend == "onnx":
        return ONNXRecognizer(config["output_dir"]/"deployment")
    return TorchRecognizer(config, checkpoint)


if __name__ == "__main__":
    main()
