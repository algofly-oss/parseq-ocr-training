"""Portable configuration: paths resolve relative to the repository root."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path="configs/alpr.yaml"):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    for key in ("dataset_root", "output_dir", "pretrained_dir"):
        value = Path(config[key]).expanduser()
        config[key] = value if value.is_absolute() else ROOT / value
    if config["model"] not in ("parseq", "parseq-tiny"):
        raise ValueError("model must be parseq or parseq-tiny")
    h, w = config["image_size"]
    if h <= 0 or w <= 0 or h % 4 or w % 8:
        raise ValueError("image_size must be positive multiples of patch height4 and width8")
    if not 0 < config["holdout_fraction"] < 1:
        raise ValueError("holdout_fraction must lie between0 and1")
    if config["normalization"] not in ("none", "upper_no_whitespace"):
        raise ValueError("Unsupported annotation-normalization metadata")
    recipe = config.get("recipe", "standard")
    if recipe not in ("standard", "fast_tiny"):
        raise ValueError("recipe must be standard or fast_tiny")
    if not 1 <= config.get("max_label_length", 25) <= 25:
        raise ValueError("max_label_length must be between 1 and 25")
    if recipe == "fast_tiny":
        if config["model"] != "parseq-tiny" or config["autoregressive"] or config["refinements"] != 2:
            raise ValueError("fast_tiny requires parseq-tiny, autoregressive: false, refinements: 2")
        if not config["refit_all_training"]:
            raise ValueError("fast_tiny requires refit_all_training: true for its staged initialization")
        for key in ("warm_start_epochs", "adaptation_epochs"):
            if config.get(key, 5) < 1:
                raise ValueError(f"{key} must be positive")
    return config


def serializable(config):
    return {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
