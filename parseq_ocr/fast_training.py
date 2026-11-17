from .metrics import require_exact_selection, EXACT_POLICY
"""Leakage-free warm start, parallel adaptation and smaller-input replay stages."""
import json
import hashlib
from PIL import Image
import torch
from .config import serializable
from .data import scan_pairs, split_holdout
from .training import _fit


def _input_fingerprint(records, training, held, replay_ratio):
    rows = []
    fit_ids = {r["image"] for r in training}
    held_ids = {r["image"] for r in held}
    for row in records:
        with Image.open(row["path"]) as image:
            rgb = image.convert("RGB")
            rows.append(dict(image=row["image"], label=row["label"],
                             size=list(rgb.size), rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                             membership="fit" if row["image"] in fit_ids else "held"))
    if fit_ids & held_ids or fit_ids | held_ids != {r["image"] for r in records}:
        raise ValueError("Invalid training/holdout membership")
    return dict(version=1, rows=rows, replay_ratio=replay_ratio,
                fit_order=[r["image"] for r in training],
                held_order=[r["image"] for r in held])


def train_fast_tiny(config, resume=True):
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose CPU with precision: fp32 for functional checks")
    if config["epochs"] < 1 or config.get("synthetic_replay_ratio", .25) < 0:
        raise ValueError("epochs must be positive and replay ratio nonnegative")
    output = config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    snapshot = serializable(config)
    path = output/"config.json"
    if path.exists() and json.loads(path.read_text()) != snapshot:
        raise ValueError("Config differs from existing run; choose a new output_dir")
    records = scan_pairs(config["dataset_root"]/"train", config["normalization"])
    training, held = split_holdout(records, config["holdout_fraction"], config["split_seed"])
    fingerprint = _input_fingerprint(records, training, held,
                                     config.get("synthetic_replay_ratio", .25))
    fingerprint_path = output/"input_fingerprint.json"
    if fingerprint_path.exists():
        if json.loads(fingerprint_path.read_text()) != fingerprint:
            raise ValueError("Input content or split changed; choose a new output_dir")
    elif any(output.rglob("*.pt")) or any(output.rglob("summary.json")):
        raise ValueError("Legacy run lacks an input fingerprint; choose a new output_dir")
    # Validate identity before overwriting any persisted configuration or split.
    fingerprint_path.write_text(json.dumps(fingerprint, indent=2))
    path.write_text(json.dumps(snapshot, indent=2))
    (output/"split.json").write_text(json.dumps({"train": [r["image"] for r in training],
                                                "holdout": [r["image"] for r in held]}, indent=2))
    max_length = config.get("max_label_length", 12)
    if any(len(r["label"]) > max_length for r in records):
        raise ValueError(f"Fast Tiny supports at most {max_length} characters; labels are never truncated")
    stages = [
        ("warm_start", dict(image_size=[64,256], autoregressive=True, refinements=1,
                            training_objective="permutation", synthetic_replay_ratio=0,
                            learning_rate=config.get("warm_start_learning_rate", 1e-4),
                            lr_schedule="cosine"), config.get("warm_start_epochs", 20)),
        ("parallel_adaptation", dict(image_size=[64,256], autoregressive=False, refinements=2,
                                     training_objective="parallel", synthetic_replay_ratio=0,
                                     learning_rate=config.get("adaptation_learning_rate", 3e-5),
                                     lr_schedule="constant"), config.get("adaptation_epochs", 5)),
        ("final", dict(image_size=config["image_size"], autoregressive=False, refinements=2,
                       training_objective="parallel",
                       synthetic_replay_ratio=config.get("synthetic_replay_ratio", .25),
                       lr_schedule="constant"), config["epochs"]),
    ]
    summaries, previous, previous_size = [], None, None
    for name, overrides, epochs in stages:
        folder = output if name == "final" else output/name
        cfg = dict(config, **overrides)
        cfg.update(recipe="standard", max_label_length=max_length, epochs=epochs, output_dir=folder)
        selection_cfg = dict(cfg)
        if previous:
            selection_cfg.update(initial_checkpoint=previous/"selection/best.pt",
                                 initial_image_size=previous_size)
        summary = folder/"selection/summary.json"
        selected = json.loads(summary.read_text()) if summary.exists() else _fit(
            training, held, selection_cfg, folder/"selection", epochs, resume)
        require_exact_selection(selected, str(summary))
        full_cfg = dict(cfg)
        if previous and config["refit_all_training"]:
            full_cfg.update(initial_checkpoint=previous/"full_fit/model.pt",
                            initial_image_size=previous_size)
        if (folder/"full_fit/summary.json").exists():
            require_exact_selection(json.loads((folder/"full_fit/summary.json").read_text()), str(folder/"full_fit/summary.json"))
        if config["refit_all_training"] and not (folder/"full_fit/summary.json").exists():
            _fit(records, [], full_cfg, folder/"full_fit", selected["selected_epoch"], resume)
        folder.mkdir(parents=True, exist_ok=True)
        (folder/"stage_config.json").write_text(json.dumps({"selection": serializable(selection_cfg),
                                                             "full_fit": serializable(full_cfg)}, indent=2))
        summaries.append(dict(stage=name, **selected))
        previous, previous_size = folder, overrides["image_size"]
    result = {"selection_policy": EXACT_POLICY, "recipe": "fast_tiny", "stages": summaries,
              "selection_protocol": "Every stage uses the same training-only label-disjoint holdout. "
                                    "Selection starts from preceding selection/best.pt; refit starts "
                                    "from preceding full_fit/model.pt when refit_all_training is enabled. Validation is never read.",
              "synthetic_replay": "Random uppercase/digits only; ratio .25 adds 1 synthetic per 4 real images (20% effective batch). "
                                  "Fonts vary by platform; exact historical accuracy is not guaranteed."}
    (output/"recipe_summary.json").write_text(json.dumps(result, indent=2))
    return result
