"""Train-only epoch selection followed by an optional full-training refit."""
from pathlib import Path
import contextlib
import json
import math
import random
import time
import numpy as np
import torch
from .config import serializable
from .data import scan_pairs, split_holdout, validate_labels, make_loader, normalize
from .models import create_system
from .metrics import score_pairs


def autocast(config):
    if config["precision"] == "fp32":
        return contextlib.nullcontext()
    if config["precision"] != "bf16" or config["device"] != "cuda":
        raise ValueError("Use fp32 for CPU or bf16 on a CUDA device supporting BF16")
    if not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support BF16; set precision: fp32")
    return torch.autocast("cuda", dtype=torch.bfloat16)


@torch.inference_mode()
def evaluate_records(system, records, config, output=None):
    system.eval()
    labels, predictions, rows = [], [], []
    for pixels, texts, names in make_loader(records, config):
        with autocast(config):
            logits = system(pixels.to(config["device"]))
        decoded = system.tokenizer.decode(logits.float().softmax(-1))[0]
        for name, gt, pred in zip(names, texts, decoded):
            pred = normalize(pred, config["normalization"])
            labels.append(gt)
            predictions.append(pred)
            rows.append({"image": name, "gt": gt, "prediction": pred})
    summary = score_pairs(labels, predictions, config["normalization"])
    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        (output/"summary.json").write_text(json.dumps(summary, indent=2))
        (output/"predictions.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    return summary


def _seed(config):
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(4)


def _fit(records, held, config, folder, epochs, resume):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    _seed(config)
    system = create_system(config).to(config["device"])
    validate_labels(records+held, system.tokenizer, config.get("max_label_length", 25))
    system.rng = np.random.default_rng(config["seed"])
    # The reference objective is unchanged; standalone loops do not have a Lightning Trainer.
    system.log = lambda *args, **kwargs: None
    optimizer = torch.optim.AdamW(system.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    training = make_loader(records, config, augment=True)
    total_steps = epochs*len(training)
    start_epoch, steps, selected_epoch, best = 1, 0, 0, (-1., -float("inf"))
    last = folder/"last.pt"
    if last.exists():
        if not resume:
            raise FileExistsError(f"Existing training state: {last}; use resume=True or a new output_dir")
        # Only restore locally generated training state, never downloaded optimizer pickles.
        state = torch.load(last, map_location=config["device"], weights_only=False)
        system.model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, steps = state["epoch"]+1, state["steps"]
        selected_epoch, best = state["selected_epoch"], tuple(state["best"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        system.rng.bit_generator.state = state["permutation_rng"]
    history_path = folder/"history.jsonl"
    started = time.perf_counter()
    for epoch in range(start_epoch, epochs+1):
        system.train()
        total_loss = 0.
        for index, (pixels, labels, _) in enumerate(training):
            scale = (min(1., (steps+1)/config.get("warmup_steps", 100))*(.1+.9*.5*(1+math.cos(math.pi*steps/total_steps)))
                     if config.get("lr_schedule", "cosine") == "cosine" else 1.)
            for group in optimizer.param_groups:
                group["lr"] = config["learning_rate"]*scale
            optimizer.zero_grad(set_to_none=True)
            if config.get("synthetic_replay_ratio", 0):
                from .parallel import synthesize
                replay, replay_labels = synthesize(max(1, int(len(labels)*config["synthetic_replay_ratio"])),
                                                   *config["image_size"], config.get("max_label_length", 12))
                pixels = torch.cat([pixels, replay])
                labels = list(labels)+replay_labels
            with autocast(config):
                if config.get("training_objective") == "parallel":
                    from .parallel import parallel_loss
                    loss = parallel_loss(system, pixels.to(config["device"]), list(labels))
                else:
                    loss = system.training_step((pixels.to(config["device"]), list(labels)), index)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(system.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            total_loss += loss.item()
        row = {"epoch": epoch, "loss": total_loss/len(training),
               "seconds_this_process": time.perf_counter()-started}
        if held:
            row["holdout"] = evaluate_records(system, held, config)
            score = (row["holdout"]["exact_accuracy"], -row["holdout"]["cer"])
        else:
            score = (epoch, 0.)
        if score > best:
            best, selected_epoch = score, epoch
            torch.save(system.model.state_dict(), folder/("best.pt" if held else "model.pt"))
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row)+"\n")
        torch.save({"model": system.model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch, "steps": steps, "selected_epoch": selected_epoch, "best": best,
                    "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state(), "permutation_rng": system.rng.bit_generator.state}, last)
        print(json.dumps(row), flush=True)
    summary = {"epochs": epochs, "selected_epoch": selected_epoch,
               "selection_score": list(best) if held else None,
               "training_images": len(records), "holdout_images": len(held),
               "parameters": sum(p.numel() for p in system.parameters())}
    (folder/"summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def train(config, resume=True):
    if config.get("recipe") == "fast_tiny":
        from .fast_training import train_fast_tiny
        return train_fast_tiny(config, resume)
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use a GPU or device: cpu with precision: fp32")
    if config["epochs"] < 1:
        raise ValueError("epochs must be positive")
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    snapshot = serializable(config)
    if (output/"config.json").exists() and json.loads((output/"config.json").read_text()) != snapshot:
        raise ValueError("Config differs from the existing run. Choose a new output_dir.")
    (output/"config.json").write_text(json.dumps(snapshot, indent=2))
    records = scan_pairs(config["dataset_root"]/"train", config["normalization"])
    training, held = split_holdout(records, config["holdout_fraction"], config["split_seed"])
    (output/"split.json").write_text(json.dumps({"train": [r["image"] for r in training],
                                                "holdout": [r["image"] for r in held]}, indent=2))
    summary = output/"selection/summary.json"
    selection = json.loads(summary.read_text()) if summary.exists() else _fit(
        training, held, config, output/"selection", config["epochs"], resume)
    if config["refit_all_training"]:
        summary = output/"full_fit/summary.json"
        if not summary.exists():
            _fit(records, [], config, output/"full_fit", selection["selected_epoch"], resume)
    return selection
