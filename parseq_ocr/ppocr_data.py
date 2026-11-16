"""Price data folds; no training framework imports."""
from pathlib import Path
import random


def canonical(rows):
    return [
        dict(
            image=r.get("image", Path(r["path"]).name),
            path=str(r["path"]),
            label=r.get("label", r.get("gt", "")).strip(),
        )
        for r in rows
    ]


def scan_labeled(folder):
    """Image/TXT prices; outer whitespace stripped, blank labels explicitly excluded."""
    from .data_io import IMAGE_EXTENSIONS

    rows = []
    for image in sorted(Path(folder).iterdir()):
        if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = image.with_suffix(".txt")
        if not label.is_file():
            raise ValueError(f"Missing label:{label.name}")
        text = label.read_text().strip()
        if not text:
            continue
        rows.append({"image": image.name, "path": str(image), "label": text})
    if not rows:
        raise ValueError("No nonblank labeled price images")
    return rows


def split_price_holdout(rows, fraction=0.15, seed=20261114):
    labels = sorted({r["label"].strip() for r in rows if r["label"].strip()})
    if len(labels) < 2:
        raise ValueError("Need at least two distinct price labels")
    random.Random(seed).shuffle(labels)
    count = max(1, min(len(labels) - 1, round(len(labels) * fraction)))
    held = set(labels[:count])
    return (
        [r for r in rows if r["label"].strip() not in held],
        [r for r in rows if r["label"].strip() in held],
    )
