"""Create one reusable, stratified CSV split manifest for all group models."""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def create_manifest(data_dir, output, seed=42, train_ratio=.70, validation_ratio=.15, classes=None):
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir(): raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    allowed = set(classes) if classes else None
    records = [(path, directory.name) for directory in sorted(root.iterdir())
               if directory.is_dir() and (allowed is None or directory.name in allowed)
               for path in sorted(directory.rglob("*"))
               if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    if not records: raise ValueError("No images found in class folders")
    rng = np.random.default_rng(seed); rows = []
    for label in sorted({label for _, label in records}):
        items = [(path, name) for path, name in records if name == label]
        order = rng.permutation(len(items)); n = len(items)
        n_train = int(n * train_ratio); n_validation = int(n * validation_ratio)
        for position, index in enumerate(order):
            split = "train" if position < n_train else "validation" if position < n_train + n_validation else "test"
            rows.append({"path": str(items[index][0]), "label": label, "split": split})
    destination = Path(output); destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "split"]); writer.writeheader(); writer.writerows(rows)
    counts = {split: sum(row["split"] == split for row in rows) for split in ("train", "validation", "test")}
    print(f"Wrote {len(rows):,} records to {destination}"); print(counts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--classes", nargs="+")
    args = parser.parse_args(); create_manifest(args.data_dir, args.output, args.seed, classes=args.classes)
