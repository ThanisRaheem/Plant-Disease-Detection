"""Read-only exploratory analysis for a class-folder image dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def discover_images(data_dir: Path, selected_classes: list[str] | None = None):
    """Return image paths grouped by class without changing source files."""
    allowed = set(selected_classes) if selected_classes else None
    class_dirs = sorted(
        path for path in data_dir.iterdir()
        if path.is_dir() and (allowed is None or path.name in allowed)
    )
    return {
        directory.name: sorted(
            path for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for directory in class_dirs
    }


def inspect_images(images_by_class, seed: int, integrity_limit: int = 0):
    """Validate images and collect one reproducible representative per class."""
    rng = random.Random(seed)
    all_images = [(label, path) for label, paths in images_by_class.items() for path in paths]
    to_check = all_images if not integrity_limit else rng.sample(all_images, min(integrity_limit, len(all_images)))
    corrupt = []
    for label, path in to_check:
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError, UnidentifiedImageError) as error:
            corrupt.append({"class": label, "path": str(path), "error": str(error)})

    representatives = []
    corrupt_paths = {row["path"] for row in corrupt}
    for label, paths in images_by_class.items():
        candidates = [path for path in paths if str(path) not in corrupt_paths]
        if not candidates:
            continue
        path = rng.choice(candidates)
        with Image.open(path) as image:
            representatives.append({
                "class": label,
                "path": str(path),
                "width": image.width,
                "height": image.height,
                "channels": len(image.getbands()),
                "mode": image.mode,
            })
    return corrupt, representatives, len(to_check)


def save_distribution(counts: Counter, output_path: Path):
    labels, values = zip(*sorted(counts.items(), key=lambda item: item[1]))
    height = max(6, len(labels) * 0.42)
    fig, ax = plt.subplots(figsize=(12, height))
    bars = ax.barh(labels, values, color="#3b7a57")
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.set(title="PlantVillage Class Distribution", xlabel="Number of images", ylabel="Class")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_samples(representatives, output_path: Path):
    columns = min(4, len(representatives))
    rows = math.ceil(len(representatives) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.4 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, sample in zip(axes.flat, representatives):
        with Image.open(sample["path"]) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(f"{sample['class']}\n{sample['width']}×{sample['height']} · {sample['channels']} ch", fontsize=8)
        ax.axis("off")
    fig.suptitle("Representative PlantVillage Images", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(config_path: str, integrity_limit: int = 0):
    with open(config_path) as handle:
        config = yaml.safe_load(handle)
    data_dir = Path(config["data_dir"]).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")
    output_dir = Path(config.get("output_dir", "results/resnet50")) / "eda"
    output_dir.mkdir(parents=True, exist_ok=True)

    images = discover_images(data_dir, config.get("classes"))
    counts = Counter({label: len(paths) for label, paths in images.items()})
    if not counts or sum(counts.values()) == 0:
        raise ValueError("No supported images found in class folders")
    corrupt, samples, checked = inspect_images(images, config["seed"], integrity_limit)
    minimum, maximum = min(counts.values()), max(counts.values())
    ratio = maximum / minimum if minimum else None
    threshold = 0.5 * maximum
    summary = {
        "dataset_path": str(data_dir),
        "class_count": len(counts),
        "total_images": sum(counts.values()),
        "smallest_class": {"name": min(counts, key=counts.get), "count": minimum},
        "largest_class": {"name": max(counts, key=counts.get), "count": maximum},
        "max_to_min_ratio": round(ratio, 3) if ratio is not None else None,
        "potentially_underrepresented_classes": sorted(label for label, count in counts.items() if count < threshold),
        "integrity_files_checked": checked,
        "corrupt_files_found": len(corrupt),
        "seed": config["seed"],
    }

    with open(output_dir / "class_counts.csv", "w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["class", "image_count"]); writer.writerows(sorted(counts.items()))
    with open(output_dir / "representative_dimensions.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class", "path", "width", "height", "channels", "mode"]); writer.writeheader(); writer.writerows(samples)
    with open(output_dir / "corrupt_images.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class", "path", "error"]); writer.writeheader(); writer.writerows(corrupt)
    with open(output_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    save_distribution(counts, output_dir / "class_distribution.png")
    save_samples(samples, output_dir / "representative_samples.png")

    report = ["# ResNet50 Dataset Inspection", "", f"- Classes: **{summary['class_count']}**", f"- Images: **{summary['total_images']:,}**", f"- Largest class: **{summary['largest_class']['name']}** ({maximum:,})", f"- Smallest class: **{summary['smallest_class']['name']}** ({minimum:,})", f"- Maximum/minimum ratio: **{summary['max_to_min_ratio']}**", f"- Integrity checks: **{checked:,}** files; **{len(corrupt)}** unreadable/corrupt", "", "Classes below 50% of the largest class are flagged as potentially underrepresented:", "", ", ".join(summary["potentially_underrepresented_classes"]) or "None"]
    (output_dir / "report.md").write_text("\n".join(report) + "\n")

    print(json.dumps(summary, indent=2))
    print("\nRepresentative image dimensions/channels:")
    for sample in samples:
        print(f"- {sample['class']}: {sample['width']}x{sample['height']}, {sample['channels']} channels ({sample['mode']})")
    print(f"\nSaved EDA outputs to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/resnet50_config.yaml")
    parser.add_argument("--integrity-limit", type=int, default=0, help="Files to validate; 0 checks all images")
    arguments = parser.parse_args()
    run(arguments.config, arguments.integrity_limit)
