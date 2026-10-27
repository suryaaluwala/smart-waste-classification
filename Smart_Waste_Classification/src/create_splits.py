from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

CLASSES = ("cardboard", "glass", "metal", "paper", "plastic", "trash")
SEED = 42


def allocate(n: int) -> tuple[int, int, int]:
    """Allocate counts to 70/15/15 using the largest remainder method."""
    ratios = (0.70, 0.15, 0.15)
    raw = [n * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    for index in sorted(range(3), key=lambda i: raw[i] - counts[i], reverse=True)[: n - sum(counts)]:
        counts[index] += 1
    return counts[0], counts[1], counts[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a reproducible stratified TrashNet split manifest.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/dataset-resized"))
    parser.add_argument("--output", type=Path, default=Path("reports/split_manifest.csv"))
    args = parser.parse_args()
    rng = random.Random(SEED)
    rows: list[tuple[str, str, str]] = []
    summary: dict[str, Counter[str]] = {}

    for class_name in CLASSES:
        files = sorted(p for p in (args.data_dir / class_name).iterdir() if p.suffix.lower() in {".jpg", ".jpeg"})
        if not files:
            raise SystemExit(f"No JPEG images found for class {class_name!r} in {args.data_dir / class_name}")
        rng.shuffle(files)
        n_train, n_val, n_test = allocate(len(files))
        assignments = (
            ("train", files[:n_train]),
            ("validation", files[n_train:n_train + n_val]),
            ("test", files[n_train + n_val:]),
        )
        summary[class_name] = Counter()
        for split_name, split_files in assignments:
            for path in split_files:
                rows.append((split_name, class_name, path.relative_to(args.data_dir).as_posix()))
                summary[class_name][split_name] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("split", "class", "relative_path"))
        writer.writerows(sorted(rows, key=lambda row: (row[0], row[1], row[2])))

    print(f"Seed: {SEED}; proportions: 70% train / 15% validation / 15% test")
    print(f"{'class':10} {'train':>7} {'validation':>12} {'test':>7} {'total':>7}")
    for class_name in CLASSES:
        counts = summary[class_name]
        print(f"{class_name:10} {counts['train']:7} {counts['validation']:12} {counts['test']:7} {sum(counts.values()):7}")
    print(f"Manifest: {args.output.resolve()}")
    print("Images stay in place; the CSV records which split each image belongs to.")


if __name__ == "__main__":
    main()
