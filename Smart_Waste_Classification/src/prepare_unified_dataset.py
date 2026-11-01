"""Merge TrashNet, RealWaste and Garbage Classification v2 into one nine-class dataset.

Every image is re-encoded to a 256 px short side JPEG under data/unified/. Exact visual
duplicates (same 64-bit dHash) are dropped so that one photo cannot appear in two splits.
The original TrashNet split is kept, so old and new models share the same 377 test images.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image, ImageOps

SEED = 42
CLASSES = ("battery", "cardboard", "glass", "metal", "organic", "paper", "plastic", "textile", "trash")
LABEL_MAP = {
    "battery": "battery", "cardboard": "cardboard", "glass": "glass", "metal": "metal", "paper": "paper",
    "plastic": "plastic", "trash": "trash", "biological": "organic", "clothes": "textile", "shoes": "textile",
    "food organics": "organic", "vegetation": "organic", "textile trash": "textile",
    "miscellaneous trash": "trash",
}
TEXTILE_CAP = 3000


def dhash(image: Image.Image) -> str:
    small = image.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    pixels = small.tobytes()
    bits = 0
    for row in range(8):
        for column in range(8):
            bits = (bits << 1) | (pixels[row * 9 + column] > pixels[row * 9 + column + 1])
    return f"{bits:016x}"


def process(job: tuple[str, str, str, bytes, Path]) -> dict[str, str] | None:
    source, class_name, stem, payload, output_dir = job
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except Exception:
        return None
    if min(image.size) < 32:
        return None
    scale = 256 / min(image.size)
    image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.BILINEAR)
    relative = f"{class_name}/{source}_{stem}.jpg"
    image.save(output_dir / relative, quality=90)
    return {"source": source, "class": class_name, "relative_path": relative, "dhash": dhash(image)}


def trashnet_jobs(data_dir: Path, output_dir: Path):
    for path in sorted(data_dir.glob("*/*.jpg")):
        yield "trashnet", path.parent.name, path.stem, path.read_bytes(), output_dir


def parquet_jobs(folder: Path, source: str, output_dir: Path):
    """Use the label column when a shard has one; the RealWaste copy keeps it in the file name instead."""
    index = 0  # File names repeat inside a source, so a running number keeps outputs unique.
    for shard in sorted(folder.glob("*.parquet")):
        table = pq.ParquetFile(shard)
        names = None
        if "label" in table.schema_arrow.names:
            names = json.loads(table.schema_arrow.metadata[b"huggingface"])["info"]["features"]["label"]["names"]
        for batch in table.iter_batches(batch_size=128):
            for row in batch.to_pylist():
                index += 1
                stem = Path(row["image"]["path"]).stem
                raw_label = names[row["label"]] if names else stem.split("_", 1)[0].lower()
                yield source, LABEL_MAP[raw_label], f"{index:05d}", row["image"]["bytes"], output_dir


def allocate(n: int) -> tuple[int, int]:
    return round(n * 0.70), round(n * 0.15)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trashnet-dir", type=Path, default=Path("data/dataset-resized"))
    parser.add_argument("--trashnet-manifest", type=Path, default=Path("reports/split_manifest.csv"))
    parser.add_argument("--realwaste-dir", type=Path, default=Path("data/external/realwaste_hf"))
    parser.add_argument("--garbage-dir", type=Path, default=Path("data/external/garbage_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/unified"))
    parser.add_argument("--manifest", type=Path, default=Path("reports/unified_manifest.csv"))
    args = parser.parse_args()

    for class_name in CLASSES:
        (args.output_dir / class_name).mkdir(parents=True, exist_ok=True)

    with args.trashnet_manifest.open(newline="", encoding="utf-8-sig") as file:
        original_split = {Path(row["relative_path"]).stem: row["split"] for row in csv.DictReader(file)}

    def all_jobs():
        yield from trashnet_jobs(args.trashnet_dir, args.output_dir)
        yield from parquet_jobs(args.realwaste_dir, "realwaste", args.output_dir)
        yield from parquet_jobs(args.garbage_dir, "garbagev2", args.output_dir)

    rows: list[dict[str, str]] = []
    unreadable = 0
    with ProcessPoolExecutor(max_workers=10) as pool:
        for result in pool.map(process, all_jobs(), chunksize=32):
            if result is None:
                unreadable += 1
            else:
                rows.append(result)
            if (len(rows) + unreadable) % 3000 == 0:
                print(f"processed {len(rows) + unreadable}", flush=True)
    print(f"readable {len(rows)}, unreadable {unreadable}")

    # TrashNet is listed first, so its copy (and split) wins when a photo also exists elsewhere.
    rows.sort(key=lambda row: (row["source"] != "trashnet", row["relative_path"]))
    seen: dict[str, dict[str, str]] = {}
    kept: list[dict[str, str]] = []
    dropped: Counter[str] = Counter()
    conflicting = 0
    for row in rows:
        first = seen.get(row["dhash"])
        if first is None or row["source"] == "trashnet":
            seen.setdefault(row["dhash"], row)
            kept.append(row)
            continue
        dropped[row["source"]] += 1
        conflicting += first["class"] != row["class"]
        (args.output_dir / row["relative_path"]).unlink(missing_ok=True)
    print(f"duplicates dropped {dict(dropped)}; of which label conflicts {conflicting}")

    rng = random.Random(SEED)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in kept:
        if row["source"] == "trashnet":
            row["split"] = original_split[Path(row["relative_path"]).stem.removeprefix("trashnet_")]
        else:
            grouped[(row["class"], row["source"])].append(row)
    for (class_name, source), group in sorted(grouped.items()):
        rng.shuffle(group)
        if class_name == "textile" and source == "garbagev2":
            for row in group[TEXTILE_CAP:]:
                (args.output_dir / row["relative_path"]).unlink(missing_ok=True)
                kept.remove(row)
            group = group[:TEXTILE_CAP]
        n_train, n_val = allocate(len(group))
        for index, row in enumerate(group):
            row["split"] = "train" if index < n_train else "validation" if index < n_train + n_val else "test"

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("split", "class", "source", "relative_path", "dhash"))
        writer.writeheader()
        writer.writerows(sorted(kept, key=lambda row: (row["split"], row["class"], row["relative_path"])))

    table = Counter((row["class"], row["split"]) for row in kept)
    print(f"{'class':10} {'train':>7} {'validation':>11} {'test':>6}")
    for class_name in CLASSES:
        print(f"{class_name:10} {table[(class_name, 'train')]:7} {table[(class_name, 'validation')]:11} "
              f"{table[(class_name, 'test')]:6}")
    print("by source", dict(Counter((row["source"], row["split"]) for row in kept)))
    print(f"total kept {len(kept)}; manifest {args.manifest}")


if __name__ == "__main__":
    main()
