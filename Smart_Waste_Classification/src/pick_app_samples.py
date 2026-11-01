"""Copy one correctly classified, unseen test photo per class to assets/samples for the app's demo buttons."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference_unified import predict
from src.train_unified import CLASSES, load_rows


def main() -> None:
    data_dir, manifest = ROOT / "data" / "unified", ROOT / "reports" / "unified_manifest.csv"
    checkpoint = str(ROOT / "models" / "mobilenet_v3_unified.pth")
    train_hashes = np.array([int(row["dhash"], 16) for row in load_rows(manifest, "train")], dtype=np.uint64)
    output = ROOT / "assets" / "samples"
    output.mkdir(parents=True, exist_ok=True)
    rows = load_rows(manifest, "test")
    rng = np.random.default_rng(42)
    for name in CLASSES:
        group = [row for row in rows if row["class"] == name]
        for index in rng.permutation(len(group)):
            row = group[int(index)]
            if np.bitwise_count(train_hashes ^ np.uint64(int(row["dhash"], 16))).min() <= 4:
                continue
            with Image.open(data_dir / row["relative_path"]) as image:
                result = predict(image, checkpoint)
            if result["best"] == name and result["certain"]:
                shutil.copy(data_dir / row["relative_path"], output / f"{name}.jpg")
                print(name, row["source"], f"{result['confidence']:.2f}")
                break


if __name__ == "__main__":
    main()
