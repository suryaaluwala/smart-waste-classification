"""Robustness checks for the app's inference path. Run: python tests/test_inference.py"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference_unified import overlay_heatmap, predict

CHECKPOINT = str(ROOT / "models" / "mobilenet_v3_unified.pth")
rng = np.random.default_rng(0)
noise = Image.fromarray(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8))
cases = {
    "noise": noise,
    "black": Image.new("RGB", (300, 300)),
    "white": Image.new("RGB", (300, 300), "white"),
    "grayscale": noise.convert("L"),
    "rgba": noise.convert("RGBA"),
    "palette": noise.convert("P"),
    "tiny 8x8": noise.resize((8, 8)),
    "panorama": noise.resize((3000, 200)),
    "huge": noise.resize((4000, 3000)),
}

for name, image in cases.items():
    started = time.time()
    result = predict(image, CHECKPOINT)
    assert abs(sum(score for _, score in result["ranking"]) - 1) < 1e-3
    assert overlay_heatmap(result["crop"], result["heatmap"]).size == (224, 224)
    print(f"{name:10} {result['best']:10} {result['confidence']:.2f} certain={result['certain']} "
          f"{(time.time() - started) * 1000:.0f} ms")
print("all passed")
