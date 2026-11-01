from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import v2

from src.train_unified import MEAN, STD, build_model

MIN_CONTRAST = 12.0  # Pixel standard deviation below this means a blank or nearly uniform photo.

_transform = v2.Compose([
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=MEAN, std=STD),
])


@lru_cache(maxsize=1)
def load_model(checkpoint_path: str) -> tuple[torch.nn.Module, tuple[str, ...], float, float]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    classes = tuple(checkpoint["classes"])
    model = build_model(len(classes), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, classes, float(checkpoint.get("temperature", 1.0)), float(checkpoint.get("confidence_threshold", 0.6))


def preprocess(image: Image.Image) -> tuple[Image.Image, torch.Tensor]:
    """Return the 224 px crop the model really sees, plus its normalised tensor."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    crop = v2.CenterCrop(224)(v2.Resize(256)(image))
    return crop, _transform(image).unsqueeze(0)


def predict(image: Image.Image, checkpoint_path: str) -> dict:
    """Classify one image with flip averaging, calibrated scores and a Grad-CAM heatmap."""
    model, classes, temperature, threshold = load_model(checkpoint_path)
    crop, batch = preprocess(image)

    captured: dict[str, torch.Tensor] = {}
    target_layer = model.features[-1]
    handle = target_layer.register_forward_hook(lambda _m, _i, output: captured.update(activation=output))
    try:
        logits = model(batch)
    finally:
        handle.remove()
    with torch.inference_mode():
        flipped = model(batch.flip(3))
    probabilities = ((logits.detach() / temperature).softmax(1) + (flipped / temperature).softmax(1))[0] / 2
    best = int(probabilities.argmax())

    activation = captured["activation"]
    gradients = torch.autograd.grad(logits[0, best], activation)[0]
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    heatmap = torch.relu((weights * activation).sum(dim=1))[0].detach().numpy()
    heatmap = heatmap / heatmap.max() if heatmap.max() > 0 else heatmap
    blank = float(np.asarray(crop).std()) < MIN_CONTRAST

    return {
        "ranking": sorted(((classes[i], float(probabilities[i])) for i in range(len(classes))), key=lambda p: -p[1]),
        "best": classes[best], "confidence": float(probabilities[best]), "threshold": threshold,
        "certain": not blank and float(probabilities[best]) >= threshold, "crop": crop, "heatmap": heatmap,
    }


def overlay_heatmap(crop: Image.Image, heatmap: np.ndarray) -> Image.Image:
    """Blend a smoothed heatmap over the crop: yellow-red shows where the model looked."""
    mask = Image.fromarray(np.uint8(heatmap * 255)).resize(crop.size, Image.Resampling.BICUBIC)
    strength = np.asarray(mask, dtype=np.float32) / 255
    colour = np.stack([np.clip(strength * 2.2, 0, 1), np.clip(1.5 - np.abs(strength * 3 - 1.8), 0, 1) * 0.85,
                       np.zeros_like(strength)], axis=2)
    alpha = (0.65 * np.clip(strength * 1.5, 0, 1))[..., None]
    base = np.asarray(crop, dtype=np.float32) / 255
    return Image.fromarray(np.uint8(np.clip(base * (1 - alpha) + colour * alpha, 0, 1) * 255))
