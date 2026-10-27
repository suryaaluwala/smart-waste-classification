from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import v2

from src.train_mobilenet_v2 import CLASSES, MobileNetV2Classifier


def image_transform() -> v2.Compose:
    return v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


@lru_cache(maxsize=2)
def load_checkpoint(checkpoint_path: str) -> tuple[MobileNetV2Classifier, tuple[str, ...]]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    classes = tuple(checkpoint["classes"])
    model = MobileNetV2Classifier(len(classes), pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, classes


def predict_image(image: Image.Image, checkpoint_path: str = "models/mobilenet_v2_baseline.pth",
                  top_k: int = 3) -> list[dict[str, float | str]]:
    model, classes = load_checkpoint(checkpoint_path)
    device = next(model.parameters()).device
    tensor = image.convert("RGB")
    batch = image_transform()(tensor).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(batch).softmax(dim=1)[0]
    count = min(top_k, len(classes))
    values, indices = probabilities.topk(count)
    return [
        {"class": classes[int(index)], "confidence": float(value)}
        for value, index in zip(values.cpu(), indices.cpu())
    ]
