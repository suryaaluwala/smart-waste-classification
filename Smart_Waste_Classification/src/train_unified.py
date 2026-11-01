"""Fine-tune MobileNetV3-Large on the unified nine-class dataset.

Run with the DirectML environment for GPU training, or the main environment for CPU:
    .venv-directml\\Scripts\\python.exe src\\train_unified.py --epochs 4
The best epoch by validation macro F1 is saved after every epoch, so a stopped run keeps its model.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
from torchvision.transforms import v2

CLASSES = ("battery", "cardboard", "glass", "metal", "organic", "paper", "plastic", "textile", "trash")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
SEED = 42


class UnifiedDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], data_dir: Path, transform: v2.Compose):
        self.rows = rows
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        with Image.open(self.data_dir / row["relative_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, CLASSES.index(row["class"])


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    model = mobilenet_v3_large(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model


def make_transforms() -> tuple[v2.Compose, v2.Compose]:
    train = v2.Compose([
        v2.RandomResizedCrop(224, scale=(0.6, 1.0)),
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(15),
        v2.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=MEAN, std=STD),
        v2.RandomErasing(p=0.2),
    ])
    evaluation = v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=MEAN, std=STD),
    ])
    return train, evaluation


def load_rows(manifest: Path, split: str) -> list[dict[str, str]]:
    with manifest.open(newline="", encoding="utf-8-sig") as file:
        return [row for row in csv.DictReader(file) if row["split"] == split]


def select_device(name: str) -> torch.device:
    if name == "dml":
        import torch_directml
        return torch_directml.device(0)
    return torch.device("cpu")


def macro_f1(matrix: torch.Tensor) -> float:
    scores = []
    for index in range(matrix.size(0)):
        true_positive = matrix[index, index].item()
        denominator = matrix[index].sum().item() + matrix[:, index].sum().item()
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return sum(scores) / len(scores)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    matrix = torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.long)
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            for label, prediction in zip(labels.tolist(), predictions.tolist()):
                matrix[label, prediction] += 1
    return matrix.diag().sum().item() / matrix.sum().item(), macro_f1(matrix)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/unified"))
    parser.add_argument("--manifest", type=Path, default=Path("reports/unified_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("models/mobilenet_v3_unified.pth"))
    parser.add_argument("--history", type=Path, default=Path("reports/unified_training_history.json"))
    parser.add_argument("--device", choices=("dml", "cpu"), default="dml")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-minutes", type=float, default=30, help="Stop after the epoch that passes this budget.")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    device = select_device(args.device)
    train_rows = load_rows(args.manifest, "train")
    validation_rows = load_rows(args.manifest, "validation")
    train_transform, evaluation_transform = make_transforms()
    train_loader = DataLoader(UnifiedDataset(train_rows, args.data_dir, train_transform), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, persistent_workers=True, drop_last=True,
                              generator=torch.Generator().manual_seed(SEED))
    validation_loader = DataLoader(UnifiedDataset(validation_rows, args.data_dir, evaluation_transform),
                                   batch_size=args.batch_size, num_workers=4, persistent_workers=True)

    counts = torch.tensor([sum(row["class"] == name for row in train_rows) for name in CLASSES], dtype=torch.float32)
    class_weights = (counts.mean() / counts).sqrt()  # Softened inverse frequency: helps rare classes, keeps precision.
    print("train", len(train_rows), "validation", len(validation_rows), "weights", class_weights.tolist(), flush=True)

    model = build_model(len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)
    head = list(model.classifier.parameters())
    head_ids = {id(parameter) for parameter in head}
    backbone = [parameter for parameter in model.parameters() if id(parameter) not in head_ids]
    optimizer = torch.optim.SGD([{"params": backbone, "lr": args.lr}, {"params": head, "lr": args.lr * 5}],
                                momentum=0.9, nesterov=True, weight_decay=1e-4)
    steps = args.epochs * len(train_loader)
    base_rates = [group["lr"] for group in optimizer.param_groups]
    warmup = max(1, len(train_loader) // 4)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    history: list[dict[str, float | int]] = []
    started = time.time()
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, seen = 0.0, 0
        epoch_started = time.time()
        for images, labels in train_loader:
            factor = min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * step / steps))
            for group, rate in zip(optimizer.param_groups, base_rates):
                group["lr"] = rate * factor
            step += 1
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            value = loss.item()
            if not math.isfinite(value):
                raise SystemExit(f"Non-finite loss at step {step}")
            running_loss += value * labels.size(0)
            seen += labels.size(0)
            if step % 100 == 0:
                print(f"  step {step}/{steps} loss {running_loss / seen:.4f} "
                      f"{seen / (time.time() - epoch_started):.1f} img/s", flush=True)
        accuracy, f1 = evaluate(model, validation_loader, device)
        history.append({"epoch": epoch, "train_loss": running_loss / seen, "validation_accuracy": accuracy,
                        "validation_macro_f1": f1, "elapsed_minutes": (time.time() - started) / 60})
        print(f"Epoch {epoch}/{args.epochs} loss {running_loss / seen:.4f} val acc {accuracy:.4f} "
              f"macro F1 {f1:.4f} | {(time.time() - started) / 60:.1f} min", flush=True)
        args.history.write_text(json.dumps(history, indent=2), encoding="utf-8")
        if f1 > best_f1:
            best_f1 = f1
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save({"state_dict": state, "classes": list(CLASSES), "image_size": 224,
                        "architecture": "mobilenet_v3_large", "pretrained": "ImageNet",
                        "best_validation_macro_f1": f1, "seed": SEED}, args.output)
        if (time.time() - started) / 60 > args.max_minutes:
            print("Time budget reached, stopping.", flush=True)
            break
    print(f"Best validation macro F1 {best_f1:.4f}; checkpoint {args.output}", flush=True)


if __name__ == "__main__":
    main()
