from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from torchvision.transforms import v2

CLASSES = ("cardboard", "glass", "metal", "paper", "plastic", "trash")
SEED = 42


class TrashNetDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], data_dir: Path, transform: v2.Compose,
                 class_transforms: dict[str, v2.Compose] | None = None):
        self.rows = rows
        self.data_dir = data_dir
        self.transform = transform
        self.class_transforms = class_transforms or {}
        self.class_to_idx = {name: index for index, name in enumerate(CLASSES)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        image_path = self.data_dir / Path(row["relative_path"])
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            class_transform = self.class_transforms.get(row["class"])
            if class_transform is not None:
                image = class_transform(image)
            tensor = self.transform(image)
        return tensor, self.class_to_idx[row["class"]]


class MobileNetV2Classifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v2(weights=weights)
        for parameter in self.backbone.features.parameters():
            parameter.requires_grad = False
        self.backbone.classifier[1] = nn.Linear(self.backbone.last_channel, num_classes)

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep frozen BatchNorm layers in inference mode while learning the new head.
        self.backbone.features.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone.features(images)
            features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = torch.flatten(features, 1)
        return self.backbone.classifier(features)


def load_split(manifest_path: Path, split: str) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8-sig") as file:
        rows = [row for row in csv.DictReader(file) if row["split"] == split]
    if not rows:
        raise ValueError(f"No rows for split {split!r} in {manifest_path}")
    invalid = [row for row in rows if row["class"] not in CLASSES]
    if invalid:
        raise ValueError(f"Unknown class in split manifest: {invalid[0]['class']}")
    return rows


def make_transforms() -> tuple[v2.Compose, v2.Compose]:
    weights = MobileNet_V2_Weights.DEFAULT
    mean, std = weights.transforms().mean, weights.transforms().std
    train_transform = v2.Compose([
        v2.RandomResizedCrop(224, scale=(0.8, 1.0)),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])
    eval_transform = v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])
    return train_transform, eval_transform


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device,
              optimizer: torch.optim.Optimizer | None = None) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_size
    return total_loss / total, correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a frozen-feature MobileNetV2 baseline on TrashNet.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/dataset-resized"))
    parser.add_argument("--manifest", type=Path, default=Path("reports/split_manifest.csv"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")

    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("This run uses the stable main CPU environment.")

    train_rows = load_split(args.manifest, "train")
    val_rows = load_split(args.manifest, "validation")
    test_rows = load_split(args.manifest, "test")
    train_transform, eval_transform = make_transforms()
    train_data = TrashNetDataset(train_rows, args.data_dir, train_transform)
    val_data = TrashNetDataset(val_rows, args.data_dir, eval_transform)
    test_data = TrashNetDataset(test_rows, args.data_dir, eval_transform)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=generator)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = MobileNetV2Classifier(len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.backbone.classifier.parameters(), lr=1e-3)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.models_dir / "mobilenet_v2_baseline.pth"
    best_val_loss = float("inf")
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy,
                        "validation_loss": val_loss, "validation_accuracy": val_accuracy})
        print(f"Epoch {epoch}/{args.epochs} | train loss {train_loss:.4f} acc {train_accuracy:.3f} | "
              f"val loss {val_loss:.4f} acc {val_accuracy:.3f} | {time.time() - started:.1f}s")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"state_dict": model.state_dict(), "classes": list(CLASSES),
                        "image_size": 224, "architecture": "mobilenet_v2", "pretrained": "ImageNet",
                        "best_validation_loss": best_val_loss, "seed": SEED}, checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images.to(device))
            true_labels.extend(labels.tolist())
            predicted_labels.extend(logits.argmax(dim=1).cpu().tolist())

    report = classification_report(true_labels, predicted_labels, labels=list(range(len(CLASSES))),
                                  target_names=CLASSES, output_dict=True, zero_division=0)
    matrix = confusion_matrix(true_labels, predicted_labels, labels=list(range(len(CLASSES))))
    results = {"model": "MobileNetV2 frozen ImageNet features", "seed": SEED,
               "train_images": len(train_data), "validation_images": len(val_data), "test_images": len(test_data),
               "history": history, "test_classification_report": report,
               "test_confusion_matrix": matrix.tolist(), "class_order": list(CLASSES)}
    results_path = args.reports_dir / "mobilenet_v2_baseline_metrics.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Test accuracy: {report['accuracy']:.3f}; macro F1: {report['macro avg']['f1-score']:.3f}")
    print(f"Checkpoint: {checkpoint_path.resolve()}")
    print(f"Metrics: {results_path.resolve()}")


if __name__ == "__main__":
    main()
