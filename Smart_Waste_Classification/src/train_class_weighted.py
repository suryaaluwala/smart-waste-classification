from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader

from train_mobilenet_v2 import (
    CLASSES,
    MobileNetV2Classifier,
    TrashNetDataset,
    load_split,
    make_transforms,
    run_epoch,
)

SEED = 42


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, object]:
    model.eval()
    actual: list[int] = []
    predicted: list[int] = []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            actual.extend(labels.tolist())
            predicted.extend(logits.argmax(dim=1).cpu().tolist())
    report = classification_report(
        actual,
        predicted,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    return report


def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = MobileNetV2Classifier(len(CLASSES), pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the baseline with class-weighted MobileNetV2 on validation data.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/dataset-resized"))
    parser.add_argument("--manifest", type=Path, default=Path("reports/split_manifest.csv"))
    parser.add_argument("--baseline", type=Path, default=Path("models/mobilenet_v2_baseline.pth"))
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
    train_rows = load_split(args.manifest, "train")
    val_rows = load_split(args.manifest, "validation")
    train_transform, eval_transform = make_transforms()
    train_data = TrashNetDataset(train_rows, args.data_dir, train_transform)
    val_data = TrashNetDataset(val_rows, args.data_dir, eval_transform)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=generator)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    train_counts = Counter(row["class"] for row in train_rows)
    weights = torch.tensor(
        [len(train_rows) / (len(CLASSES) * train_counts[class_name]) for class_name in CLASSES],
        dtype=torch.float32,
        device=device,
    )
    weights = weights / weights.mean()
    print(f"Device: {device}")
    print("Class weights:")
    for class_name, weight in zip(CLASSES, weights.tolist()):
        print(f"  {class_name:10} {weight:.3f} (training images: {train_counts[class_name]})")

    model = MobileNetV2Classifier(len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.backbone.classifier.parameters(), lr=1e-3)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    weighted_checkpoint = args.models_dir / "mobilenet_v2_class_weighted.pth"
    best_macro_f1 = -1.0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer)
        val_report = evaluate(model, val_loader, device)
        macro_f1 = float(val_report["macro avg"]["f1-score"])
        trash_recall = float(val_report["trash"]["recall"])
        history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy,
                        "validation_accuracy": float(val_report["accuracy"]),
                        "validation_macro_f1": macro_f1, "validation_trash_recall": trash_recall})
        print(f"Epoch {epoch}/{args.epochs} | train loss {train_loss:.4f} acc {train_accuracy:.3f} | "
              f"val acc {val_report['accuracy']:.3f} macro F1 {macro_f1:.3f} trash recall {trash_recall:.3f}")
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save({"state_dict": model.state_dict(), "classes": list(CLASSES),
                        "image_size": 224, "architecture": "mobilenet_v2", "pretrained": "ImageNet",
                        "class_weights": weights.cpu().tolist(), "best_validation_macro_f1": best_macro_f1,
                        "seed": SEED}, weighted_checkpoint)

    baseline_model = load_model(args.baseline, device)
    baseline_report = evaluate(baseline_model, val_loader, device)
    weighted_model = load_model(weighted_checkpoint, device)
    weighted_report = evaluate(weighted_model, val_loader, device)

    results = {
        "selection": "best validation macro F1; test split not used",
        "seed": SEED,
        "class_weights": {name: float(weight) for name, weight in zip(CLASSES, weights.tolist())},
        "history": history,
        "validation": {
            "baseline": baseline_report,
            "class_weighted": weighted_report,
        },
        "checkpoint": str(weighted_checkpoint),
    }
    report_path = args.reports_dir / "class_weighted_validation_comparison.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nValidation comparison (baseline -> class weighted):")
    print(f"Accuracy:  {baseline_report['accuracy']:.3f} -> {weighted_report['accuracy']:.3f}")
    print(f"Macro F1:  {baseline_report['macro avg']['f1-score']:.3f} -> {weighted_report['macro avg']['f1-score']:.3f}")
    print(f"Trash recall: {baseline_report['trash']['recall']:.3f} -> {weighted_report['trash']['recall']:.3f}")
    print(f"Weighted checkpoint: {weighted_checkpoint.resolve()}")
    print(f"Validation report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
