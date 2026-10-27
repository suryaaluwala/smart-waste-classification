from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from train_class_weighted import evaluate, load_model
from train_mobilenet_v2 import (
    CLASSES,
    MobileNetV2Classifier,
    TrashNetDataset,
    load_split,
    make_transforms,
    run_epoch,
)

SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a MobileNetV2 with extra geometric augmentation for trash images.")
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
    trash_only = v2.Compose([
        v2.RandomRotation(degrees=25, fill=(255, 255, 255)),
        v2.RandomAffine(degrees=0, translate=(0.05, 0.05), fill=(255, 255, 255)),
    ])
    train_data = TrashNetDataset(train_rows, args.data_dir, train_transform,
                                 class_transforms={"trash": trash_only})
    val_data = TrashNetDataset(val_rows, args.data_dir, eval_transform)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=generator)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = MobileNetV2Classifier(len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.backbone.classifier.parameters(), lr=1e-3)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.models_dir / "mobilenet_v2_trash_augmented.pth"
    best_macro_f1 = -1.0
    history: list[dict[str, float | int]] = []
    print(f"Device: {device}; extra rotation/translation applied only to trash training images.")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device,
                                               optimizer)
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
                        "augmentation": "trash_only_rotation_25_translation_5pct",
                        "best_validation_macro_f1": best_macro_f1, "seed": SEED}, checkpoint_path)

    baseline_report = evaluate(load_model(args.baseline, device), val_loader, device)
    augmented_report = evaluate(load_model(checkpoint_path, device), val_loader, device)
    results = {
        "selection": "best validation macro F1; test split not used",
        "seed": SEED,
        "augmentation": "trash-only random rotation up to 25 degrees and translation up to 5 percent",
        "history": history,
        "validation": {"baseline": baseline_report, "trash_augmented": augmented_report},
        "checkpoint": str(checkpoint_path),
    }
    report_path = args.reports_dir / "trash_augmentation_validation_comparison.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nValidation comparison (baseline -> trash augmentation):")
    print(f"Accuracy:  {baseline_report['accuracy']:.3f} -> {augmented_report['accuracy']:.3f}")
    print(f"Macro F1:  {baseline_report['macro avg']['f1-score']:.3f} -> {augmented_report['macro avg']['f1-score']:.3f}")
    print(f"Trash recall: {baseline_report['trash']['recall']:.3f} -> {augmented_report['trash']['recall']:.3f}")
    print(f"Augmented checkpoint: {checkpoint_path.resolve()}")
    print(f"Validation report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
