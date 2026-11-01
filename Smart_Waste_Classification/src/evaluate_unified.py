"""Evaluate the unified model once on the held-out test split and compare it with the old baseline.

Temperature and the "not sure" threshold are fitted on validation data only.
The baseline comparison uses the six TrashNet classes, with the new model's three extra classes masked out.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.train_mobilenet_v2 import CLASSES as OLD_CLASSES, MobileNetV2Classifier
from src.train_unified import CLASSES, UnifiedDataset, build_model, load_rows, make_transforms


def logits_for(model: torch.nn.Module, rows: list[dict[str, str]], data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    _, evaluation_transform = make_transforms()
    loader = DataLoader(UnifiedDataset(rows, data_dir, evaluation_transform), batch_size=64, num_workers=6)
    outputs, targets = [], []
    with torch.inference_mode():
        for images, labels in loader:
            outputs.append(model(images).numpy())
            targets.append(labels.numpy())
    return np.concatenate(outputs), np.concatenate(targets)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    logits_tensor, labels_tensor = torch.from_numpy(logits), torch.from_numpy(labels)
    candidates = np.arange(0.4, 2.51, 0.02)
    losses = [torch.nn.functional.cross_entropy(logits_tensor / float(t), labels_tensor).item() for t in candidates]
    return float(candidates[int(np.argmin(losses))])


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    return torch.from_numpy(logits / temperature).softmax(dim=1).numpy()


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    confidence, prediction = probabilities.max(axis=1), probabilities.argmax(axis=1)
    error = 0.0
    for low in np.arange(0, 1, 1 / bins):
        mask = (confidence > low) & (confidence <= low + 1 / bins)
        if mask.any():
            error += mask.mean() * abs((prediction[mask] == labels[mask]).mean() - confidence[mask].mean())
    return float(error)


def threshold_for_precision(probabilities: np.ndarray, labels: np.ndarray, target: float = 0.97) -> float:
    confidence, correct = probabilities.max(axis=1), probabilities.argmax(axis=1) == labels
    for threshold in np.arange(0.3, 0.96, 0.01):
        mask = confidence >= threshold
        if mask.any() and correct[mask].mean() >= target:
            return float(threshold)
    return 0.9


def load_baseline(path: Path) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = MobileNetV2Classifier(len(OLD_CLASSES), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def save_confusion_figure(matrix: np.ndarray, path: Path) -> None:
    normalised = matrix / matrix.sum(axis=1, keepdims=True)
    figure, axis = plt.subplots(figsize=(7.2, 6.2), dpi=150)
    axis.imshow(normalised, cmap="Greens", vmin=0, vmax=1)
    axis.set_xticks(range(len(CLASSES)), CLASSES, rotation=40, ha="right")
    axis.set_yticks(range(len(CLASSES)), CLASSES)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title("Test confusion matrix (share of each actual class)")
    for row in range(len(CLASSES)):
        for column in range(len(CLASSES)):
            value = normalised[row, column]
            if value >= 0.005:
                axis.text(column, row, f"{value:.0%}", ha="center", va="center", fontsize=8,
                          color="white" if value > 0.55 else "#17301f")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/unified"))
    parser.add_argument("--manifest", type=Path, default=Path("reports/unified_manifest.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/mobilenet_v3_unified.pth"))
    parser.add_argument("--baseline", type=Path, default=Path("models/mobilenet_v2_baseline.pth"))
    parser.add_argument("--output", type=Path, default=Path("reports/unified_metrics.json"))
    parser.add_argument("--figure", type=Path, default=Path("reports/unified_confusion_matrix.png"))
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = build_model(len(CLASSES), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    validation_rows, test_rows = load_rows(args.manifest, "validation"), load_rows(args.manifest, "test")
    validation_logits, validation_labels = logits_for(model, validation_rows, args.data_dir)
    temperature = fit_temperature(validation_logits, validation_labels)
    validation_probabilities = softmax(validation_logits, temperature)
    threshold = threshold_for_precision(validation_probabilities, validation_labels)

    test_logits, test_labels = logits_for(model, test_rows, args.data_dir)
    probabilities = softmax(test_logits, temperature)
    predictions = probabilities.argmax(axis=1)
    report = classification_report(test_labels, predictions, labels=list(range(len(CLASSES))), target_names=CLASSES,
                                   output_dict=True, zero_division=0)
    matrix = confusion_matrix(test_labels, predictions, labels=list(range(len(CLASSES))))
    save_confusion_figure(matrix, args.figure)

    sources = np.array([row["source"] for row in test_rows])
    by_source = {}
    for source in sorted(set(sources)):
        mask = sources == source
        by_source[source] = {"images": int(mask.sum()), "accuracy": float((predictions[mask] == test_labels[mask]).mean())}

    # Stricter headline: drop test images that are near-copies of a training image (dHash distance <= 4).
    train_hashes = np.array([int(row["dhash"], 16) for row in load_rows(args.manifest, "train")], dtype=np.uint64)
    unseen = np.array([np.bitwise_count(train_hashes ^ np.uint64(int(row["dhash"], 16))).min() > 4
                       for row in test_rows])
    unseen_report = classification_report(test_labels[unseen], predictions[unseen], labels=list(range(len(CLASSES))),
                                          target_names=CLASSES, output_dict=True, zero_division=0)
    leak_free = {"images": int(unseen.sum()), "accuracy": unseen_report["accuracy"],
                 "macro_f1": unseen_report["macro avg"]["f1-score"],
                 "per_class": {name: unseen_report[name] for name in CLASSES}, "by_source": {}}
    for source in sorted(set(sources)):
        mask = (sources == source) & unseen
        leak_free["by_source"][source] = {"images": int(mask.sum()),
                                          "accuracy": float((predictions[mask] == test_labels[mask]).mean())}

    confident = probabilities.max(axis=1) >= threshold
    selective = {"threshold": threshold, "coverage": float(confident.mean()),
                 "accuracy_when_confident": float((predictions[confident] == test_labels[confident]).mean())}

    # Same images, same six labels, for both models.
    shared_indices = [CLASSES.index(name) for name in OLD_CLASSES]
    shared_rows = [row for row in test_rows if row["class"] in OLD_CLASSES]
    shared_mask = np.array([row["class"] in OLD_CLASSES for row in test_rows])
    shared_truth = np.array([OLD_CLASSES.index(row["class"]) for row in shared_rows])
    shared_sources = sources[shared_mask]
    shared_unseen = unseen[shared_mask]
    new_shared = test_logits[shared_mask][:, shared_indices].argmax(axis=1)
    baseline = load_baseline(args.baseline)
    old_logits, _ = logits_for(baseline, shared_rows, args.data_dir)
    old_shared = old_logits.argmax(axis=1)
    comparison = {}
    for source in ["all", *sorted(set(shared_sources))]:
        mask = shared_unseen if source == "all" else (shared_sources == source) & shared_unseen
        comparison[source] = {"images": int(mask.sum()),
                              "old_baseline_accuracy": float((old_shared[mask] == shared_truth[mask]).mean()),
                              "new_model_accuracy": float((new_shared[mask] == shared_truth[mask]).mean())}

    results = {
        "model": checkpoint["architecture"], "classes": list(CLASSES), "temperature": temperature,
        "best_validation_macro_f1": checkpoint["best_validation_macro_f1"],
        "test_images": len(test_rows), "test_accuracy": report["accuracy"],
        "test_macro_f1": report["macro avg"]["f1-score"], "per_class": {name: report[name] for name in CLASSES},
        "calibration_error_before": expected_calibration_error(softmax(test_logits), test_labels),
        "calibration_error_after": expected_calibration_error(probabilities, test_labels),
        "selective_prediction": selective, "by_source": by_source, "leak_free_test": leak_free,
        "six_class_comparison_leak_free": comparison,
        "confusion_matrix": matrix.tolist(),
    }
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    checkpoint["temperature"] = temperature
    checkpoint["confidence_threshold"] = threshold
    torch.save(checkpoint, args.checkpoint)
    print(json.dumps({key: value for key, value in results.items()
                      if key not in {"per_class", "confusion_matrix", "classes"}}, indent=2))
    print({name: round(report[name]["f1-score"], 3) for name in CLASSES})


if __name__ == "__main__":
    main()
