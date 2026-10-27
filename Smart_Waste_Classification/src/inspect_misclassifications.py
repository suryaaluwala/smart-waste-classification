from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from train_mobilenet_v2 import CLASSES, MobileNetV2Classifier, TrashNetDataset, load_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Review MobileNetV2 mistakes on the held-out TrashNet test split.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/dataset-resized"))
    parser.add_argument("--manifest", type=Path, default=Path("reports/split_manifest.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/mobilenet_v2_baseline.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/misclassified"))
    args = parser.parse_args()

    weights_mean = (0.485, 0.456, 0.406)
    weights_std = (0.229, 0.224, 0.225)
    transform = v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=weights_mean, std=weights_std),
    ])
    rows = load_split(args.manifest, "test")
    dataset = TrashNetDataset(rows, args.data_dir, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = MobileNetV2Classifier(len(CLASSES), pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    mistakes: list[dict[str, object]] = []
    offset = 0
    with torch.inference_mode():
        for images, labels in loader:
            probabilities = model(images.to(device)).softmax(dim=1).cpu()
            predictions = probabilities.argmax(dim=1)
            for local_index, (true_index, predicted_index) in enumerate(zip(labels, predictions)):
                row = rows[offset + local_index]
                true_name = CLASSES[int(true_index)]
                predicted_name = CLASSES[int(predicted_index)]
                if true_index != predicted_index:
                    mistakes.append({
                        "true_class": true_name,
                        "predicted_class": predicted_name,
                        "predicted_confidence": float(probabilities[local_index, predicted_index]),
                        "relative_path": row["relative_path"],
                    })
            offset += len(labels)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "test_misclassifications.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("true_class", "predicted_class", "predicted_confidence", "relative_path"))
        writer.writeheader()
        for mistake in mistakes:
            writer.writerow({**mistake, "predicted_confidence": f"{mistake['predicted_confidence']:.4f}"})

    counts = Counter(str(item["true_class"]) for item in mistakes)
    print(f"Test images: {len(rows)}; misclassified: {len(mistakes)}")
    print("Mistakes by true class:")
    for class_name in CLASSES:
        print(f"  {class_name:10} {counts[class_name]:3}")
    print(f"CSV: {csv_path.resolve()}")

    columns, rows_per_page = 4, 4
    thumb_w, thumb_h, caption_h = 210, 165, 48
    page_capacity = columns * rows_per_page
    sorted_mistakes = sorted(mistakes, key=lambda item: (CLASSES.index(str(item["true_class"])), str(item["relative_path"])))
    for page_index in range(0, len(sorted_mistakes), page_capacity):
        page_items = sorted_mistakes[page_index:page_index + page_capacity]
        sheet = Image.new("RGB", (columns * thumb_w, rows_per_page * (thumb_h + caption_h)), "white")
        draw = ImageDraw.Draw(sheet)
        for local_index, item in enumerate(page_items):
            column, row_index = local_index % columns, local_index // columns
            path = args.data_dir / Path(str(item["relative_path"]))
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.thumbnail((thumb_w - 8, thumb_h - 8))
                x = column * thumb_w + (thumb_w - image.width) // 2
                y = row_index * (thumb_h + caption_h) + (thumb_h - image.height) // 2
                sheet.paste(image, (x, y))
            label = f"True: {item['true_class']}  ->  Pred: {item['predicted_class']}\n"
            label += f"Confidence: {float(item['predicted_confidence']):.0%}\n{Path(str(item['relative_path'])).name}"
            draw.text((column * thumb_w + 5, row_index * (thumb_h + caption_h) + thumb_h + 2), label, fill="black")
        image_path = args.output_dir / f"test_mistakes_page_{page_index // page_capacity + 1:02}.png"
        sheet.save(image_path)
        print(f"Contact sheet: {image_path.resolve()}")

    trash_mistakes = [item for item in mistakes if item["true_class"] == "trash"]
    print("Trash mistakes:")
    for item in trash_mistakes:
        print(f"  {item['relative_path']} -> {item['predicted_class']} ({float(item['predicted_confidence']):.1%})")


if __name__ == "__main__":
    main()
