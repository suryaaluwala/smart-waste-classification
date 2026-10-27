from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def main() -> None:
    parser = argparse.ArgumentParser(description="Create contact sheets for one split and one class.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/dataset-resized"))
    parser.add_argument("--manifest", type=Path, default=Path("reports/split_manifest.csv"))
    parser.add_argument("--class-name", default="trash")
    parser.add_argument("--split", default="train", choices=("train", "validation", "test"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/class_review"))
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8-sig") as file:
        files = [row["relative_path"] for row in csv.DictReader(file)
                 if row["split"] == args.split and row["class"] == args.class_name]
    files.sort()
    if not files:
        raise SystemExit(f"No samples for {args.split}/{args.class_name}")

    columns, rows_per_page = 4, 6
    thumb_w, thumb_h, label_h = 210, 165, 28
    per_page = columns * rows_per_page
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reviewing {len(files)} {args.split} images from class {args.class_name!r}.")
    for page_start in range(0, len(files), per_page):
        page_files = files[page_start:page_start + per_page]
        sheet = Image.new("RGB", (columns * thumb_w, rows_per_page * (thumb_h + label_h)), "white")
        draw = ImageDraw.Draw(sheet)
        for index, relative in enumerate(page_files):
            column, row = index % columns, index // columns
            path = args.data_dir / relative
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.thumbnail((thumb_w - 8, thumb_h - 8))
                x = column * thumb_w + (thumb_w - image.width) // 2
                y = row * (thumb_h + label_h) + (thumb_h - image.height) // 2
                sheet.paste(image, (x, y))
            draw.text((column * thumb_w + 5, row * (thumb_h + label_h) + thumb_h + 3),
                      relative, fill="black")
        output = args.output_dir / f"{args.class_name}_{args.split}_page_{page_start // per_page + 1:02}.png"
        sheet.save(output)
        print(output.resolve())


if __name__ == "__main__":
    main()
