from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

CLASSES = ("cardboard", "glass", "metal", "paper", "plastic", "trash")


def inspect(data_dir: Path, report_dir: Path, samples_per_class: int) -> int:
    thumb_w, thumb_h, label_h = 240, 180, 28
    sheet = Image.new("RGB", (3 * thumb_w, len(CLASSES) * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    total = 0
    bad: list[tuple[Path, str]] = []

    for row, class_name in enumerate(CLASSES):
        class_dir = data_dir / class_name
        if not class_dir.is_dir():
            print(f"MISSING CLASS FOLDER: {class_dir}")
            continue
        files = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"})
        print(f"{class_name:10} {len(files):4} images")
        total += len(files)
        for path in files:
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception as error:
                bad.append((path, str(error)))

        for col, path in enumerate(files[:samples_per_class]):
            try:
                with Image.open(path) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    image.thumbnail((thumb_w - 8, thumb_h - 8))
                    x = col * thumb_w + (thumb_w - image.width) // 2
                    y = row * (thumb_h + label_h) + (thumb_h - image.height) // 2
                    sheet.paste(image, (x, y))
                draw.text(
                    (col * thumb_w + 6, row * (thumb_h + label_h) + thumb_h + 4),
                    f"{class_name} / {path.name}",
                    fill="black",
                )
            except Exception as error:
                bad.append((path, str(error)))

    report_dir.mkdir(parents=True, exist_ok=True)
    output = report_dir / "data_sample_review.png"
    sheet.save(output)
    print(f"Total JPEG images: {total}")
    print(f"Unreadable JPEG images: {len(bad)}")
    for path, error in bad:
        print(f"  BAD: {path}: {error}")
    print(f"Sample sheet: {output.resolve()}")
    return 1 if bad else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Count TrashNet images and build a labeled sample sheet.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/dataset-resized"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--samples-per-class", type=int, default=3)
    args = parser.parse_args()
    raise SystemExit(inspect(args.data_dir, args.report_dir, args.samples_per_class))


if __name__ == "__main__":
    main()
