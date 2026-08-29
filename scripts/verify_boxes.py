"""
verify_boxes.py
-----------------
Draws the recorded bboxes from metadata/*.json back onto their matching
images/*.png, so you can visually confirm the boxes actually line up
with the rendered text before trusting the dataset for training.

Usage:
    python verify_boxes.py --doc-dir data/output/Cameroon_NIC_v1 --num-samples 8
"""

import os
import json
import glob
import random
import argparse
from PIL import Image, ImageDraw, ImageFont

COLORS = [
    (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (128, 128, 0),
]


def label_color(label, label_list):
    return COLORS[label_list.index(label) % len(COLORS)]


def draw_preview(image_path, meta_path, out_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    side = "front" if "_front" in os.path.basename(image_path) else "back"
    words = meta.get(side, [])
    labels = sorted(set(w["label"] for w in words))

    for w in words:
        x0, y0, x1, y1 = w["bbox"]
        color = label_color(w["label"], labels)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        draw.text((x0, max(0, y0 - 12)), f'{w["label"]}[{w["word_index"]}]',
                   font=font, fill=color)

    img.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Overlay recorded bboxes on generated images for visual QC")
    parser.add_argument("--doc-dir", required=True, help="e.g. data/output/Cameroon_NIC_v1")
    parser.add_argument("--num-samples", type=int, default=8)
    args = parser.parse_args()

    images_dir = os.path.join(args.doc_dir, "images")
    metadata_dir = os.path.join(args.doc_dir, "metadata")
    out_dir = os.path.join(args.doc_dir, "_bbox_preview")
    os.makedirs(out_dir, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    if not image_paths:
        print(f"No images found in {images_dir}")
        return

    sample = random.sample(image_paths, min(args.num_samples, len(image_paths)))
    for image_path in sample:
        base = os.path.basename(image_path)
        # e.g. Cameroon_NIC_v1_0042_front.png -> Cameroon_NIC_v1_0042_meta.json
        sample_id = base.rsplit("_", 1)[0]
        meta_path = os.path.join(metadata_dir, f"{sample_id}_meta.json")
        if not os.path.exists(meta_path):
            print(f"  [skip] no metadata for {base}")
            continue
        out_path = os.path.join(out_dir, f"preview_{base}")
        draw_preview(image_path, meta_path, out_path)
        print(f"  wrote {out_path}")

    print(f"\nCheck {out_dir} — boxes should sit tightly around each word, "
          f"and should still line up correctly even on heavily zoomed/panned samples.")


if __name__ == "__main__":
    main()