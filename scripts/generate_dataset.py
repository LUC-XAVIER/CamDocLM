"""
generate_dataset.py
--------------------
Pillow-based synthetic document generator for CamDocLM.
Replaces the SynthText pipeline entirely.

For each YAML config in configs/:
  - loads the front/back background templates
  - loads one or more JSON field-value samples
  - draws each field's text at its configured position/font/size
  - records the exact bounding box of every drawn field (for free —
    no OCR or manual labeling needed, since we drew the text ourselves)
  - applies a realism augmentation pass (slight rotation, blur,
    brightness/contrast jitter, noise, JPEG compression)
  - saves the image(s) + a metadata JSON with bboxes and labels,
    ready to convert into LayoutLMv3 or Donut training format.

Usage:
    python generate_dataset.py --num-per-json 20
"""

import io
import os
import glob
import json
import random
import argparse
import yaml
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter

CONFIG_DIR = "configs"
OUTPUT_DIR = "data/output"


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json_values(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def draw_fields(bg_path, fields, values, font_dir):
    """Draw every field belonging to one side onto its background image.
    Returns the composited PIL image plus a list of {label, text, bbox}."""
    img = Image.open(bg_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    boxes = []

    for field in fields:
        name = field["name"]
        if name not in values:
            continue
        text = str(values[name])
        font_path = os.path.join(font_dir, field["font"])
        font = ImageFont.truetype(font_path, field["size"])
        pos = tuple(field["position"])

        draw.text(pos, text, font=font, fill=(20, 20, 20))
        bbox = draw.textbbox(pos, text, font=font)  # (x0, y0, x1, y1)

        boxes.append({"label": name, "text": text, "bbox": list(bbox)})

    return img, boxes


def augment_image(pil_img):
    """Pure Pillow/numpy realism pass: slight rotation, brightness/contrast
    jitter, blur, noise, and JPEG compression artifacts. Each effect fires
    with some probability so samples vary in how "real" they look."""
    img = pil_img

    if random.random() < 0.5:
        angle = random.uniform(-3, 3)
        img = img.rotate(angle, expand=False, fillcolor=(255, 255, 255),
                          resample=Image.BICUBIC)

    if random.random() < 0.6:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.15))

    if random.random() < 0.25:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))

    if random.random() < 0.3:
        arr = np.array(img).astype(np.int16)
        noise = np.random.normal(0, random.uniform(3, 10), arr.shape).astype(np.int16)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    if random.random() < 0.5:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(60, 95))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

    return img


def generate_sample(config, values, sample_idx, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    doc_type = config["document_type"]
    font_dir = config["fonts"]
    backgrounds = config["backgrounds"]
    fields = config["fields"]

    sides = {
        "front": backgrounds[0] if len(backgrounds) > 0 else None,
        "back": backgrounds[1] if len(backgrounds) > 1 else None,
    }

    all_boxes = {}
    for side, bg_path in sides.items():
        if not bg_path:
            continue
        side_fields = [f for f in fields if f.get("side") == side]
        if not side_fields:
            continue
        img, boxes = draw_fields(bg_path, side_fields, values, font_dir)
        img = augment_image(img)

        img_name = f"{doc_type}_{sample_idx:04d}_{side}.png"
        img.save(os.path.join(out_dir, img_name))
        all_boxes[side] = boxes

    meta_path = os.path.join(out_dir, f"{doc_type}_{sample_idx:04d}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_boxes, f, ensure_ascii=False, indent=2)


def main(num_per_json=1):
    config_files = sorted(glob.glob(os.path.join(CONFIG_DIR, "*.yaml")))
    if not config_files:
        print(f"No YAML configs found in {CONFIG_DIR}/")
        return

    for config_path in config_files:
        config = load_config(config_path)
        doc_type = config.get("document_type", os.path.basename(config_path))
        json_samples = config.get("json_samples", [])

        if not json_samples:
            print(f"[{doc_type}] no 'json_samples' listed in {config_path} — skipping")
            continue

        out_dir = os.path.join(OUTPUT_DIR, doc_type)
        idx = 0
        for json_path in json_samples:
            if not os.path.exists(json_path):
                print(f"[{doc_type}] missing JSON: {json_path}")
                continue
            values = load_json_values(json_path)
            for _ in range(num_per_json):
                generate_sample(config, values, idx, out_dir)
                idx += 1

        print(f"[{doc_type}] generated {idx} sample(s) -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic documents with Pillow")
    parser.add_argument("--num-per-json", type=int, default=1,
                         help="How many augmented variants to generate per JSON sample")
    args = parser.parse_args()
    main(num_per_json=args.num_per_json)