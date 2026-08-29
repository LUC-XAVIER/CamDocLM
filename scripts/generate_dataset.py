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
import math
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


def _load_field_font(font_dir, field, scale=1):
    """Load a font, applying a weight instance if the field asks for one
    and the font file is a variable font. `scale` renders at a higher
    pixel size (used for supersampling). `weight` can be a name ("bold",
    "black") to match a named instance, or a number (100-900) to set the
    weight axis to an exact value."""
    font_path = os.path.join(font_dir, field["font"])
    font = ImageFont.truetype(font_path, int(round(field["size"] * scale)))

    weight = field.get("weight")
    if weight and hasattr(font, "get_variation_names"):
        try:
            if isinstance(weight, (int, float)):
                axes = font.get_variation_axes()
                values = [ax["default"] for ax in axes]
                for i, ax in enumerate(axes):
                    axis_name = ax["name"]
                    axis_name = axis_name.decode() if isinstance(axis_name, bytes) else axis_name
                    if "wght" in axis_name.lower() or "weight" in axis_name.lower():
                        values[i] = max(ax["minimum"], min(ax["maximum"], weight))
                font.set_variation_by_axes(values)
            else:
                names = [n.decode() if isinstance(n, bytes) else n
                         for n in font.get_variation_names()]
                match = next((n for n in names if str(weight).lower() in n.lower()), None)
                if match:
                    font.set_variation_by_name(match)
        except OSError:
            pass  # not a variable font — ignore and use as-is
    return font


SUPERSAMPLE = 4  # render text at 4x then downsample with anti-aliasing —
                  # this is what makes sub-pixel letter_spacing (e.g. 0.5)
                  # actually visible: 0.5px at native size becomes a full
                  # 2px at 4x, gets drawn as a real pixel, then blends back
                  # down smoothly instead of being rounded away to nothing

LETTER_SPACING_UNIT = 3.0  # global sensitivity multiplier: every field's
                            # letter_spacing value is multiplied by this
                            # before rendering. Bump this up if your usual
                            # values (0.2-0.5) look too subtle across the
                            # board; tune per-field values as before, this
                            # just scales all of them together.


def _render_field_text(font_dir, field, text, fill=(20, 20, 20), scale=SUPERSAMPLE):
    """Render `text` for this field onto a tight transparent RGBA patch at
    `scale`x resolution (letter_spacing/stroke_width applied at that same
    scale), then downsample. Returns (patch, width, height, word_boxes) —
    word_boxes is a list of {"text": word, "bbox": [x0,y0,x1,y1]} at
    NATIVE resolution, in coordinates local to the patch (add the field's
    own position to get absolute coordinates). This gives exact per-word
    boxes for free, since we're the ones drawing the text — no OCR or
    proportional-splitting approximation needed."""
    spacing = field.get("letter_spacing", 0) * LETTER_SPACING_UNIT * scale
    stroke_width = int(round(field.get("stroke_width", 0) * scale))
    big_font = _load_field_font(font_dir, field, scale=scale)

    probe = Image.new("RGBA", (4, 4))
    pdraw = ImageDraw.Draw(probe)
    widths = [pdraw.textlength(ch, font=big_font) for ch in text] if text else [0]
    total_w = sum(widths) + spacing * max(0, len(text) - 1) + 2 * stroke_width
    ref_bbox = (pdraw.textbbox((0, 0), text, font=big_font, stroke_width=stroke_width)
                if text else (0, 0, 0, 0))
    total_h = (ref_bbox[3] - ref_bbox[1]) + 2 * stroke_width

    canvas = Image.new("RGBA", (max(1, int(total_w) + 4), max(1, int(total_h) + 4)), (0, 0, 0, 0))
    cdraw = ImageDraw.Draw(canvas)
    x = 2.0
    y = 2 - ref_bbox[1]
    canvas_word_boxes = []  # (word_text, x0, x1) in big-scale canvas coords
    current_word = []
    word_start_x = None
    for ch in text:
        if ch == " ":
            if current_word:
                canvas_word_boxes.append(("".join(current_word), word_start_x, x))
                current_word = []
                word_start_x = None
            x += pdraw.textlength(ch, font=big_font) + spacing
            continue
        if word_start_x is None:
            word_start_x = x
        cdraw.text((x, y), ch, font=big_font, fill=fill + (255,),
                    stroke_width=stroke_width, stroke_fill=fill + (255,))
        current_word.append(ch)
        x += pdraw.textlength(ch, font=big_font) + spacing
    if current_word:
        canvas_word_boxes.append(("".join(current_word), word_start_x, x))

    final_w = max(1, canvas.width // scale)
    final_h = max(1, canvas.height // scale)
    patch = canvas.resize((final_w, final_h), Image.LANCZOS)

    native_stroke = stroke_width / scale  # stroke bleeds outward in every
                                            # direction by ~stroke_width; the
                                            # plain advance-width box doesn't
                                            # account for that on its own
    word_boxes = [
        {"text": w, "bbox": [
            max(0, wx0 / scale - native_stroke), 0,
            min(final_w, wx1 / scale + native_stroke), final_h,
        ]}
        for (w, wx0, wx1) in canvas_word_boxes
    ]
    return patch, final_w, final_h, word_boxes


def draw_fields(bg_path, fields, values, font_dir):
    """Draw every field belonging to one side onto its background image.
    Text is rendered supersampled (see _render_field_text) so fractional
    letter_spacing is actually visible, then composited with alpha for
    clean anti-aliased edges. Assumes the background is already a clean
    (blank-field) template — no erase step. Returns the composited PIL
    image plus a list of one entry per WORD: {label, text, bbox,
    word_index} — word_index is 0 for the first word of a field, 1 for
    the second, etc., so a later conversion step can assign B-<label> to
    word_index 0 and I-<label> to the rest without extra bookkeeping."""
    img = Image.open(bg_path).convert("RGB")
    boxes = []

    for field in fields:
        name = field["name"]
        if name not in values:
            continue
        text = str(values[name])
        pos = tuple(field["position"])

        img_w, img_h = img.size
        if not (0 <= pos[0] < img_w and 0 <= pos[1] < img_h):
            print(f"  [warn] field '{name}' position {pos} is outside "
                  f"this background's bounds {img.size} — check its "
                  f"coordinates against the actual image size")

        patch, pw, ph, word_boxes = _render_field_text(font_dir, field, text)
        paste_pos = (int(round(pos[0])), int(round(pos[1])))
        img.paste(patch, paste_pos, patch)  # patch's own alpha as mask

        for i, w in enumerate(word_boxes):
            wx0, wy0, wx1, wy1 = w["bbox"]
            boxes.append({
                "label": name,
                "text": w["text"],
                "bbox": [pos[0] + wx0, pos[1] + wy0, pos[0] + wx1, pos[1] + wy1],
                "word_index": i,
            })

    return img, boxes


def _rotate_boxes(boxes, angle_degrees, center):
    """Rotate each box's corners by the same transform PIL's Image.rotate
    applies to pixels, then re-enclose in an axis-aligned box. Keeps
    metadata accurate when augment_image rotates the image."""
    if angle_degrees == 0:
        return boxes
    theta = math.radians(angle_degrees)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    cx, cy = center

    def rotate_point(x, y):
        dx, dy = x - cx, y - cy
        return (cx + dx * cos_t - dy * sin_t,
                cy + dx * sin_t + dy * cos_t)

    rotated = []
    for b in boxes:
        x0, y0, x1, y1 = b["bbox"]
        corners = [rotate_point(x0, y0), rotate_point(x1, y0),
                   rotate_point(x1, y1), rotate_point(x0, y1)]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        rotated.append({
            **b,
            "bbox": [min(xs), min(ys), max(xs), max(ys)],
        })
    return rotated


def augment_image(pil_img, boxes=None):
    """Pure Pillow/numpy realism pass: slight rotation, brightness/contrast
    jitter, blur, noise, and JPEG compression artifacts. Each effect fires
    with some probability so samples vary in how "real" they look. If
    `boxes` is given, rotation is applied to them too (same transform as
    the pixels), so labels stay accurate. Returns (img, boxes) if boxes
    was given, else just img."""
    img = pil_img

    if random.random() < 0.5:
        angle = random.uniform(-3, 3)
        img = img.rotate(angle, expand=False, fillcolor=(255, 255, 255),
                          resample=Image.BICUBIC)
        if boxes is not None:
            center = (img.size[0] / 2, img.size[1] / 2)
            boxes = _rotate_boxes(boxes, -angle, center)

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

    return (img, boxes) if boxes is not None else img


def add_specimen_watermark(img, text="SPECIMEN — SYNTHETIC DATA"):
    """Diagonal low-opacity watermark. Standard practice for synthetic ID
    datasets (keeps generated cards unambiguously non-genuine at a glance)
    and doubles as noise the model has to learn to see past when reading
    fields. Set add_watermark=False in generate_sample()/main() to skip."""
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    step = 160
    for y in range(0, img.size[1] + step, step):
        for x in range(0, img.size[0] + step, step):
            draw.text((x, y), text, font=font, fill=(200, 30, 30, 70))
    rotated = overlay.rotate(30, expand=False)
    return Image.alpha_composite(img.convert("RGBA"), rotated).convert("RGB")


FRAME_CANVAS_EXPAND = 1.8          # final canvas is this much larger than
                                    # the card's own template size — gives
                                    # room to pan the card left/right/up/down
FRAME_SCALE_RANGE = (0.35, 2.0)    # random zoom applied to the card before
                                    # placing it on the canvas. Below 1.0 =
                                    # card looks farther away (more backdrop
                                    # visible); above 1.0 can exceed the
                                    # canvas entirely — mimics a too-close
                                    # photo with edges cropped off
FRAME_BACKDROP_COLOR = (210, 205, 195)  # neutral area around the card


def randomize_framing(img, boxes, canvas_expand=FRAME_CANVAS_EXPAND,
                       scale_range=FRAME_SCALE_RANGE,
                       backdrop_color=FRAME_BACKDROP_COLOR):
    """Places the fully-rendered card onto a larger canvas at a random
    scale and offset, so the card isn't always centered/full-frame like
    the raw template — mimics a photo taken from varying distance and
    position. When the scaled card is bigger than the canvas, it's
    allowed to hang off any edge (cropped), same as a photo taken too
    close. Remaps every field's bbox to match, clipping to what's
    actually visible and dropping fields that end up entirely out of
    frame — metadata never claims a field is there when it's been
    cropped out. Field positions on the card itself are untouched —
    this only changes where the card sits in the frame."""
    card_w, card_h = img.size
    canvas_w = int(card_w * canvas_expand)
    canvas_h = int(card_h * canvas_expand)

    scale = random.uniform(*scale_range)
    scaled_w = max(1, int(card_w * scale))
    scaled_h = max(1, int(card_h * scale))
    scaled_img = img.resize((scaled_w, scaled_h), Image.LANCZOS)

    lo_x, hi_x = sorted((0, canvas_w - scaled_w))
    lo_y, hi_y = sorted((0, canvas_h - scaled_h))
    offset_x = random.randint(lo_x, hi_x)
    offset_y = random.randint(lo_y, hi_y)

    canvas = Image.new("RGB", (canvas_w, canvas_h), backdrop_color)
    canvas.paste(scaled_img, (offset_x, offset_y))

    remapped_boxes = []
    for b in boxes:
        x0, y0, x1, y1 = b["bbox"]
        nx0, ny0 = x0 * scale + offset_x, y0 * scale + offset_y
        nx1, ny1 = x1 * scale + offset_x, y1 * scale + offset_y
        cx0, cy0 = max(0, nx0), max(0, ny0)
        cx1, cy1 = min(canvas_w, nx1), min(canvas_h, ny1)
        if cx1 - cx0 < 2 or cy1 - cy0 < 2:
            continue  # cropped out of frame — don't mislabel it as visible
        remapped_boxes.append({
            "label": b["label"],
            "text": b["text"],
            "bbox": [cx0, cy0, cx1, cy1],
            "word_index": b["word_index"],
        })

    return canvas, remapped_boxes


def generate_sample(config, values, sample_idx, out_dir, add_watermark=True):
    images_dir = os.path.join(out_dir, "images")
    metadata_dir = os.path.join(out_dir, "metadata")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
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
        img, boxes = augment_image(img, boxes)
        if add_watermark:
            img = add_specimen_watermark(img)
        img, boxes = randomize_framing(img, boxes)

        img_name = f"{doc_type}_{sample_idx:04d}_{side}.png"
        img.save(os.path.join(images_dir, img_name))
        all_boxes[side] = boxes

    meta_path = os.path.join(metadata_dir, f"{doc_type}_{sample_idx:04d}_meta.json")
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
        json_samples = list(config.get("json_samples", []))
        json_dir = config.get("json_dir")
        if json_dir:
            json_samples += sorted(glob.glob(os.path.join(json_dir, "*.json")))

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