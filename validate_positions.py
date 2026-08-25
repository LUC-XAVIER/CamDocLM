"""
Precise validation: draws crosshairs + coordinate rulers on annotated images.
"""
from PIL import Image, ImageDraw, ImageFont
import yaml, os

os.makedirs("data/output/validation", exist_ok=True)

CONFIGS = {
    "cameroon_driving_license": "configs/cameroon_driving_license.yaml",
    "cameroon_nic_v1":          "configs/cameroon_nic_v1.yaml",
    "cameroon_passport":        "configs/cameroon_passport.yaml",
}

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)

def draw_crosshair(draw, x, y, label, color=(0, 230, 0, 220), size=12):
    draw.line([x - size, y, x + size, y], fill=color, width=2)
    draw.line([x, y - size, x, y + size], fill=color, width=2)
    draw.rectangle([x-3, y-3, x+3, y+3], fill=color)
    draw.text((x + 14, y - 9), f"{label} ({x},{y})", fill=color)

def draw_grid(draw, W, H, step=50):
    """Light grid every 50px for reference."""
    for x in range(0, W, step):
        draw.line([x, 0, x, H], fill=(180, 180, 180, 80), width=1)
    for y in range(0, H, step):
        draw.line([0, y, W, y], fill=(180, 180, 180, 80), width=1)
    # ruler labels
    for x in range(0, W, 100):
        draw.text((x+2, 2), str(x), fill=(100, 100, 100, 180))
    for y in range(0, H, 100):
        draw.text((2, y+2), str(y), fill=(100, 100, 100, 180))

def annotate(doc_key, cfg_path):
    cfg = load_yaml(cfg_path)
    backgrounds = cfg["backgrounds"]
    fields      = cfg["fields"]
    side_map = {"front": 0, "back": 1}

    by_side = {}
    for field in fields:
        side = field.get("side", "front")
        by_side.setdefault(side, []).append(field)

    for side, flds in by_side.items():
        idx = side_map.get(side, 0)
        if idx >= len(backgrounds):
            print(f"  [WARN] No background for side='{side}' (idx {idx})")
            continue
        img = Image.open(backgrounds[idx]).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0,0,0,0))
        draw = ImageDraw.Draw(overlay)
        W, H = img.size
        draw_grid(draw, W, H)
        print(f"\n--- {doc_key} | {side} | {W}x{H} ---")
        for f in flds:
            x, y = f["position"]
            name = f["name"]
            in_bounds = (0 <= x < W) and (0 <= y < H)
            flag = "OK" if in_bounds else "OUT-OF-BOUNDS!"
            print(f"  {name:30s}  ({x:4d},{y:4d})  {flag}")
            color = (0, 230, 0, 220) if in_bounds else (255, 50, 50, 220)
            draw_crosshair(draw, x, y, name, color)

        out = Image.alpha_composite(img, overlay)
        out_path = f"data/output/validation/{doc_key}_{side}_precise.png"
        out.convert("RGB").save(out_path)
        print(f"  -> {out_path}")

for doc_key, cfg_path in CONFIGS.items():
    print(f"\n{'='*60}")
    print(f"DOC: {doc_key}")
    annotate(doc_key, cfg_path)
