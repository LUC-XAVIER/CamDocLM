"""
convert_to_hf_dataset.py
-------------------------
Converts one document type's generated images/ + metadata/ into
train/val/test JSONL files for LayoutLMv3 token classification.
Front and back sides become separate training examples (LayoutLMv3
takes one image at a time). Bboxes are normalized to the 0-1000 scale
LayoutLMv3 expects. BIO tags are derived directly from each word's
`label` + `word_index` (0 = start of a field = B-, otherwise I-).

Known limitation (see conversation): only field-value words are
included — no "O"-tagged static label text. Fine for a fast first
model; revisit if real-world inference struggles with label text.

Usage:
    python convert_to_hf_dataset.py --doc-dir data/output/Cameroon_NIC_v1 \
        --out-dir hf_data/nic_v1 --val-frac 0.1 --test-frac 0.1 --seed 42
"""

import os
import json
import glob
import random
import shutil
import argparse
from PIL import Image


def normalize_bbox(bbox, width, height):
    x0, y0, x1, y1 = bbox
    nx0 = min(1000, max(0, int(1000 * x0 / width)))
    ny0 = min(1000, max(0, int(1000 * y0 / height)))
    nx1 = min(1000, max(0, int(1000 * x1 / width)))
    ny1 = min(1000, max(0, int(1000 * y1 / height)))
    return [nx0, ny0, nx1, ny1]


def build_example(image_path, words_meta, example_id, path_prefix=""):
    with Image.open(image_path) as img:
        width, height = img.size

    words, bboxes, ner_tags = [], [], []
    for w in words_meta:
        words.append(w["text"])
        bboxes.append(normalize_bbox(w["bbox"], width, height))
        prefix = "B-" if w["word_index"] == 0 else "I-"
        ner_tags.append(f'{prefix}{w["label"].upper()}')

    stored_path = image_path.replace(os.sep, "/")
    if path_prefix:
        stored_path = f"{path_prefix.rstrip('/')}/{stored_path}"

    return {
        "id": example_id,
        "image_path": stored_path,
        "width": width,
        "height": height,
        "words": words,
        "bboxes": bboxes,
        "ner_tags": ner_tags,
    }


def collect_examples(doc_dir, path_prefix=""):
    images_dir = os.path.join(doc_dir, "images")
    metadata_dir = os.path.join(doc_dir, "metadata")
    meta_paths = sorted(glob.glob(os.path.join(metadata_dir, "*_meta.json")))

    examples = []
    all_labels = set()

    for meta_path in meta_paths:
        base = os.path.basename(meta_path).replace("_meta.json", "")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        for side, words_meta in meta.items():
            if not words_meta:
                continue
            image_path = os.path.join(images_dir, f"{base}_{side}.png")
            if not os.path.exists(image_path):
                print(f"  [skip] missing image for {base} ({side})")
                continue
            example_id = f"{base}_{side}"
            examples.append(build_example(image_path, words_meta, example_id, path_prefix))
            all_labels.update(w["label"].upper() for w in words_meta)

    return examples, sorted(all_labels)


def write_jsonl(examples, path):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Convert generated document data to LayoutLMv3 JSONL format")
    parser.add_argument("--doc-dir", required=True, help="e.g. data/output/Cameroon_NIC_v1")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--path-prefix", default="",
                         help="Prepended to every stored image_path, e.g. 'sample_data' "
                              "if that's where the files actually end up on Colab")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    examples, labels = collect_examples(args.doc_dir, args.path_prefix)
    if not examples:
        print("No examples found — check --doc-dir path.")
        return

    random.shuffle(examples)
    n = len(examples)
    n_val = int(n * args.val_frac)
    n_test = int(n * args.test_frac)
    val_examples = examples[:n_val]
    test_examples = examples[n_val:n_val + n_test]
    train_examples = examples[n_val + n_test:]

    write_jsonl(train_examples, os.path.join(args.out_dir, "train.jsonl"))
    write_jsonl(val_examples, os.path.join(args.out_dir, "val.jsonl"))
    write_jsonl(test_examples, os.path.join(args.out_dir, "test.jsonl"))

    # "O" included for schema completeness / future-proofing even though
    # no current example uses it — see the known limitation noted above
    bio_labels = ["O"] + [f"{p}{l}" for l in labels for p in ("B-", "I-")]
    id2label = {i: l for i, l in enumerate(bio_labels)}
    label2id = {l: i for i, l in enumerate(bio_labels)}
    with open(os.path.join(args.out_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f, indent=2)

    print(f"Wrote {len(train_examples)} train / {len(val_examples)} val / "
          f"{len(test_examples)} test examples to {args.out_dir}")
    print(f"{len(bio_labels)} labels: {bio_labels}")


if __name__ == "__main__":
    main()