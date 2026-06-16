"""Convert Widget Captioning annotations from OS-Atlas-data to project JSONL.

The current team-agreed public training data is Widget Captioning. The exact
raw annotation filename may differ, so this converter supports common field
names observed in GUI grounding datasets:

- image path: img_filename, image, file_name, filename
- instruction: instruction, query, question, text
- bbox: bbox, box, target_bbox

The bbox is expected to be normalized [left, top, right, bottom] in [0, 1].
"""

import argparse
import json
from pathlib import Path

from .region import bbox_center, clamp_1000
from .schema import write_jsonl


IMAGE_KEYS = ("img_filename", "image", "file_name", "filename", "img")
INSTRUCTION_KEYS = ("instruction", "query", "question", "text", "caption")
BBOX_KEYS = ("bbox", "box", "target_bbox", "bounds")


def first_existing(row, keys):
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def iter_records(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return
    if text[0] == "[":
        for row in json.loads(text):
            yield row
    else:
        for line_no, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON") from exc


def rel_bbox_to_1000(bbox):
    if len(bbox) != 4:
        raise ValueError(f"bbox should have 4 values, got {bbox}")
    return [clamp_1000(float(v) * 1000) for v in bbox]


def normalize_image_path(img_filename, image_root=None):
    path = Path(str(img_filename))
    if path.is_absolute():
        return str(path)
    if image_root:
        return str(Path(image_root) / path)
    return str(path).replace("\\", "/")


def convert(annotations, output, image_root=None, limit=None):
    rows = []
    skipped = 0
    for idx, row in enumerate(iter_records(annotations)):
        if limit is not None and idx >= limit:
            break
        img_filename = first_existing(row, IMAGE_KEYS)
        instruction = first_existing(row, INSTRUCTION_KEYS)
        bbox = first_existing(row, BBOX_KEYS)
        if img_filename is None or instruction is None or bbox is None:
            skipped += 1
            continue

        bbox_1000 = rel_bbox_to_1000(bbox)
        cx, cy = bbox_center(bbox_1000)
        rows.append(
            {
                "id": row.get("id") or f"widget_captioning_{idx:08d}",
                "image": normalize_image_path(img_filename, image_root),
                "instruction": str(instruction),
                "bbox_1000": bbox_1000,
                "point_1000": [clamp_1000(cx), clamp_1000(cy)],
                "platform": row.get("platform") or row.get("data_source") or "mobile",
                "target_type": row.get("target_type") or row.get("data_type") or "unknown",
                "source": "widget_captioning",
            }
        )

    write_jsonl(output, rows)
    print(f"Wrote {len(rows)} samples to {output}; skipped={skipped}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-root")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    convert(args.annotations, args.output, args.image_root, args.limit)


if __name__ == "__main__":
    main()
