import argparse
import json
from pathlib import Path

from .region import bbox_center, clamp_1000
from .schema import write_jsonl


def iter_records(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return
    if text[0] == "[":
        data = json.loads(text)
        for row in data:
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
    path = Path(img_filename)
    if path.is_absolute():
        return str(path)
    if image_root:
        return str(Path(image_root) / path)
    return str(path).replace("\\", "/")


def convert(annotations, output, image_root=None, source="seeclick", limit=None):
    rows = []
    for idx, row in enumerate(iter_records(annotations)):
        if limit is not None and idx >= limit:
            break
        bbox = row.get("bbox")
        instruction = row.get("instruction")
        img_filename = row.get("img_filename") or row.get("image") or row.get("file_name")
        if bbox is None or instruction is None or img_filename is None:
            continue
        bbox_1000 = rel_bbox_to_1000(bbox)
        cx, cy = bbox_center(bbox_1000)
        item = {
            "id": row.get("id") or f"{source}_{idx:08d}",
            "image": normalize_image_path(img_filename, image_root),
            "instruction": str(instruction),
            "bbox_1000": bbox_1000,
            "point_1000": [clamp_1000(cx), clamp_1000(cy)],
            "platform": row.get("platform") or row.get("data_source") or "unknown",
            "target_type": row.get("target_type") or row.get("data_type") or "unknown",
            "source": source,
        }
        rows.append(item)
    write_jsonl(output, rows)
    print(f"Wrote {len(rows)} converted samples to {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-root")
    parser.add_argument("--source", default="seeclick")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    convert(args.annotations, args.output, args.image_root, args.source, args.limit)


if __name__ == "__main__":
    main()

