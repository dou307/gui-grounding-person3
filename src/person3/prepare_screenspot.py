import argparse
from pathlib import Path

from datasets import load_dataset

from .region import bbox_center, clamp_1000
from .schema import write_jsonl


def rel_bbox_to_1000(bbox):
    return [clamp_1000(float(v) * 1000) for v in bbox]


def prepare(output_jsonl, image_dir, limit=None):
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("rootsautomation/ScreenSpot", split="test")
    rows = []
    for idx, sample in enumerate(ds):
        if limit is not None and idx >= limit:
            break
        image = sample["image"]
        file_name = sample.get("file_name") or f"screenspot_{idx:06d}.png"
        image_path = image_dir / file_name
        image.save(image_path)
        bbox_1000 = rel_bbox_to_1000(sample["bbox"])
        cx, cy = bbox_center(bbox_1000)
        rows.append(
            {
                "id": f"screenspot_{idx:06d}",
                "image": str(image_path),
                "instruction": sample["instruction"],
                "bbox_1000": bbox_1000,
                "point_1000": [clamp_1000(cx), clamp_1000(cy)],
                "platform": sample.get("data_source", "unknown"),
                "target_type": sample.get("data_type", "unknown"),
                "source": "screenspot",
                "file_name": file_name,
            }
        )
    write_jsonl(output_jsonl, rows)
    print(f"Wrote {len(rows)} ScreenSpot samples to {output_jsonl}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    prepare(args.output_jsonl, args.image_dir, args.limit)


if __name__ == "__main__":
    main()

