import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from .region import point_in_bbox
from .schema import enriched_sample, parse_prediction, read_jsonl, resolve_image_path


def load_predictions(path):
    preds = {}
    for row in read_jsonl(path):
        preds[row["id"]] = parse_prediction(row)
    return preds


def scale_bbox_to_pixel(bbox_1000, width, height):
    x1, y1, x2, y2 = bbox_1000
    return [x1 / 1000 * width, y1 / 1000 * height, x2 / 1000 * width, y2 / 1000 * height]


def scale_point_to_pixel(point_1000, width, height):
    x, y = point_1000
    return [x / 1000 * width, y / 1000 * height]


def visualize(truth_path, pred_path, out_dir, limit):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = load_predictions(pred_path)
    written = 0
    for sample in read_jsonl(truth_path):
        if written >= limit:
            break
        gt = enriched_sample(sample, truth_path)
        pred = preds.get(gt["id"])
        if not pred or not pred.get("parse_success"):
            continue

        image_path = resolve_image_path(gt, truth_path)
        with Image.open(image_path).convert("RGB") as img:
            draw = ImageDraw.Draw(img)
            width, height = img.size
            bbox_px = scale_bbox_to_pixel(gt["bbox_1000"], width, height)
            point_px = scale_point_to_pixel([pred["x"], pred["y"]], width, height)

            draw.rectangle(bbox_px, outline=(0, 220, 0), width=4)
            r = 7
            draw.ellipse(
                [point_px[0] - r, point_px[1] - r, point_px[0] + r, point_px[1] + r],
                fill=(255, 0, 0),
                outline=(255, 255, 255),
                width=2,
            )
            ok = point_in_bbox([pred["x"], pred["y"]], gt["bbox_1000"])
            title = f"{gt['id']} ok={ok} gt_region={gt['region']} pred_region={pred.get('region', 'NA')}"
            draw.rectangle([0, 0, min(width, 1100), 34], fill=(255, 255, 255))
            draw.text((8, 8), title, fill=(0, 0, 0))

            safe_id = str(gt["id"]).replace("/", "_").replace("\\", "_")
            img.save(out_dir / f"{written:03d}_{safe_id}.png")
            written += 1
    print(f"Wrote {written} visualizations to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    visualize(args.truth, args.pred, args.out_dir, args.limit)


if __name__ == "__main__":
    main()
