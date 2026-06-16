import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .region import bbox_center, bbox_to_region, point_in_bbox, size_bucket
from .schema import enriched_sample, parse_prediction, read_jsonl


def distance_to_center(point, bbox):
    cx, cy = bbox_center(bbox)
    x, y = point
    return ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5


def safe_div(num, den):
    return num / den if den else 0.0


def load_predictions(path):
    preds = {}
    for row in read_jsonl(path):
        if "id" not in row:
            raise ValueError(f"Prediction row has no id: {row}")
        preds[row["id"]] = parse_prediction(row)
    return preds


def add_group(groups, key, correct):
    groups[key]["total"] += 1
    groups[key]["correct"] += int(correct)


def evaluate(truth_path, pred_path):
    preds = load_predictions(pred_path)
    total = 0
    parsed_count = 0
    click_correct = 0
    region_total = 0
    region_correct = 0
    region_wrong = 0
    precision_wrong = 0
    missing_pred = 0
    distances = []
    groups = defaultdict(lambda: {"total": 0, "correct": 0})
    error_types = Counter()

    for sample in read_jsonl(truth_path):
        total += 1
        gt = enriched_sample(sample, truth_path)
        pred = preds.get(gt["id"])
        if pred is None:
            missing_pred += 1
            error_types["missing_prediction"] += 1
            continue
        if not pred.get("parse_success"):
            error_types["parse_failed"] += 1
            continue

        parsed_count += 1
        point = [pred.get("x"), pred.get("y")]
        bbox = gt["bbox_1000"]
        correct = point_in_bbox(point, bbox)
        click_correct += int(correct)
        distances.append(distance_to_center(point, bbox))

        gt_region = gt["region"]
        pred_region = pred.get("region")
        if pred_region:
            region_total += 1
            is_region_correct = pred_region == gt_region
            region_correct += int(is_region_correct)
            if not is_region_correct:
                region_wrong += 1
            elif not correct:
                precision_wrong += 1
        elif not correct:
            error_types["point_outside_bbox"] += 1

        platform = gt.get("platform", "unknown")
        target_type = gt.get("target_type", "unknown")
        target_size = size_bucket(bbox)
        add_group(groups, f"platform:{platform}", correct)
        add_group(groups, f"target_type:{target_type}", correct)
        add_group(groups, f"target_size:{target_size}", correct)

    metrics = {
        "total": total,
        "parsed": parsed_count,
        "missing_prediction": missing_pred,
        "parse_success_rate": safe_div(parsed_count, total),
        "click_accuracy": safe_div(click_correct, total),
        "click_accuracy_on_parsed": safe_div(click_correct, parsed_count),
        "region_accuracy": safe_div(region_correct, region_total),
        "region_total": region_total,
        "region_wrong_count": region_wrong,
        "precision_wrong_count": precision_wrong,
        "mean_center_distance": safe_div(sum(distances), len(distances)),
        "groups": {
            key: {
                "total": val["total"],
                "correct": val["correct"],
                "accuracy": safe_div(val["correct"], val["total"]),
            }
            for key, val in sorted(groups.items())
        },
        "error_types": dict(error_types),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    metrics = evaluate(args.truth, args.pred)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

