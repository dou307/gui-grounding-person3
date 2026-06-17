import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .evaluate import distance_to_center
from .region import point_in_bbox, size_bucket
from .schema import enriched_sample, parse_prediction, read_jsonl, write_jsonl


METHODS = [
    ("direct", "p3_direct"),
    ("region_point", "p3_region_point"),
    ("target_region_point", "p3_target_region_point"),
]


def load_preds(path):
    preds = {}
    path = Path(path)
    if not path.exists():
        return preds
    for row in read_jsonl(path):
        preds[row["id"]] = parse_prediction(row)
    return preds


def prediction_status(sample, pred):
    if pred is None:
        return {
            "present": False,
            "parse_success": False,
            "click_correct": False,
            "distance": None,
            "region": None,
            "region_correct": None,
        }
    if not pred.get("parse_success"):
        return {
            "present": True,
            "parse_success": False,
            "click_correct": False,
            "distance": None,
            "region": pred.get("region"),
            "region_correct": None,
        }

    point = [pred.get("x"), pred.get("y")]
    bbox = sample["bbox_1000"]
    region = pred.get("region")
    region_correct = None
    if region:
        region_correct = region == sample["region"]
    return {
        "present": True,
        "parse_success": True,
        "click_correct": point_in_bbox(point, bbox),
        "x": pred.get("x"),
        "y": pred.get("y"),
        "distance": distance_to_center(point, bbox),
        "region": region,
        "region_correct": region_correct,
    }


def classify(statuses):
    direct = statuses.get("direct", {})
    region = statuses.get("region_point", {})
    target = statuses.get("target_region_point", {})
    d_ok = bool(direct.get("click_correct"))
    r_ok = bool(region.get("click_correct"))
    t_ok = bool(target.get("click_correct"))

    labels = []
    if r_ok and not d_ok:
        labels.append("region_improves_direct")
    if t_ok and not d_ok:
        labels.append("target_improves_direct")
    if d_ok and not r_ok:
        labels.append("region_regresses_direct")
    if d_ok and not t_ok:
        labels.append("target_regresses_direct")
    if d_ok and r_ok and t_ok:
        labels.append("all_correct")
    if not d_ok and not r_ok and not t_ok:
        labels.append("all_wrong")
    if r_ok and not t_ok:
        labels.append("target_worse_than_region")
    if t_ok and not r_ok:
        labels.append("target_better_than_region")
    return labels or ["mixed"]


def compare(args):
    pred_paths = {
        method: Path(args.pred_dir) / f"{prefix}_{args.split}.jsonl"
        for method, prefix in METHODS
    }
    preds_by_method = {method: load_preds(path) for method, path in pred_paths.items()}

    rows = []
    counts = Counter()
    by_platform = defaultdict(Counter)
    by_target_type = defaultdict(Counter)
    by_size = defaultdict(Counter)

    for raw_sample in read_jsonl(args.truth):
        sample = enriched_sample(raw_sample, args.truth, point_fallback=True)
        statuses = {
            method: prediction_status(sample, preds_by_method[method].get(sample["id"]))
            for method, _prefix in METHODS
        }
        labels = classify(statuses)
        for label in labels:
            counts[label] += 1
            by_platform[sample.get("platform", "unknown")][label] += 1
            by_target_type[sample.get("target_type", "unknown")][label] += 1
            by_size[size_bucket(sample["bbox_1000"])][label] += 1

        rows.append(
            {
                "id": sample["id"],
                "image": sample["image"],
                "instruction": sample["instruction"],
                "platform": sample.get("platform", "unknown"),
                "target_type": sample.get("target_type", "unknown"),
                "target_size": size_bucket(sample["bbox_1000"]),
                "bbox_1000": sample["bbox_1000"],
                "point_1000": sample["point_1000"],
                "gt_region": sample["region"],
                "labels": labels,
                "direct": statuses["direct"],
                "region_point": statuses["region_point"],
                "target_region_point": statuses["target_region_point"],
            }
        )

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    all_path = Path(args.out_dir) / f"person3_compare_{args.split}.jsonl"
    write_jsonl(all_path, rows)

    for label in sorted(counts):
        subset = [row for row in rows if label in row["labels"]]
        write_jsonl(Path(args.out_dir) / f"{label}_{args.split}.jsonl", subset[: args.limit_per_label])

    summary = {
        "split": args.split,
        "truth": args.truth,
        "prediction_paths": {method: str(path) for method, path in pred_paths.items()},
        "total": len(rows),
        "counts": dict(counts),
        "by_platform": {key: dict(value) for key, value in sorted(by_platform.items())},
        "by_target_type": {key: dict(value) for key, value in sorted(by_target_type.items())},
        "by_target_size": {key: dict(value) for key, value in sorted(by_size.items())},
    }
    summary_path = Path(args.out_dir) / f"person3_compare_{args.split}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {all_path}")
    print(f"Wrote {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", required=True)
    parser.add_argument("--pred-dir", default="outputs/predictions/person3")
    parser.add_argument("--split", required=True, help="For example: val_1000 or screenspot")
    parser.add_argument("--out-dir", default="outputs/analysis/person3")
    parser.add_argument("--limit-per-label", type=int, default=50)
    args = parser.parse_args()
    compare(args)


if __name__ == "__main__":
    main()
