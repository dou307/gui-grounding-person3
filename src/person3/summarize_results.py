import argparse
import csv
import json
from pathlib import Path


METHODS = [
    ("direct", "P3-1 Direct Point", "p3_direct"),
    ("region_point", "P3-2 Region -> Point", "p3_region_point"),
    ("target_region_point", "P3-3 Target -> Region -> Point", "p3_target_region_point"),
]

SPLITS = [
    ("val_1000", "Val_1000"),
    ("screenspot", "ScreenSpot"),
]


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value, digits=4):
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def group_accuracy(metrics, key):
    if not metrics:
        return None
    group = metrics.get("groups", {}).get(key)
    if not group:
        return None
    return group.get("accuracy")


def collect_rows(metrics_dir):
    rows = []
    for method, label, prefix in METHODS:
        for split, split_label in SPLITS:
            metrics_path = Path(metrics_dir) / f"{prefix}_{split}_metrics.json"
            metrics = load_json(metrics_path)
            rows.append(
                {
                    "method": method,
                    "label": label,
                    "split": split,
                    "split_label": split_label,
                    "metrics_path": str(metrics_path),
                    "exists": metrics is not None,
                    "total": metrics.get("total") if metrics else None,
                    "parsed": metrics.get("parsed") if metrics else None,
                    "parse_success_rate": metrics.get("parse_success_rate") if metrics else None,
                    "click_accuracy": metrics.get("click_accuracy") if metrics else None,
                    "click_accuracy_on_parsed": metrics.get("click_accuracy_on_parsed") if metrics else None,
                    "region_accuracy": metrics.get("region_accuracy") if metrics else None,
                    "region_total": metrics.get("region_total") if metrics else None,
                    "region_wrong_count": metrics.get("region_wrong_count") if metrics else None,
                    "precision_wrong_count": metrics.get("precision_wrong_count") if metrics else None,
                    "mean_center_distance": metrics.get("mean_center_distance") if metrics else None,
                    "small_accuracy": group_accuracy(metrics, "target_size:small"),
                    "medium_accuracy": group_accuracy(metrics, "target_size:medium"),
                    "large_accuracy": group_accuracy(metrics, "target_size:large"),
                    "icon_accuracy": group_accuracy(metrics, "target_type:icon"),
                    "text_accuracy": group_accuracy(metrics, "target_type:text"),
                    "desktop_accuracy": group_accuracy(metrics, "platform:desktop"),
                    "mobile_accuracy": group_accuracy(metrics, "platform:mobile"),
                    "web_accuracy": group_accuracy(metrics, "platform:web"),
                }
            )
    return rows


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "label",
        "split",
        "total",
        "parsed",
        "parse_success_rate",
        "click_accuracy",
        "click_accuracy_on_parsed",
        "region_accuracy",
        "region_total",
        "region_wrong_count",
        "precision_wrong_count",
        "mean_center_distance",
        "small_accuracy",
        "medium_accuracy",
        "large_accuracy",
        "icon_accuracy",
        "text_accuracy",
        "desktop_accuracy",
        "mobile_accuracy",
        "web_accuracy",
        "metrics_path",
        "exists",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_markdown(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Person3 Results Summary",
        "",
        "## Main Metrics",
        "",
        "| Method | Split | Total | Click Acc | Parse Success | Region Acc | Mean Center Distance |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {split_label} | {total} | {click} | {parse} | {region} | {dist} |".format(
                label=row["label"],
                split_label=row["split_label"],
                total=fmt(row["total"]),
                click=fmt(row["click_accuracy"]),
                parse=fmt(row["parse_success_rate"]),
                region=fmt(row["region_accuracy"]),
                dist=fmt(row["mean_center_distance"]),
            )
        )

    lines.extend(
        [
            "",
            "## Error Decomposition",
            "",
            "| Method | Split | Region Total | Region Wrong | Precision Wrong |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {label} | {split_label} | {region_total} | {region_wrong} | {precision_wrong} |".format(
                label=row["label"],
                split_label=row["split_label"],
                region_total=fmt(row["region_total"]),
                region_wrong=fmt(row["region_wrong_count"]),
                precision_wrong=fmt(row["precision_wrong_count"]),
            )
        )

    lines.extend(
        [
            "",
            "## Target Size Breakdown",
            "",
            "| Method | Split | Small | Medium | Large |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {label} | {split_label} | {small} | {medium} | {large} |".format(
                label=row["label"],
                split_label=row["split_label"],
                small=fmt(row["small_accuracy"]),
                medium=fmt(row["medium_accuracy"]),
                large=fmt(row["large_accuracy"]),
            )
        )

    lines.extend(
        [
            "",
            "## Platform Breakdown",
            "",
            "| Method | Split | Desktop | Mobile | Web |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {label} | {split_label} | {desktop} | {mobile} | {web} |".format(
                label=row["label"],
                split_label=row["split_label"],
                desktop=fmt(row["desktop_accuracy"]),
                mobile=fmt(row["mobile_accuracy"]),
                web=fmt(row["web_accuracy"]),
            )
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", default="outputs/metrics/person3")
    parser.add_argument("--out-md", default="outputs/metrics/person3/person3_results_summary.md")
    parser.add_argument("--out-csv", default="outputs/metrics/person3/person3_results_summary.csv")
    args = parser.parse_args()

    rows = collect_rows(args.metrics_dir)
    write_markdown(args.out_md, rows)
    write_csv(args.out_csv, rows)
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
