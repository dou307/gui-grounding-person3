import argparse
import csv
import json
from pathlib import Path


FINAL_METHODS = [
    ("B0", "Original Qwen3-VL zero-shot", "b0_screenspot_metrics.json"),
    ("B1", "Hard data + Point LoRA", "b1_screenspot_metrics.json"),
    ("Final", "B1 + Coarse-to-Fine + Retry", "final_c2f_retry_screenspot_metrics.json"),
]


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def group_accuracy(metrics, key):
    if not metrics:
        return None
    group = metrics.get("groups", {}).get(key)
    if not group:
        return None
    return group.get("accuracy")


def fmt(value, digits=4):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def collect_rows(metrics_dir):
    rows = []
    for name, label, filename in FINAL_METHODS:
        metrics_path = Path(metrics_dir) / filename
        metrics = load_json(metrics_path)
        rows.append(
            {
                "method": name,
                "label": label,
                "metrics_path": str(metrics_path),
                "exists": metrics is not None,
                "total": metrics.get("total") if metrics else None,
                "parsed": metrics.get("parsed") if metrics else None,
                "parse_success_rate": metrics.get("parse_success_rate") if metrics else None,
                "click_accuracy": metrics.get("click_accuracy") if metrics else None,
                "click_accuracy_on_parsed": metrics.get("click_accuracy_on_parsed") if metrics else None,
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
        "total",
        "parsed",
        "parse_success_rate",
        "click_accuracy",
        "click_accuracy_on_parsed",
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
        "# Final Experiment Results Summary",
        "",
        "| Method | Description | Total | Click Acc | Parse Success | Mean Center Distance |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {label} | {total} | {click} | {parse} | {dist} |".format(
                method=row["method"],
                label=row["label"],
                total=fmt(row["total"]),
                click=fmt(row["click_accuracy"]),
                parse=fmt(row["parse_success_rate"]),
                dist=fmt(row["mean_center_distance"]),
            )
        )

    lines.extend(
        [
            "",
            "## Target Size Breakdown",
            "",
            "| Method | Small | Medium | Large |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {method} | {small} | {medium} | {large} |".format(
                method=row["method"],
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
            "| Method | Desktop | Mobile | Web |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {method} | {desktop} | {mobile} | {web} |".format(
                method=row["method"],
                desktop=fmt(row["desktop_accuracy"]),
                mobile=fmt(row["mobile_accuracy"]),
                web=fmt(row["web_accuracy"]),
            )
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    rows = collect_rows(args.metrics_dir)
    write_markdown(args.out_md, rows)
    write_csv(args.out_csv, rows)
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
