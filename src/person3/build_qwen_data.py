import argparse

from .schema import enriched_sample, read_jsonl, to_qwen_conversation, write_json


def build(input_path, output_path, method, limit=None):
    rows = []
    for idx, sample in enumerate(read_jsonl(input_path)):
        if limit is not None and idx >= limit:
            break
        enriched = enriched_sample(sample, input_path, point_fallback=True)
        rows.append(to_qwen_conversation(enriched, method))
    write_json(output_path, rows)
    print(f"Wrote {len(rows)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", required=True, choices=["direct", "region_point", "target_region_point"])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    build(args.input, args.output, args.method, args.limit)


if __name__ == "__main__":
    main()
