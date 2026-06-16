import argparse
import random

from .schema import read_jsonl, write_jsonl


def split(input_path, train_output, val_output, val_ratio, seed, max_samples=None):
    rows = list(read_jsonl(input_path))
    if max_samples is not None:
        rows = rows[:max_samples]
    rng = random.Random(seed)
    rng.shuffle(rows)
    val_size = max(1, int(round(len(rows) * val_ratio))) if rows else 0
    val_rows = rows[:val_size]
    train_rows = rows[val_size:]
    write_jsonl(train_output, train_rows)
    write_jsonl(val_output, val_rows)
    print(f"total={len(rows)} train={len(train_rows)} val={len(val_rows)} seed={seed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--val-output", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    split(args.input, args.train_output, args.val_output, args.val_ratio, args.seed, args.max_samples)


if __name__ == "__main__":
    main()

