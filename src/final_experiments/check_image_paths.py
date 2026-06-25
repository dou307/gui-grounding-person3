import argparse
from pathlib import Path

from src.person3.schema import read_jsonl, resolve_image_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--show", type=int, default=10)
    args = parser.parse_args()

    total = 0
    missing = []
    found_dirs = {}
    for idx, sample in enumerate(read_jsonl(args.input)):
        if args.limit is not None and idx >= args.limit:
            break
        total += 1
        path = resolve_image_path(sample, args.input)
        if path.exists():
            parent = str(Path(path).parent)
            found_dirs[parent] = found_dirs.get(parent, 0) + 1
        else:
            missing.append((idx, sample.get("image"), str(path)))

    print(f"total: {total}")
    print(f"missing: {len(missing)}")
    print(f"found_dirs: {found_dirs}")
    if missing:
        print("missing_examples:")
        for idx, image, path in missing[: args.show]:
            print(f"  index={idx}, image={image}, resolved={path}")

    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
