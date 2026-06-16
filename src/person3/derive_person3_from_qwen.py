import argparse
import json
import re

from .region import point_to_region
from .schema import prompt_for_method, read_jsonl, target_from_instruction, write_jsonl


def extract_instruction(prompt: str) -> str:
    prompt = prompt.replace("<image>\n", "")
    match = re.search(r"指令[:：](.*?)(?:。只输出|。|$)", prompt)
    if match:
        return match.group(1).strip()
    match = re.search(r"instruction[:：](.*?)(?:\.|$)", prompt, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return prompt.strip()


def parse_answer(value: str):
    value = value.strip()
    start = value.find("{")
    end = value.rfind("}")
    if start >= 0 and end > start:
        value = value[start : end + 1]
    obj = json.loads(value)
    return int(round(float(obj["x"]))), int(round(float(obj["y"])))


def derive_row(row, method: str):
    conversations = row["conversations"]
    human = conversations[0]["value"]
    assistant = conversations[-1]["value"]
    instruction = extract_instruction(human)
    x, y = parse_answer(assistant)
    region = point_to_region(x, y)
    if method == "direct":
        answer = {"action": "click", "x": x, "y": y}
    elif method == "region_point":
        answer = {"region": region, "action": "click", "x": x, "y": y}
    elif method == "target_region_point":
        answer = {
            "target": target_from_instruction(instruction),
            "region": region,
            "action": "click",
            "x": x,
            "y": y,
        }
    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "image": row["image"],
        "conversations": [
            {"from": "human", "value": prompt_for_method(instruction, method)},
            {"from": "gpt", "value": json.dumps(answer, ensure_ascii=False, separators=(",", ":"))},
        ],
    }


def derive(input_path, output_path, method, limit=None):
    rows = []
    skipped = 0
    for idx, row in enumerate(read_jsonl(input_path)):
        if limit is not None and idx >= limit:
            break
        try:
            rows.append(derive_row(row, method))
        except Exception:
            skipped += 1
    write_jsonl(output_path, rows)
    print(f"Wrote {len(rows)} samples to {output_path}; skipped={skipped}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", required=True, choices=["direct", "region_point", "target_region_point"])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    derive(args.input, args.output, args.method, args.limit)


if __name__ == "__main__":
    main()
