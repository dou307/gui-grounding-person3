import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

from .schema import extract_json_object, prompt_for_method, read_jsonl, resolve_image_path


def clamp_1000(value):
    return max(0, min(1000, int(round(float(value)))))


def point_from_text(text):
    obj = extract_json_object(text)
    if not isinstance(obj, dict):
        return None
    try:
        return clamp_1000(obj["x"]), clamp_1000(obj["y"])
    except (KeyError, TypeError, ValueError):
        return None


def final_raw_output(point):
    if point is None:
        return ""
    x, y = point
    return json.dumps({"action": "click", "x": x, "y": y}, ensure_ascii=False, separators=(",", ":"))


def load_model(model_id, adapter=None):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    if adapter:
        if PeftModel is None:
            raise RuntimeError("peft is required when --adapter is set")
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model


def load_processor(model_id, min_pixels=None, max_pixels=None):
    kwargs = {}
    if min_pixels is not None:
        kwargs["min_pixels"] = min_pixels
    if max_pixels is not None:
        kwargs["max_pixels"] = max_pixels
    processor = AutoProcessor.from_pretrained(model_id, **kwargs)
    if getattr(processor, "tokenizer", None) is not None:
        processor.tokenizer.padding_side = "left"
    return processor


def generate_one(processor, model, image, prompt, max_new_tokens=64, temperature=0.0):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = 0.9

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generation_kwargs)
    generated_trimmed = generated_ids[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def crop_around_point(image, point_1000, crop_ratio):
    width, height = image.size
    cx = point_1000[0] / 1000 * width
    cy = point_1000[1] / 1000 * height
    crop_w = width * crop_ratio
    crop_h = height * crop_ratio

    x1 = max(0, int(cx - crop_w / 2))
    y1 = max(0, int(cy - crop_h / 2))
    x2 = min(width, int(cx + crop_w / 2))
    y2 = min(height, int(cy + crop_h / 2))
    return image.crop((x1, y1, x2, y2)), x1, y1, x2 - x1, y2 - y1


def crop_local_to_global(local_point, crop_x1, crop_y1, crop_w, crop_h, img_w, img_h):
    local_x, local_y = local_point
    local_px = local_x / 1000 * crop_w
    local_py = local_y / 1000 * crop_h
    global_px = crop_x1 + local_px
    global_py = crop_y1 + local_py
    return clamp_1000(global_px / img_w * 1000), clamp_1000(global_py / img_h * 1000)


def direct_prompt(instruction):
    return prompt_for_method(instruction, "direct").replace("<image>\n", "")


def coarse_prompt(instruction):
    return (
        "请根据指令定位需要点击的元素的大致位置。"
        f"指令：{instruction}\n"
        "只输出 JSON，格式为 {\"action\":\"click\",\"x\":整数,\"y\":整数}。"
        "坐标范围是 0 到 1000。"
    )


def fine_prompt(instruction):
    return (
        "这是一张围绕粗定位结果裁剪出的局部截图。"
        "请在这张局部截图中精确定位指令对应的元素。"
        f"指令：{instruction}\n"
        "只输出 JSON，格式为 {\"action\":\"click\",\"x\":整数,\"y\":整数}。"
        "坐标范围是当前局部截图内的 0 到 1000。"
    )


def predict_direct(processor, model, image, instruction, args):
    raw = generate_one(processor, model, image, direct_prompt(instruction), args.max_new_tokens, args.temperature)
    point = point_from_text(raw)
    return {
        "point": point,
        "raw_output": raw,
        "parse_success": point is not None,
        "stage": "direct" if point is not None else "direct_fail",
    }


def predict_coarse_to_fine(processor, model, image, instruction, args, temperature=None):
    temp = args.temperature if temperature is None else temperature
    start = time.time()
    img_w, img_h = image.size

    raw_coarse = generate_one(processor, model, image, coarse_prompt(instruction), args.max_new_tokens, temp)
    coarse_point = point_from_text(raw_coarse)
    if coarse_point is None:
        return {
            "point": None,
            "raw_output": raw_coarse,
            "parse_success": False,
            "stage": "coarse_fail",
            "coarse_point": None,
            "infer_time": time.time() - start,
        }

    cropped, crop_x1, crop_y1, crop_w, crop_h = crop_around_point(image, coarse_point, args.crop_ratio)
    raw_fine = generate_one(processor, model, cropped, fine_prompt(instruction), args.max_new_tokens, temp)
    fine_local_point = point_from_text(raw_fine)
    if fine_local_point is None:
        return {
            "point": coarse_point,
            "raw_output": raw_fine,
            "parse_success": True,
            "stage": "coarse_fallback",
            "coarse_point": coarse_point,
            "infer_time": time.time() - start,
        }

    final_point = crop_local_to_global(fine_local_point, crop_x1, crop_y1, crop_w, crop_h, img_w, img_h)
    return {
        "point": final_point,
        "raw_output": raw_fine,
        "parse_success": True,
        "stage": "fine",
        "coarse_point": coarse_point,
        "fine_local_point": fine_local_point,
        "infer_time": time.time() - start,
    }


def consistency(points):
    valid = [p for p in points if p is not None]
    if not valid:
        return {
            "num_valid": 0,
            "avg_distance": None,
            "consistency_score": None,
            "is_reliable": False,
            "center": None,
        }
    xs = sorted(p[0] for p in valid)
    ys = sorted(p[1] for p in valid)
    center = (xs[len(xs) // 2], ys[len(ys) // 2])
    if len(valid) == 1:
        return {
            "num_valid": 1,
            "avg_distance": None,
            "consistency_score": None,
            "is_reliable": False,
            "center": center,
        }

    distances = []
    for i, p1 in enumerate(valid):
        for p2 in valid[i + 1 :]:
            distances.append(((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5)
    avg_distance = sum(distances) / len(distances)
    score = avg_distance / 1000
    return {
        "num_valid": len(valid),
        "avg_distance": avg_distance,
        "consistency_score": score,
        "is_reliable": score < 0.15,
        "center": center,
    }


def predict_final(processor, model, image, instruction, args):
    attempts = []
    for _ in range(args.retry_samples):
        attempts.append(predict_coarse_to_fine(processor, model, image, instruction, args, args.retry_temperature))

    info = consistency([item["point"] for item in attempts])
    retries = 0
    while not info["is_reliable"] and retries < args.max_retries:
        retries += 1
        for _ in range(args.retry_samples):
            attempts.append(predict_coarse_to_fine(processor, model, image, instruction, args, args.retry_temperature))
        info = consistency([item["point"] for item in attempts])

    final_point = info["center"]
    return {
        "point": final_point,
        "raw_output": final_raw_output(final_point),
        "parse_success": final_point is not None,
        "stage": "final_retry" if final_point is not None else "final_fail",
        "retry": {
            "num_attempts": len(attempts),
            "num_retries": retries,
            "consistency": info,
        },
        "attempts": attempts,
        "infer_time": sum(item.get("infer_time", 0) for item in attempts),
    }


def append_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_completed_ids(path):
    path = Path(path)
    if not path.exists():
        return set()
    return {row["id"] for row in read_jsonl(path) if "id" in row}


def prediction_row(sample, result):
    point = result.pop("point", None)
    model_raw_output = result.pop("raw_output", "")
    result.pop("parse_success", None)
    row = {
        "id": sample["id"],
        "raw_output": final_raw_output(point) if point is not None else model_raw_output,
        "model_raw_output": model_raw_output,
        "parse_success": point is not None,
        **result,
    }
    if point is not None:
        row["x"], row["y"] = point
    return row


def infer(args):
    processor = load_processor(args.model, args.min_pixels, args.max_pixels)
    model = load_model(args.model, args.adapter)

    completed = load_completed_ids(args.output) if args.resume else set()
    if not args.resume and Path(args.output).exists():
        Path(args.output).unlink()

    written = len(completed)
    for idx, sample in enumerate(read_jsonl(args.input)):
        if args.limit is not None and idx >= args.limit:
            break
        if sample["id"] in completed:
            continue

        image_path = resolve_image_path(sample, args.input)
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        if args.mode == "direct":
            result = predict_direct(processor, model, image, sample["instruction"], args)
        elif args.mode == "coarse_to_fine":
            result = predict_coarse_to_fine(processor, model, image, sample["instruction"], args)
        elif args.mode == "final":
            result = predict_final(processor, model, image, sample["instruction"], args)
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

        append_jsonl(args.output, [prediction_row(sample, result)])
        written += 1
        if written % args.log_every == 0:
            print(f"inferred {written} samples")

    print(f"Wrote {written} predictions to {args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--adapter")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["direct", "coarse_to_fine", "final"], required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=1, help="Accepted for command compatibility; final inference runs sample by sample.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--min-pixels", type=int)
    parser.add_argument("--max-pixels", type=int, default=401408)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--crop-ratio", type=float, default=0.3)
    parser.add_argument("--retry-samples", type=int, default=3)
    parser.add_argument("--retry-temperature", type=float, default=0.7)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    infer(args)


if __name__ == "__main__":
    main()
