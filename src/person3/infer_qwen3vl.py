import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

from .schema import prompt_for_method, read_jsonl, resolve_image_path, write_jsonl


def chunks(items, size):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_messages(sample, method, image_path):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt_for_method(sample["instruction"], method).replace("<image>\n", "")},
            ],
        }
    ]


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


def infer_batch(processor, model, samples, input_path, method, args):
    texts = []
    images = []
    ids = []
    for sample in samples:
        image_path = resolve_image_path(sample, input_path)
        messages = build_messages(sample, method, image_path)
        texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        with Image.open(image_path) as image:
            images.append(image.convert("RGB"))
        ids.append(sample["id"])

    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt").to(model.device)
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
    }
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generation_kwargs)
    generated_trimmed = generated_ids[:, inputs["input_ids"].shape[1] :]
    raw_outputs = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return [{"id": sample_id, "raw_output": raw_output} for sample_id, raw_output in zip(ids, raw_outputs)]


def infer(args):
    processor = load_processor(args.model, args.min_pixels, args.max_pixels)
    model = load_model(args.model, args.adapter)

    outputs = []
    rows = []
    for idx, sample in enumerate(read_jsonl(args.input)):
        if args.limit is not None and idx >= args.limit:
            break
        rows.append(sample)

    for batch in chunks(rows, args.batch_size):
        outputs.extend(infer_batch(processor, model, batch, args.input, args.method, args))
        if len(outputs) % args.log_every < args.batch_size:
            print(f"inferred {len(outputs)} samples")

    write_jsonl(args.output, outputs)
    print(f"Wrote {len(outputs)} predictions to {args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--adapter")
    parser.add_argument("--input", required=True)
    parser.add_argument("--method", required=True, choices=["direct", "region_point", "target_region_point"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--min-pixels", type=int)
    parser.add_argument("--max-pixels", type=int, default=401408)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()
    infer(args)


if __name__ == "__main__":
    main()
