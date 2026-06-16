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
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    if adapter:
        if PeftModel is None:
            raise RuntimeError("peft is required when --adapter is set")
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model


def infer(args):
    processor = AutoProcessor.from_pretrained(args.model)
    model = load_model(args.model, args.adapter)

    outputs = []
    for idx, sample in enumerate(read_jsonl(args.input)):
        if args.limit is not None and idx >= args.limit:
            break
        image_path = resolve_image_path(sample, args.input)
        messages = build_messages(sample, args.method, image_path)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
            )
        generated_trimmed = generated_ids[:, inputs["input_ids"].shape[1] :]
        raw_output = processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        outputs.append({"id": sample["id"], "raw_output": raw_output})
        if (idx + 1) % 20 == 0:
            print(f"inferred {idx + 1} samples")

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
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    infer(args)


if __name__ == "__main__":
    main()

