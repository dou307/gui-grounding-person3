import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Trainer, TrainingArguments


IGNORE_INDEX = -100


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON") from exc


def split_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_image_path(image, input_path, image_root=None):
    image = Path(image)
    if image.is_absolute():
        return image

    candidates = []
    if image_root:
        candidates.append(Path(image_root) / image)
    candidates.extend(
        [
            Path(input_path).parent / image,
            Path.cwd() / image,
        ]
    )
    project_root = os.environ.get("PROJECT_ROOT")
    if project_root:
        candidates.extend(
            [
                Path(project_root) / image,
                Path(project_root) / "person3" / image,
                Path(project_root) / "data" / "images" / image,
                Path(project_root) / "data" / "splits" / image,
                Path(project_root) / "person3" / "data" / "images" / image,
                Path(project_root) / "person3" / "data" / "splits" / image,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def qwen_row_to_messages(row):
    conversations = row["conversations"]
    human = conversations[0]["value"].replace("<image>\n", "")
    assistant = conversations[-1]["value"]
    return (
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": row["image"]},
                    {"type": "text", "text": human},
                ],
            }
        ],
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": row["image"]},
                    {"type": "text", "text": human},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant}],
            },
        ],
    )


class QwenGuiDataset(Dataset):
    def __init__(self, data_path, image_root=None, limit=None):
        self.data_path = str(data_path)
        self.image_root = image_root
        self.rows = []
        for idx, row in enumerate(read_jsonl(data_path)):
            if limit is not None and idx >= limit:
                break
            self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        prompt_messages, full_messages = qwen_row_to_messages(row)
        image_path = resolve_image_path(row["image"], self.data_path, self.image_root)
        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found for sample index={index}, image={row['image']}. "
                f"Resolved path: {image_path}. Set --image-root or IMAGE_ROOT to the directory containing images."
            )
        return {
            "id": row.get("id", str(index)),
            "image_path": str(image_path),
            "prompt_messages": prompt_messages,
            "full_messages": full_messages,
        }


@dataclass
class QwenGuiDataCollator:
    processor: object
    max_length: int

    def __call__(self, features):
        images = [Image.open(item["image_path"]).convert("RGB") for item in features]
        prompt_texts = [
            self.processor.apply_chat_template(
                item["prompt_messages"],
                tokenize=False,
                add_generation_prompt=True,
            )
            for item in features
        ]
        full_texts = [
            self.processor.apply_chat_template(
                item["full_messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            for item in features
        ]

        model_inputs = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_inputs = self.processor(
            text=prompt_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = model_inputs["input_ids"].clone()
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1).tolist()
        for row_idx, prompt_length in enumerate(prompt_lengths):
            labels[row_idx, : int(prompt_length)] = IGNORE_INDEX
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX
        model_inputs["labels"] = labels
        return model_inputs


def load_processor(model_path, min_pixels, max_pixels):
    kwargs = {}
    if min_pixels is not None:
        kwargs["min_pixels"] = min_pixels
    if max_pixels is not None:
        kwargs["max_pixels"] = max_pixels
    processor = AutoProcessor.from_pretrained(model_path, **kwargs)
    if getattr(processor, "tokenizer", None) is not None:
        processor.tokenizer.padding_side = "right"
    return processor


def build_model(args):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=split_csv(args.lora_target_modules),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def check_images(dataset, limit):
    missing = []
    check_count = len(dataset) if limit is None or limit < 0 else min(len(dataset), limit)
    for idx in range(check_count):
        row = dataset.rows[idx]
        image_path = resolve_image_path(row["image"], dataset.data_path, dataset.image_root)
        if not image_path.exists():
            missing.append((idx, row["image"], str(image_path)))
            if len(missing) >= 5:
                break
    if missing:
        lines = [
            "Training images are missing. Pass the correct image directory with --image-root or IMAGE_ROOT.",
            "Examples:",
            "  IMAGE_ROOT=/home/ma-user/work/gui-project/data/images bash scripts/train_p3_direct_lora.sh --limit 8 --max-steps 2",
            "  bash scripts/train_p3_direct_lora.sh --image-root /home/ma-user/work/gui-project/data/images --limit 8 --max-steps 2",
            "Missing examples:",
        ]
        lines.extend(f"  index={idx}, image={image}, resolved={path}" for idx, image, path in missing)
        raise FileNotFoundError("\n".join(lines))


def train(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.logging_dir).mkdir(parents=True, exist_ok=True)

    dataset = QwenGuiDataset(args.train_file, image_root=args.image_root, limit=args.limit)
    if len(dataset) == 0:
        raise ValueError(f"No training samples loaded from {args.train_file}")
    check_images(dataset, args.check_image_limit)

    processor = load_processor(args.model, args.min_pixels, args.max_pixels)
    model = build_model(args)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16=True,
        bf16=False,
        optim=args.optim,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        report_to="none",
        logging_dir=args.logging_dir,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=QwenGuiDataCollator(processor=processor, max_length=args.max_length),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter and processor to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--logging-dir", required=True)
    parser.add_argument("--image-root")
    parser.add_argument("--check-image-limit", type=int, default=-1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--min-pixels", type=int)
    parser.add_argument("--max-pixels", type=int, default=602112)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
