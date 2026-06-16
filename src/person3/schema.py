import json
import os
import re
from pathlib import Path

from PIL import Image

from .region import bbox_center, bbox_to_region, clamp_1000, pixel_bbox_to_1000


CLICK_WORDS = (
    "点击",
    "点按",
    "选择",
    "打开",
    "进入",
    "press",
    "tap",
    "click",
    "select",
    "open",
)


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


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_image_path(sample, input_path):
    image = Path(sample["image"])
    if image.is_absolute():
        return image
    candidates = [
        Path(input_path).parent / image,
        Path.cwd() / image,
    ]
    image_root = os.environ.get("IMAGE_ROOT")
    if image_root:
        candidates.append(Path(image_root) / image)
    project_root = os.environ.get("PROJECT_ROOT")
    if project_root:
        candidates.extend(
            [
                Path(project_root) / image,
                Path(project_root) / "person3" / image,
                Path(project_root) / "data" / "images" / image,
                Path(project_root) / "data" / "splits" / image,
                Path(project_root) / "data" / "rico_imgs" / "combined" / image,
                Path(project_root) / "person3" / "data" / "images" / image,
                Path(project_root) / "person3" / "data" / "splits" / image,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def normalized_bbox(sample, input_path=None):
    if "bbox_1000" in sample:
        return [clamp_1000(v) for v in sample["bbox_1000"]]

    if "bbox_pixel" not in sample:
        raise ValueError(f"Sample {sample.get('id')} has no bbox_1000 or bbox_pixel")

    if "img_w" in sample and "img_h" in sample:
        return pixel_bbox_to_1000(sample["bbox_pixel"], int(sample["img_w"]), int(sample["img_h"]))

    if input_path is None:
        raise ValueError("input_path is required when converting bbox_pixel")

    image_path = resolve_image_path(sample, input_path)
    with Image.open(image_path) as img:
        width, height = img.size
    return pixel_bbox_to_1000(sample["bbox_pixel"], width, height)


def normalized_point(sample, bbox):
    if "point_1000" in sample:
        return [clamp_1000(v) for v in sample["point_1000"]]
    x, y = bbox_center(bbox)
    return [clamp_1000(x), clamp_1000(y)]


def target_from_instruction(instruction: str) -> str:
    target = instruction.strip()
    target = re.sub(r"^[请你帮我\s]*", "", target)
    for word in CLICK_WORDS:
        target = re.sub(rf"^{re.escape(word)}", "", target, flags=re.IGNORECASE)
    target = target.strip(" ：:，,。.;；")
    return target or instruction.strip()


def enriched_sample(sample, input_path=None, point_fallback=False):
    try:
        bbox = normalized_bbox(sample, input_path)
    except (FileNotFoundError, ValueError):
        if not point_fallback or "point_1000" not in sample:
            raise
        x, y = normalized_point(sample, [0, 0, 0, 0])
        bbox = [x, y, x, y]
    point = normalized_point(sample, bbox)
    instruction = sample["instruction"]
    return {
        **sample,
        "bbox_1000": bbox,
        "point_1000": point,
        "region": bbox_to_region(bbox),
        "target": sample.get("target") or target_from_instruction(instruction),
    }


def training_answer(sample, method: str):
    point = sample["point_1000"]
    if method == "direct":
        return {"action": "click", "x": point[0], "y": point[1]}
    if method == "region_point":
        return {"region": sample["region"], "action": "click", "x": point[0], "y": point[1]}
    if method == "target_region_point":
        return {
            "target": sample["target"],
            "region": sample["region"],
            "action": "click",
            "x": point[0],
            "y": point[1],
        }
    raise ValueError(f"Unknown method: {method}")


def prompt_for_method(instruction: str, method: str) -> str:
    if method == "direct":
        return (
            "<image>\n"
            "请根据指令定位需要点击的元素。"
            f"指令：{instruction}\n"
            "只输出 JSON，格式为 {\"action\":\"click\",\"x\":整数,\"y\":整数}。"
            "坐标范围是 0 到 1000。"
        )
    if method == "region_point":
        return (
            "<image>\n"
            "请根据指令定位需要点击的元素。先判断目标所在九宫格粗区域，再输出点击坐标。"
            f"指令：{instruction}\n"
            "region 只能是 top-left、top、top-right、left、center、right、bottom-left、bottom、bottom-right。"
            "只输出 JSON，格式为 {\"region\":\"top-right\",\"action\":\"click\",\"x\":整数,\"y\":整数}。"
            "坐标范围是 0 到 1000。"
        )
    if method == "target_region_point":
        return (
            "<image>\n"
            "请根据指令定位需要点击的元素。先写出目标控件，再判断九宫格粗区域，最后输出点击坐标。"
            f"指令：{instruction}\n"
            "region 只能是 top-left、top、top-right、left、center、right、bottom-left、bottom、bottom-right。"
            "只输出 JSON，格式为 {\"target\":\"目标控件\",\"region\":\"top-right\",\"action\":\"click\",\"x\":整数,\"y\":整数}。"
            "坐标范围是 0 到 1000。"
        )
    raise ValueError(f"Unknown method: {method}")


def to_qwen_conversation(sample, method: str):
    answer = training_answer(sample, method)
    return {
        "image": sample["image"],
        "conversations": [
            {"from": "human", "value": prompt_for_method(sample["instruction"], method)},
            {"from": "gpt", "value": json.dumps(answer, ensure_ascii=False, separators=(",", ":"))},
        ],
    }


def extract_json_object(text):
    if isinstance(text, dict):
        return text
    if text is None:
        return None
    text = str(text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def parse_prediction(row):
    obj = None
    for key in ("prediction", "output", "raw_output", "text"):
        if key in row:
            obj = extract_json_object(row[key])
            if obj is not None:
                break
    if obj is None:
        obj = extract_json_object(row)
    if obj is None:
        return {"parse_success": False, "raw": row}

    parsed = {"parse_success": True, "raw": row}
    if "region" in obj:
        parsed["region"] = str(obj["region"]).strip()
    if "target" in obj:
        parsed["target"] = str(obj["target"]).strip()
    try:
        parsed["x"] = clamp_1000(float(obj["x"]))
        parsed["y"] = clamp_1000(float(obj["y"]))
    except (KeyError, TypeError, ValueError):
        parsed["parse_success"] = False
    return parsed
