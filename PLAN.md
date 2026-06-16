# 第三成员完整执行计划

## 0. 已确定的公共数据方案

本项目公共数据采用以下配置：

| 类型 | 数据源 | 用途 | 文件名 |
|---|---|---|---|
| 训练集 | Widget Captioning，来自 OS-Atlas-data 仓库 | LoRA 训练 | `train_base_v1.jsonl` |
| 验证集 | 从 Widget Captioning 公共训练数据中固定划分 | 方法选择和消融对比 | `val_base_v1.jsonl` |
| 测试集 | ScreenSpot test split | 最终统一测试 | `screenspot_test.jsonl` |

规则：

- ScreenSpot 只用于最终测试，不参与训练和调参。
- RICO 第一阶段不加入公共基线，避免额外转换和数据噪声。
- 所有 bbox 和 point 统一为 `[0,1000]` 坐标。
- 第三成员的所有实验必须与公共 `Direct Point` 基线比较。

当前 1 号已统一的数据版本：

| 版本 | 样本数 | 说明 |
|---|---:|---|
| D1-Base | 101,425 | 原始 Widget Captioning 数据，未做任何平衡 |
| D1-Balanced | 86,412 | 按 Small/Medium/Large 目标大小重采样，每类 28,804 条 |
| D1-Hard | 115,216 | 在 Balanced 基础上，对小目标过采样 2 倍 |

当前已收到并放置的文件：

| 本地文件 | 对应含义 | 状态 |
|---|---|---|
| `data/splits/D1-Base_train_qwen.jsonl` | D1-Base 训练集，已是 Direct Point 的 Qwen 对话格式 | 可直接派生 3 号三套训练数据 |
| `data/splits/D1-Base_val.jsonl` | D1-Base 验证集，统一评测格式 | 可用于验证集推理和评测；若需按 bbox 评测，需要图片文件或补充 `bbox_1000/img_w/img_h` |
| `data/splits/screenspot_eval.jsonl` | ScreenSpot 最终测试集，统一评测格式 | 可用于最终测试评测 |

注意：`D1-Base_train_qwen.jsonl` 已经不是原始标注，而是 Direct Point Qwen 训练格式。因此 3 号不需要重新转换原始 Widget Captioning 标注，可以直接用 `derive_person3_from_qwen.py` 派生结构化训练集。

## 1. 第三成员研究问题

研究：

> 先预测目标所在粗区域或目标语义，再预测点击坐标，是否比直接预测点击点更好？

三组实验：

| 实验编号 | 方法 | 变量 |
|---|---|---|
| P3-1 | Direct Point | 直接预测坐标 |
| P3-2 | Region -> Point | 先预测九宫格区域，再预测坐标 |
| P3-3 | Target -> Region -> Point | 先预测目标语义，再预测区域和坐标 |

## 2. ModelArts 环境

每位成员使用：

```text
ModelArts Notebook
GPU：Tesla V100-PCIE-32GB
持久化工作目录：/home/ma-user/work，50GB
项目目录：/home/ma-user/work/gui-project/person3
```

创建环境：

```bash
cd /home/ma-user/work/gui-project/person3
bash scripts/install_env.sh
conda activate qwen3vl
```

第一次网络安装完成后，立刻把所有 wheel 包缓存到持久化目录：

```bash
bash scripts/cache_wheels.sh
```

以后如果 Notebook 重建或 `qwen3vl` 环境丢失，再运行 `bash scripts/install_env.sh` 时会优先从：

```text
/home/ma-user/work/gui-project/wheelhouse
```

本地安装，不需要重新下载几 GB 的 PyTorch/CUDA 依赖。

验证：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY
```

## 3. 公共数据准备

### 3.1 使用 1 号已提供的数据

1 号当前已经提供了 `D1-Base_train_qwen.jsonl`，格式如下：

```json
{
  "image": "10554.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\n请根据指令定位需要点击的元素。指令：...。只输出 JSON。"
    },
    {
      "from": "gpt",
      "value": "{\"action\": \"click\", \"x\": 69, \"y\": 71}"
    }
  ]
}
```

该文件已包含图片、指令和 Direct Point 答案，足够派生 3 号三种训练格式。若后续拿到原始 Widget Captioning 标注，再使用 `convert_widget_captioning.py`。

当前文件放置位置：

```text
$PROJECT_ROOT/person3/data/splits/D1-Base_train_qwen.jsonl
$PROJECT_ROOT/person3/data/splits/D1-Base_val.jsonl
$PROJECT_ROOT/person3/data/splits/screenspot_eval.jsonl
```

建议同步到统一项目目录：

```bash
mkdir -p $PROJECT_ROOT/data/splits
cp $PROJECT_ROOT/person3/data/splits/D1-Base_train_qwen.jsonl $PROJECT_ROOT/data/splits/
cp $PROJECT_ROOT/person3/data/splits/D1-Base_val.jsonl $PROJECT_ROOT/data/splits/
cp $PROJECT_ROOT/person3/data/splits/screenspot_eval.jsonl $PROJECT_ROOT/data/splits/
```

### 3.2 如果后续拿到原始 Widget Captioning 标注

```bash
python -m src.person3.convert_widget_captioning \
  --annotations $PROJECT_ROOT/raw/widget_captioning/annotations.json \
  --image-root $PROJECT_ROOT/raw/widget_captioning/images \
  --output $PROJECT_ROOT/data/splits/widget_captioning_all.jsonl
```

## 4. 生成第三成员三套训练数据

```bash
mkdir -p $PROJECT_ROOT/data/person3

python -m src.person3.derive_person3_from_qwen \
  --input $PROJECT_ROOT/data/splits/D1-Base_train_qwen.jsonl \
  --output $PROJECT_ROOT/data/person3/p3_direct_train.jsonl \
  --method direct

python -m src.person3.derive_person3_from_qwen \
  --input $PROJECT_ROOT/data/splits/D1-Base_train_qwen.jsonl \
  --output $PROJECT_ROOT/data/person3/p3_region_point_train.jsonl \
  --method region_point

python -m src.person3.derive_person3_from_qwen \
  --input $PROJECT_ROOT/data/splits/D1-Base_train_qwen.jsonl \
  --output $PROJECT_ROOT/data/person3/p3_target_region_point_train.jsonl \
  --method target_region_point
```

验证集 `D1-Base_val.jsonl` 不是 Qwen 训练对话格式，仍使用 `build_qwen_data.py` 生成验证提示：

```bash
python -m src.person3.build_qwen_data \
  --input $PROJECT_ROOT/data/splits/D1-Base_val.jsonl \
  --output $PROJECT_ROOT/data/person3/p3_direct_val.json \
  --method direct

python -m src.person3.build_qwen_data \
  --input $PROJECT_ROOT/data/splits/D1-Base_val.jsonl \
  --output $PROJECT_ROOT/data/person3/p3_region_point_val.json \
  --method region_point

python -m src.person3.build_qwen_data \
  --input $PROJECT_ROOT/data/splits/D1-Base_val.jsonl \
  --output $PROJECT_ROOT/data/person3/p3_target_region_point_val.json \
  --method target_region_point
```

## 5. LoRA 训练

当前仓库已经提供第三成员独立 LoRA 训练入口：

```bash
python -m src.person3.train_lora_qwen3vl
```

第三成员需要分别训练：

```text
P3-1: p3_direct_train.jsonl
P3-2: p3_region_point_train.jsonl
P3-3: p3_target_region_point_train.jsonl
```

输出目录：

```text
$PROJECT_ROOT/checkpoints/person3/p3_direct
$PROJECT_ROOT/checkpoints/person3/p3_region_point
$PROJECT_ROOT/checkpoints/person3/p3_target_region_point
```

要求：

- 三组实验除训练数据输出格式外，其他 LoRA 参数保持一致。
- 使用 FP16。
- 不使用 BF16。
- 不使用 FlashAttention 2。
- 训练日志保存到 `outputs/logs/person3/`。

先用 8 条样本跑通链路：

```bash
cd /home/ma-user/work/gui-project/person3
export PROJECT_ROOT=/home/ma-user/work/gui-project
export MODEL_DIR=$PROJECT_ROOT/models/Qwen3-VL-4B-Instruct
conda activate qwen3vl

bash scripts/train_p3_direct_lora.sh --limit 8 --max-steps 2 --save-steps 2
```

正式训练：

```bash
bash scripts/train_all_person3_lora.sh
```

如需单独重跑某一组，分别执行 `scripts/train_p3_direct_lora.sh`、`scripts/train_p3_region_point_lora.sh`、`scripts/train_p3_target_region_point_lora.sh`。

## 6. 推理

以验证集为例：

```bash
python -m src.person3.infer_qwen3vl \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --adapter $PROJECT_ROOT/checkpoints/person3/p3_region_point \
  --input $PROJECT_ROOT/data/splits/D1-Base_val.jsonl \
  --method region_point \
  --output $PROJECT_ROOT/outputs/predictions/person3/p3_region_point_val.jsonl \
  --limit 100
```

先用 `--limit 20` 或 `--limit 100` 调试，确认输出格式稳定后再跑完整验证集。

## 7. 评测

```bash
python -m src.person3.evaluate \
  --truth $PROJECT_ROOT/data/splits/D1-Base_val.jsonl \
  --pred $PROJECT_ROOT/outputs/predictions/person3/p3_region_point_val.jsonl \
  --out $PROJECT_ROOT/outputs/metrics/person3/p3_region_point_val_metrics.json
```

三组实验都要生成：

```text
metrics.json
predictions.jsonl
case images
```

## 8. 可视化

```bash
python -m src.person3.visualize_cases \
  --truth $PROJECT_ROOT/data/splits/D1-Base_val.jsonl \
  --pred $PROJECT_ROOT/outputs/predictions/person3/p3_region_point_val.jsonl \
  --out-dir $PROJECT_ROOT/outputs/cases/person3/p3_region_point \
  --limit 50
```

## 9. 最终报告需要交付

至少包含：

- 三组实验总表。
- Click Accuracy。
- Region Accuracy。
- Parse Success Rate。
- text/icon 分类结果。
- small/medium/large 分类结果。
- 区域判断错误与精确定位错误数量。
- 10 个成功案例。
- 10 个失败案例。

## 10. 判断标准

如果 `Region -> Point` 或 `Target -> Region -> Point` 准确率更高：

- 说明显式中间结构有助于定位。
- 重点展示相似控件、图标控件案例。

如果两阶段方法没有提升：

- 分析输出更长导致解析失败。
- 分析区域标签粒度是否太粗。
- 分析模型是否因为多任务输出增加了学习难度。

这两种结果都可以写成有效实验结论。
