# 第三成员完整执行计划

## 0. 已确定的公共数据方案

本项目公共数据采用以下配置：

| 类型 | 数据源 | 用途 | 文件名 |
|---|---|---|---|
| 训练集 | SeeClick GUI grounding 数据 | LoRA 训练 | `train_base_v1.jsonl` |
| 验证集 | SeeClick 中固定划分 10% | 方法选择和消融对比 | `val_base_v1.jsonl` |
| 测试集 | ScreenSpot test split | 最终统一测试 | `screenspot_test.jsonl` |

规则：

- ScreenSpot 只用于最终测试，不参与训练和调参。
- RICO 第一阶段不加入公共基线，避免额外转换和数据噪声。
- 所有 bbox 和 point 统一为 `[0,1000]` 坐标。
- 第三成员的所有实验必须与公共 `Direct Point` 基线比较。

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

### 3.1 转换 SeeClick 标注

SeeClick 数据每条样本通常包含：

```json
{
  "img_filename": "...png",
  "instruction": "...",
  "bbox": [left, top, right, bottom]
}
```

其中 bbox 是 `[0,1]` 相对坐标。

转换命令示例：

```bash
export PROJECT_ROOT=/home/ma-user/work/gui-project
cd $PROJECT_ROOT/person3

python -m src.person3.convert_seeclick \
  --annotations $PROJECT_ROOT/raw/seeclick/annotations.json \
  --image-root $PROJECT_ROOT/raw/seeclick/images \
  --output $PROJECT_ROOT/data/splits/seeclick_all.jsonl \
  --source seeclick
```

### 3.2 划分训练集和验证集

```bash
python -m src.person3.split_jsonl \
  --input $PROJECT_ROOT/data/splits/seeclick_all.jsonl \
  --train-output $PROJECT_ROOT/data/splits/train_base_v1.jsonl \
  --val-output $PROJECT_ROOT/data/splits/val_base_v1.jsonl \
  --val-ratio 0.1 \
  --seed 42
```

### 3.3 准备 ScreenSpot 测试集

```bash
python -m src.person3.prepare_screenspot \
  --output-jsonl $PROJECT_ROOT/data/splits/screenspot_test.jsonl \
  --image-dir $PROJECT_ROOT/data/screenspot/images
```

## 4. 生成第三成员三套训练数据

```bash
mkdir -p $PROJECT_ROOT/data/person3

python -m src.person3.build_qwen_data \
  --input $PROJECT_ROOT/data/splits/train_base_v1.jsonl \
  --output $PROJECT_ROOT/data/person3/p3_direct_train.json \
  --method direct

python -m src.person3.build_qwen_data \
  --input $PROJECT_ROOT/data/splits/train_base_v1.jsonl \
  --output $PROJECT_ROOT/data/person3/p3_region_point_train.json \
  --method region_point

python -m src.person3.build_qwen_data \
  --input $PROJECT_ROOT/data/splits/train_base_v1.jsonl \
  --output $PROJECT_ROOT/data/person3/p3_target_region_point_train.json \
  --method target_region_point
```

验证集同理：

```bash
python -m src.person3.build_qwen_data \
  --input $PROJECT_ROOT/data/splits/val_base_v1.jsonl \
  --output $PROJECT_ROOT/data/person3/p3_direct_val.json \
  --method direct

python -m src.person3.build_qwen_data \
  --input $PROJECT_ROOT/data/splits/val_base_v1.jsonl \
  --output $PROJECT_ROOT/data/person3/p3_region_point_val.json \
  --method region_point

python -m src.person3.build_qwen_data \
  --input $PROJECT_ROOT/data/splits/val_base_v1.jsonl \
  --output $PROJECT_ROOT/data/person3/p3_target_region_point_val.json \
  --method target_region_point
```

## 5. LoRA 训练

训练脚本使用全组统一的 Qwen3-VL 微调入口。第三成员需要分别训练：

```text
P3-1: p3_direct_train.json
P3-2: p3_region_point_train.json
P3-3: p3_target_region_point_train.json
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

## 6. 推理

以验证集为例：

```bash
python -m src.person3.infer_qwen3vl \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --adapter $PROJECT_ROOT/checkpoints/person3/p3_region_point \
  --input $PROJECT_ROOT/data/splits/val_base_v1.jsonl \
  --method region_point \
  --output $PROJECT_ROOT/outputs/predictions/person3/p3_region_point_val.jsonl \
  --limit 100
```

先用 `--limit 20` 或 `--limit 100` 调试，确认输出格式稳定后再跑完整验证集。

## 7. 评测

```bash
python -m src.person3.evaluate \
  --truth $PROJECT_ROOT/data/splits/val_base_v1.jsonl \
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
  --truth $PROJECT_ROOT/data/splits/val_base_v1.jsonl \
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
