# 第三人实验：直接定位与结构化两阶段定位

## 目标

本实验研究：

> 在 GUI 单步元素定位任务中，先预测粗区域或目标语义，再预测点击坐标，是否能减少相似控件混淆并提高 Click Accuracy。

对应三组实验：

| 实验编号 | 方法 | 输出 |
|---|---|---|
| P3-1 | Direct Point | `{"action":"click","x":947,"y":69}` |
| P3-2 | Region -> Point | `{"region":"top-right","action":"click","x":947,"y":69}` |
| P3-3 | Target -> Region -> Point | `{"target":"搜索按钮","region":"top-right","action":"click","x":947,"y":69}` |

所有坐标统一使用 `[0,1000]`。

## 输入数据格式

本目录脚本要求输入 JSONL，每行一条样本：

```json
{
  "id": "sample_000001",
  "image": "images/sample_000001.png",
  "instruction": "点击右上角的搜索按钮",
  "bbox_pixel": [1420, 36, 1500, 112],
  "point_1000": [947, 69],
  "platform": "web",
  "target_type": "icon",
  "source": "widget_captioning"
}
```

如果数据里已经有 `bbox_1000`，脚本会直接使用；如果只有 `bbox_pixel`，需要图片文件存在，用于根据图片宽高转换到 `[0,1000]`。

## 目录结构

```text
gui-grounding-person3/
├── README.md
├── configs/
│   ├── p3_direct.yaml
│   ├── p3_region_point.yaml
│   └── p3_target_region_point.yaml
├── src/
│   └── person3/
│       ├── __init__.py
│       ├── build_qwen_data.py
│       ├── evaluate.py
│       ├── region.py
│       ├── schema.py
│       └── visualize_cases.py
└── scripts/
    └── run_person3_data_eval_demo.sh
```

## 在 ModelArts 上放置位置

建议放入：

```text
/home/ma-user/work/gui-project/person3
```

并设置：

```bash
export PROJECT_ROOT=/home/ma-user/work/gui-project
```

## 生成三种训练数据

当前 1 号已经提供了公共数据：

```text
$PROJECT_ROOT/data/splits/D1-Base_train_qwen.jsonl
$PROJECT_ROOT/data/splits/D1-Base_val.jsonl
$PROJECT_ROOT/data/splits/screenspot_eval.jsonl
```

其中 `D1-Base_train_qwen.jsonl` 已经是 Direct Point 的 Qwen 对话训练格式。第三成员从该文件派生三种训练格式：

```bash
cd /home/ma-user/work/gui-project/person3

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

验证集 `D1-Base_val.jsonl` 使用 `build_qwen_data.py` 生成三个版本。

## 训练

本仓库提供独立 LoRA 训练入口：

```text
src/person3/train_lora_qwen3vl.py
```

默认配置按单卡 V100 32GB 保守设置：FP16、SDPA、LoRA rank 16、batch size 1、gradient accumulation 16、1 epoch。

先设置环境变量：

```bash
cd /home/ma-user/work/gui-project/person3
export PROJECT_ROOT=/home/ma-user/work/gui-project
export MODEL_DIR=$PROJECT_ROOT/models/Qwen3-VL-4B-Instruct
conda activate qwen3vl
```

先用少量样本检查训练链路：

```bash
bash scripts/train_p3_direct_lora.sh --limit 8 --max-steps 2 --save-steps 2
```

正式训练三组实验：

```bash
bash scripts/train_all_person3_lora.sh
```

如果需要单独重跑某一组，也可以分别执行 `scripts/train_p3_direct_lora.sh`、`scripts/train_p3_region_point_lora.sh`、`scripts/train_p3_target_region_point_lora.sh`。

输出目录：

```text
$PROJECT_ROOT/checkpoints/person3/p3_direct
$PROJECT_ROOT/checkpoints/person3/p3_region_point
$PROJECT_ROOT/checkpoints/person3/p3_target_region_point
```

训练日志：

```text
$PROJECT_ROOT/outputs/logs/person3/p3_direct
$PROJECT_ROOT/outputs/logs/person3/p3_region_point
$PROJECT_ROOT/outputs/logs/person3/p3_target_region_point
```

如显存不足，降低图像 token 或增大梯度累积：

```bash
bash scripts/train_p3_direct_lora.sh \
  --max-pixels 401408 \
  --gradient-accumulation-steps 32
```

## 评测

验证集推理时建议使用批量推理，提高 GPU 利用率：

```bash
python -m src.person3.infer_qwen3vl \
  --model $MODEL_DIR \
  --adapter $PROJECT_ROOT/checkpoints/person3/p3_direct \
  --input $PROJECT_ROOT/data/splits/D1-Base_val.jsonl \
  --method direct \
  --output $PROJECT_ROOT/outputs/predictions/person3/p3_direct_val.jsonl \
  --batch-size 4 \
  --max-new-tokens 64 \
  --max-pixels 401408
```

如果显存允许，可以把 `--batch-size 4` 提高到 `8`；如显存不足则降回 `1` 或 `2`。

预测文件使用 JSONL，每行推荐保存：

```json
{
  "id": "sample_000001",
  "raw_output": "{\"region\":\"top-right\",\"action\":\"click\",\"x\":947,\"y\":69}"
}
```

运行评测：

```bash
python -m src.person3.evaluate \
  --truth $PROJECT_ROOT/data/splits/D1-Base_val.jsonl \
  --pred $PROJECT_ROOT/outputs/predictions/person3/p3_region_point_val.jsonl \
  --out $PROJECT_ROOT/outputs/metrics/person3/p3_region_point_val_metrics.json
```

指标包括：

- `click_accuracy`
- `parse_success_rate`
- `region_accuracy`
- `region_wrong_count`
- `precision_wrong_count`
- 按 `platform`、`target_type`、目标大小分类的 Click Accuracy

## 可视化案例

```bash
python -m src.person3.visualize_cases \
  --truth $PROJECT_ROOT/data/splits/D1-Base_val.jsonl \
  --pred $PROJECT_ROOT/outputs/predictions/person3/p3_region_point_val.jsonl \
  --out-dir $PROJECT_ROOT/outputs/cases/person3/p3_region_point \
  --limit 30
```

生成图片会画出：

- 绿色框：真实目标框。
- 红点：模型预测点击点。
- 标题：样本 id、指令、真实区域、预测区域、是否点击正确。

## 需要写进报告的结论模板

1. Direct Point 是公共比较基线。
2. Region -> Point 增加了可解释中间变量，可以区分“区域判断错误”和“精确坐标错误”。
3. Target -> Region -> Point 进一步显式建模目标语义，重点观察它对相似控件、图标控件的提升。
4. 如果两阶段方法总准确率不升反降，也要分析是否是输出更长导致解析失败、训练目标更复杂或区域标签噪声带来的影响。
