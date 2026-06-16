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

假设统一训练集为：

```text
$PROJECT_ROOT/data/splits/train_base_v1.jsonl
$PROJECT_ROOT/data/splits/val_base_v1.jsonl
```

生成 Qwen 对话格式：

```bash
cd /home/ma-user/work/gui-project/person3

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

验证集同理生成三个版本。

## 训练

本目录负责生成训练数据和评测代码；具体 LoRA 训练命令使用全组统一的 Qwen3-VL 微调脚本。

训练时分别传入：

```text
p3_direct_train.json
p3_region_point_train.json
p3_target_region_point_train.json
```

建议输出目录：

```text
$PROJECT_ROOT/checkpoints/person3/p3_direct
$PROJECT_ROOT/checkpoints/person3/p3_region_point
$PROJECT_ROOT/checkpoints/person3/p3_target_region_point
```

训练配置必须继承公共基线，只改变输出格式和训练数据。

## 评测

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
  --truth $PROJECT_ROOT/data/splits/val_base_v1.jsonl \
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
  --truth $PROJECT_ROOT/data/splits/val_base_v1.jsonl \
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
