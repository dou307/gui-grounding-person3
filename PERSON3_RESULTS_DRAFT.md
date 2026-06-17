# Person3 结果草稿

## 实验设置

成员 3 负责直接定位与结构化两阶段定位方向。任务是单步 GUI 元素定位：输入界面截图和指令，输出点击坐标，坐标统一为 `[0,1000]`。

三组实验只改变输出结构：

| 实验 | 方法 | 输出结构 |
|---|---|---|
| P3-1 | Direct Point | `{"action":"click","x":整数,"y":整数}` |
| P3-2 | Region -> Point | `{"region":"九宫格区域","action":"click","x":整数,"y":整数}` |
| P3-3 | Target -> Region -> Point | `{"target":"目标控件","region":"九宫格区域","action":"click","x":整数,"y":整数}` |

训练设置：

| 项目 | 设置 |
|---|---|
| 基础模型 | Qwen3-VL-4B-Instruct |
| 微调方式 | LoRA |
| 训练数据 | D1-Base 派生训练集，每组 96,354 条 |
| 训练预算 | `max_steps=500` |
| 图像上限 | `max_pixels=401408` |
| 梯度累积 | `gradient_accumulation_steps=8` |
| 验证集 | D1-Base_val 前 1000 条固定子集 |
| 测试集 | ScreenSpot 全量 881 条 |

说明：受单卡 V100 训练和推理时间限制，验证集消融实验使用 D1-Base_val 的前 1000 条固定子集；最终测试在完整 ScreenSpot 881 条上进行。

## 当前结果

| 实验 | 验证样本 | 验证 Click Acc | 验证 Parse | ScreenSpot 样本 | ScreenSpot Click Acc | ScreenSpot Parse |
|---|---:|---:|---:|---:|---:|---:|
| P3-1 Direct Point | 1000 | 0.7580 | 1.0000 | 881 | 0.4472 | 1.0000 |
| P3-2 Region -> Point | 待补 | 待补 | 待补 | 待补 | 待补 | 待补 |
| P3-3 Target -> Region -> Point | 待补 | 待补 | 待补 | 待补 | 待补 | 待补 |

## P3-1 详细结果

### D1-Base_val_1000

| 指标 | 数值 |
|---|---:|
| total | 1000 |
| parsed | 1000 |
| parse_success_rate | 1.0000 |
| click_accuracy | 0.7580 |
| mean_center_distance | 107.4972 |
| point_outside_bbox | 242 |

按目标大小：

| target_size | total | correct | accuracy |
|---|---:|---:|---:|
| large | 367 | 289 | 0.7875 |
| medium | 548 | 416 | 0.7591 |
| small | 85 | 53 | 0.6235 |

### ScreenSpot

| 指标 | 数值 |
|---|---:|
| total | 881 |
| parsed | 881 |
| parse_success_rate | 1.0000 |
| click_accuracy | 0.4472 |
| mean_center_distance | 175.1444 |
| point_outside_bbox | 487 |

按平台：

| platform | total | correct | accuracy |
|---|---:|---:|---:|
| desktop | 282 | 2 | 0.0071 |
| mobile | 163 | 3 | 0.0184 |
| web | 436 | 389 | 0.8922 |

按目标类型：

| target_type | total | correct | accuracy |
|---|---:|---:|---:|
| icon | 412 | 179 | 0.4345 |
| text | 469 | 215 | 0.4584 |

## 初步观察

P3-1 在验证子集上的点击准确率为 75.8%，JSON 解析成功率为 100%，说明 Direct Point 输出格式稳定，训练后的模型能稳定生成可解析坐标。

ScreenSpot 上总体点击准确率为 44.72%，但平台差异很大：web 子集达到 89.22%，desktop 和 mobile 子集很低。这可能说明训练数据分布与 ScreenSpot 的 desktop/mobile 截图存在明显域差异，或者图片缩放、界面风格、目标尺寸分布造成泛化困难。后续 P3-2/P3-3 需要重点观察结构化输出是否能改善这些域外子集。

## 后续一键评测命令

P3-2 训练完成后：

```bash
cd /home/ma-user/work/gui-project/person3
export PROJECT_ROOT=/home/ma-user/work/gui-project
export PY=$PROJECT_ROOT/envs/qwen3vl/bin/python
export MODEL_DIR=$PROJECT_ROOT/models/Qwen3-VL-4B-Instruct
export VAL_IMAGE_ROOT=$PROJECT_ROOT/data/rico_imgs/combined
export SCREENSPOT_IMAGE_ROOT=$PROJECT_ROOT/data/screenspot_hf/images

bash scripts/eval_person3_method.sh region_point
```

P3-3 训练完成后：

```bash
bash scripts/eval_person3_method.sh target_region_point
```

P3-1 如需复跑：

```bash
bash scripts/eval_person3_method.sh direct
```

## 三组完成后的汇总与打包

三组训练、验证和 ScreenSpot 测试都完成后，生成总表：

```bash
$PY -m src.person3.summarize_results \
  --metrics-dir "$PROJECT_ROOT/outputs/metrics/person3" \
  --out-md "$PROJECT_ROOT/outputs/metrics/person3/person3_results_summary.md" \
  --out-csv "$PROJECT_ROOT/outputs/metrics/person3/person3_results_summary.csv"
```

生成三组预测对比案例：

```bash
$PY -m src.person3.compare_predictions \
  --truth "$PROJECT_ROOT/data/person3/D1-Base_val_1000.jsonl" \
  --pred-dir "$PROJECT_ROOT/outputs/predictions/person3" \
  --split val_1000 \
  --out-dir "$PROJECT_ROOT/outputs/analysis/person3"

$PY -m src.person3.compare_predictions \
  --truth "$PROJECT_ROOT/data/splits/screenspot_eval.jsonl" \
  --pred-dir "$PROJECT_ROOT/outputs/predictions/person3" \
  --split screenspot \
  --out-dir "$PROJECT_ROOT/outputs/analysis/person3"
```

最终打包：

```bash
bash scripts/package_person3_results.sh
```

如果云端文件浏览器仍有 100MB 下载限制，继续用 `split -b 90M` 分片下载。
