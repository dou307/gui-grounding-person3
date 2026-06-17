# Person3 实验续接记录

本文记录 3 号方向当前实验状态、已经踩过的坑、固定路径和后续命令。后续新开 Notebook 或新对话时，优先读这个文件。

## 1. 实验定位

课程大作业当前走的是单步 GUI 元素定位，不是完整 Visual Agent。

任务形式：

```text
输入：界面截图 + 指令
输出：点击位置，坐标统一为 [0,1000]
```

3 号方向研究问题：

```text
显式加入 region / target 这种中间结构，是否比直接预测点击点更好？
```

三组实验：

```text
P3-1 Direct Point
输出：{"action":"click","x":69,"y":71}
角色：3 号实验内部 baseline

P3-2 Region -> Point
输出：{"region":"top-left","action":"click","x":69,"y":71}
作用：先判断九宫格粗区域，再预测点击点

P3-3 Target -> Region -> Point
输出：{"target":"menu option","region":"top-left","action":"click","x":69,"y":71}
作用：先预测目标语义，再预测区域和点击点
```

P3-1 是 baseline。为了公平，三组应保持相同模型、训练数据来源、训练预算和评测样本，只改变输出结构。

## 2. 数据方案

公共训练集：

```text
Widget Captioning，来源 OS-Atlas-data
```

公共验证集：

```text
从 Widget Captioning 公共训练数据中固定划分
```

公共测试集：

```text
ScreenSpot test split
```

ScreenSpot 只用于最终测试，不参与训练和调参。

当前 ModelArts 关键数据文件：

```text
$PROJECT_ROOT/data/splits/D1-Base_train_qwen.jsonl
$PROJECT_ROOT/data/splits/D1-Base_val.jsonl
$PROJECT_ROOT/data/splits/screenspot_eval.jsonl
```

实际规模：

```text
D1-Base_train_qwen.jsonl: 96354 条
D1-Base_val.jsonl: 5071 条
screenspot_eval.jsonl: 881 条
```

已经生成的 3 号训练数据：

```text
$PROJECT_ROOT/data/person3/p3_direct_train.jsonl
$PROJECT_ROOT/data/person3/p3_region_point_train.jsonl
$PROJECT_ROOT/data/person3/p3_target_region_point_train.jsonl
```

每组训练数据：

```text
96354 条
```

图片目录：

```text
$PROJECT_ROOT/data/rico_imgs/combined
```

已确认示例图片存在：

```text
$PROJECT_ROOT/data/rico_imgs/combined/10554.jpg
$PROJECT_ROOT/data/rico_imgs/combined/48215.jpg
```

注意：三份 `.jsonl` 只是标注/对话数据，不包含图片本体。Qwen3-VL 训练和推理必须能读到真实截图。

## 3. ModelArts 固定路径

统一使用：

```bash
export PROJECT_ROOT=/home/ma-user/work/gui-project
export MODEL_DIR=$PROJECT_ROOT/models/Qwen3-VL-4B-Instruct
export IMAGE_ROOT=$PROJECT_ROOT/data/rico_imgs/combined
export HF_HOME=$PROJECT_ROOT/hf_cache
```

仓库路径：

```text
/home/ma-user/work/gui-project/person3
```

模型路径：

```text
/home/ma-user/work/gui-project/models/Qwen3-VL-4B-Instruct
```

P3-1 checkpoint：

```text
/home/ma-user/work/gui-project/checkpoints/person3/p3_direct
```

关键文件：

```text
adapter_config.json
adapter_model.safetensors
```

## 4. 环境状态和处理方式

ModelArts 重启后，named env `qwen3vl` 曾失效：

```text
Could not find conda environment: qwen3vl
```

后来又出现 named env 残留坏状态：

```text
Python 3.9
ModuleNotFoundError: No module named 'torch'
```

因此后续建议不要再依赖 named env：

```bash
conda activate qwen3vl
```

改用持久化路径环境：

```text
/home/ma-user/work/gui-project/envs/qwen3vl
```

但当前 shell 里 `python` 命令可能被外层环境劫持，即使 `CONDA_PREFIX` 正确，`python -V` 仍可能显示 Python 3.9。

所以统一使用绝对路径变量：

```bash
export PROJECT_ROOT=/home/ma-user/work/gui-project
export PY=$PROJECT_ROOT/envs/qwen3vl/bin/python
```

检查：

```bash
$PY -V
```

已确认路径环境里的 Python 是：

```text
Python 3.10.20
```

依赖安装命令：

```bash
export PROJECT_ROOT=/home/ma-user/work/gui-project
export PY=$PROJECT_ROOT/envs/qwen3vl/bin/python
export WHEELHOUSE=$PROJECT_ROOT/wheelhouse

cd $PROJECT_ROOT/person3

$PY -m pip install --upgrade pip

$PY - <<'PY' > .requirements-local.txt
from pathlib import Path

for line in Path("requirements-lock.txt").read_text(encoding="utf-8").splitlines():
    if line.startswith("torch @ "):
        print("torch==2.6.0+cu118")
    elif line.startswith("torchvision @ "):
        print("torchvision==0.21.0+cu118")
    elif line.strip():
        print(line)
PY

$PY -m pip install \
  --no-index \
  --find-links "$WHEELHOUSE" \
  -r .requirements-local.txt
```

如果出现：

```text
No space left on device
```

先清理空间。

## 5. 空间清理记录

曾出现 `/home/ma-user/work` 空间占满：

```text
49G used, only 758M available
```

当时目录占用：

```text
data       24G
models     8.3G
raw        6.1G
hf_cache   3.8G
wheelhouse 3.0G
envs       2.8G
outputs    817M
```

可以删除：

```bash
rm -rf "$PROJECT_ROOT/raw"
rm -f "$PROJECT_ROOT/outputs/person3_shutdown_notes_"*.tar.gz
rm -rf "$PROJECT_ROOT/hf_cache"
rm -rf "$PROJECT_ROOT/envs/qwen3vl"
conda env remove -n qwen3vl -y || true
conda clean -a -y
```

不要删：

```text
$PROJECT_ROOT/data/rico_imgs/combined
$PROJECT_ROOT/models/Qwen3-VL-4B-Instruct
$PROJECT_ROOT/checkpoints/person3/p3_direct
$PROJECT_ROOT/data/splits
$PROJECT_ROOT/data/person3
$PROJECT_ROOT/wheelhouse
```

## 6. 已完成事项

P3-1 Direct Point 已完成 LoRA 训练。

训练输出目录：

```text
$PROJECT_ROOT/checkpoints/person3/p3_direct
```

已确认存在：

```text
adapter_config.json
adapter_model.safetensors
```

P3-1 训练命令使用或推荐记录：

```bash
bash scripts/train_p3_direct_lora.sh \
  --image-root "$IMAGE_ROOT" \
  --max-steps 500 \
  --save-steps 250 \
  --max-pixels 401408 \
  --gradient-accumulation-steps 8
```

说明：

```text
使用 max_steps=500 是因为完整 1 epoch 约 6023 step，三组全跑成本过高。
三组实验应统一使用相同 max_steps，以保证公平。
```

报告可写：

```text
受单卡 V100 训练时间限制，第三成员三组实验统一采用 LoRA 微调 max_steps=500，而不是完整 1 epoch。三组保持相同训练预算，用于公平比较 Direct Point、Region -> Point、Target -> Region -> Point 三种输出结构。
```

## 7. 推理脚本更新

`src/person3/infer_qwen3vl.py` 已更新：

```text
支持 --batch-size 批量推理
支持 --resume 断点续跑
每个 batch 推理后立即追加写入 JSONL，避免 Ctrl+C 后全部丢失
批量推理使用 left padding
```

旧版本在 Ctrl+C 后没有保存前面约 2000 条推理结果，因为旧脚本最后统一写文件。

当前之后如果中断，最多丢一个 batch。

## 8. 当前验证策略

完整验证集有 5071 条，推理时间较长。

当前决定：

```text
验证集消融对比使用固定前 1000 条
ScreenSpot 最终测试仍跑完整 881 条
```

报告可写：

```text
受单卡推理时间限制，验证集消融实验使用 D1-Base_val 的前 1000 条固定子集；最终测试在完整 ScreenSpot 881 条上进行。
```

生成 1000 条验证子集：

```bash
mkdir -p "$PROJECT_ROOT/data/person3"

head -n 1000 "$PROJECT_ROOT/data/splits/D1-Base_val.jsonl" \
  > "$PROJECT_ROOT/data/person3/D1-Base_val_1000.jsonl"
```

P3-1 1000 条验证推理：

```bash
$PY -m src.person3.infer_qwen3vl \
  --model "$MODEL_DIR" \
  --adapter "$PROJECT_ROOT/checkpoints/person3/p3_direct" \
  --input "$PROJECT_ROOT/data/person3/D1-Base_val_1000.jsonl" \
  --method direct \
  --output "$PROJECT_ROOT/outputs/predictions/person3/p3_direct_val_1000.jsonl" \
  --batch-size 4 \
  --max-new-tokens 64 \
  --max-pixels 401408 \
  --resume
```

确认条数：

```bash
wc -l "$PROJECT_ROOT/outputs/predictions/person3/p3_direct_val_1000.jsonl"
```

应为：

```text
1000
```

评测：

```bash
$PY -m src.person3.evaluate \
  --truth "$PROJECT_ROOT/data/person3/D1-Base_val_1000.jsonl" \
  --pred "$PROJECT_ROOT/outputs/predictions/person3/p3_direct_val_1000.jsonl" \
  --out "$PROJECT_ROOT/outputs/metrics/person3/p3_direct_val_1000_metrics.json"
```

查看指标：

```bash
cat "$PROJECT_ROOT/outputs/metrics/person3/p3_direct_val_1000_metrics.json"
```

可视化：

```bash
$PY -m src.person3.visualize_cases \
  --truth "$PROJECT_ROOT/data/person3/D1-Base_val_1000.jsonl" \
  --pred "$PROJECT_ROOT/outputs/predictions/person3/p3_direct_val_1000.jsonl" \
  --out-dir "$PROJECT_ROOT/outputs/cases/person3/p3_direct_val_1000" \
  --limit 50
```

## 9. P3-1 ScreenSpot 测试命令

ScreenSpot 共 881 条，建议完整跑。

推理：

```bash
$PY -m src.person3.infer_qwen3vl \
  --model "$MODEL_DIR" \
  --adapter "$PROJECT_ROOT/checkpoints/person3/p3_direct" \
  --input "$PROJECT_ROOT/data/splits/screenspot_eval.jsonl" \
  --method direct \
  --output "$PROJECT_ROOT/outputs/predictions/person3/p3_direct_screenspot.jsonl" \
  --batch-size 4 \
  --max-new-tokens 64 \
  --max-pixels 401408 \
  --resume
```

确认条数：

```bash
wc -l "$PROJECT_ROOT/outputs/predictions/person3/p3_direct_screenspot.jsonl"
wc -l "$PROJECT_ROOT/data/splits/screenspot_eval.jsonl"
```

评测：

```bash
$PY -m src.person3.evaluate \
  --truth "$PROJECT_ROOT/data/splits/screenspot_eval.jsonl" \
  --pred "$PROJECT_ROOT/outputs/predictions/person3/p3_direct_screenspot.jsonl" \
  --out "$PROJECT_ROOT/outputs/metrics/person3/p3_direct_screenspot_metrics.json"
```

可视化：

```bash
$PY -m src.person3.visualize_cases \
  --truth "$PROJECT_ROOT/data/splits/screenspot_eval.jsonl" \
  --pred "$PROJECT_ROOT/outputs/predictions/person3/p3_direct_screenspot.jsonl" \
  --out-dir "$PROJECT_ROOT/outputs/cases/person3/p3_direct_screenspot" \
  --limit 50
```

如果 ScreenSpot 报图片找不到：

```bash
head -n 1 "$PROJECT_ROOT/data/splits/screenspot_eval.jsonl"
find "$PROJECT_ROOT" -type f \( -name "*.jpg" -o -name "*.png" \) | head -n 50
```

## 10. 后续 P3-2 / P3-3 训练命令

P3-2：

```bash
bash scripts/train_p3_region_point_lora.sh \
  --image-root "$IMAGE_ROOT" \
  --max-steps 500 \
  --save-steps 250 \
  --max-pixels 401408 \
  --gradient-accumulation-steps 8
```

P3-3：

```bash
bash scripts/train_p3_target_region_point_lora.sh \
  --image-root "$IMAGE_ROOT" \
  --max-steps 500 \
  --save-steps 250 \
  --max-pixels 401408 \
  --gradient-accumulation-steps 8
```

P3-2 验证 1000 条：

```bash
$PY -m src.person3.infer_qwen3vl \
  --model "$MODEL_DIR" \
  --adapter "$PROJECT_ROOT/checkpoints/person3/p3_region_point" \
  --input "$PROJECT_ROOT/data/person3/D1-Base_val_1000.jsonl" \
  --method region_point \
  --output "$PROJECT_ROOT/outputs/predictions/person3/p3_region_point_val_1000.jsonl" \
  --batch-size 4 \
  --max-new-tokens 64 \
  --max-pixels 401408 \
  --resume
```

P3-3 验证 1000 条：

```bash
$PY -m src.person3.infer_qwen3vl \
  --model "$MODEL_DIR" \
  --adapter "$PROJECT_ROOT/checkpoints/person3/p3_target_region_point" \
  --input "$PROJECT_ROOT/data/person3/D1-Base_val_1000.jsonl" \
  --method target_region_point \
  --output "$PROJECT_ROOT/outputs/predictions/person3/p3_target_region_point_val_1000.jsonl" \
  --batch-size 4 \
  --max-new-tokens 96 \
  --max-pixels 401408 \
  --resume
```

P3-3 输出更长，所以 `--max-new-tokens` 使用 96。

## 11. 仓库关键提交

已经推到 GitHub 的关键提交：

```text
ba7d0a8 Add Qwen3-VL LoRA training scripts
db217bc Improve training image path handling
98fb471 Check training images before loading model
df9e62c Use IMAGE_ROOT when resolving images
4ce8bae Add batched Qwen3-VL inference
1038ba5 Use left padding for batched inference
51b0ab2 Write inference predictions incrementally
9bbb95e Recreate qwen env with Python 3.10
c5ad51d Fix conda activation in install script
```

## 12. 新会话恢复模板

如果后续新开 Notebook 或新对话，从这里开始：

```bash
cd /home/ma-user/work/gui-project/person3
git pull

export PROJECT_ROOT=/home/ma-user/work/gui-project
export PY=$PROJECT_ROOT/envs/qwen3vl/bin/python
export MODEL_DIR=$PROJECT_ROOT/models/Qwen3-VL-4B-Instruct
export IMAGE_ROOT=$PROJECT_ROOT/data/rico_imgs/combined
export HF_HOME=$PROJECT_ROOT/hf_cache

$PY -V

$PY - <<'PY'
import torch, transformers, peft
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("transformers:", transformers.__version__)
print("peft:", peft.__version__)
PY
```

如果 `$PY` 检查通过，继续 P3-1 1000 条验证或 ScreenSpot 测试。

## 13. 后置自动化脚本

P3-2 或 P3-3 训练完成后，一键跑验证、测试、评测和可视化：

```bash
bash scripts/eval_person3_method.sh region_point
bash scripts/eval_person3_method.sh target_region_point
```

三组结果都完成后生成汇总表：

```bash
$PY -m src.person3.summarize_results \
  --metrics-dir "$PROJECT_ROOT/outputs/metrics/person3" \
  --out-md "$PROJECT_ROOT/outputs/metrics/person3/person3_results_summary.md" \
  --out-csv "$PROJECT_ROOT/outputs/metrics/person3/person3_results_summary.csv"
```

三组预测都完成后生成对比案例：

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

最终交付打包：

```bash
bash scripts/package_person3_results.sh
```

## 14. 已知 ScreenSpot 结果

P3-1 Direct Point:

```text
total = 881
parse_success_rate = 1.0000
click_accuracy = 0.4472
```

P3-2 Region -> Point:

```text
total = 881
parse_success_rate = 1.0000
click_accuracy = 0.4472
region_accuracy = 0.5108
region_wrong_count = 431
precision_wrong_count = 178
```

P3-3 Target -> Region -> Point:

```text
total = 881
parsed = 880
parse_success_rate = 0.9989
click_accuracy = 0.4291
click_accuracy_on_parsed = 0.4295
region_accuracy = 0.5000
region_wrong_count = 440
precision_wrong_count = 183
mean_center_distance = 183.7717
```

当前 ScreenSpot 初步结论：P3-2 与 P3-1 的最终点击准确率持平，但提供 region 错误分解；P3-3 加入 target 后准确率下降到 0.4291，说明更复杂的结构化输出未必带来收益。
