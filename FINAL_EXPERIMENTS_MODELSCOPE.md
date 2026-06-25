# Final Experiments on ModelScope

本文档用于把最终三组实验迁移到 ModelScope 平台运行。

最终实验只保留三组：

| 实验 | 说明 | 是否训练 |
|---|---|---|
| B0 | 原始 Qwen3-VL 零样本直接推理 | 不训练 |
| B1 | SeeClick-Hard 数据 + Point 输出 LoRA | 训练 1 次 |
| Final | B1 + Coarse-to-Fine + Retry | 不额外训练 |

## 1. 目录约定

ModelScope 上建议使用一个持久化项目目录，例如：

```bash
export PROJECT_ROOT=/mnt/workspace/gui-project
export REPO_DIR=$PROJECT_ROOT/person3
export TEAM_REPOS=$PROJECT_ROOT/team_repos
export TEAM1_SPLITS=$TEAM_REPOS/person1/data/splits

export MODEL_DIR=$PROJECT_ROOT/models/Qwen3-VL-4B-Instruct
export HF_HOME=$PROJECT_ROOT/hf_cache
export PY=python
```

如果你创建了自己的虚拟环境，把 `PY` 改成对应 Python：

```bash
export PY=$PROJECT_ROOT/envs/qwen3vl/bin/python
```

旧 ModelArts 路径是 `/home/ma-user/work/gui-project`。迁移到 ModelScope 后不要写死旧路径，统一通过 `PROJECT_ROOT`、`REPO_DIR`、`MODEL_DIR`、`IMAGE_ROOT`、`SCREENSPOT_IMAGE_ROOT` 控制。

## 2. 拉取代码

```bash
mkdir -p "$PROJECT_ROOT" "$TEAM_REPOS"

cd "$PROJECT_ROOT"
git clone https://github.com/dou307/gui-grounding-person3 person3

cd "$TEAM_REPOS"
git clone https://github.com/LuChron/NLP_Person1 person1
git clone https://github.com/RanderDouble/NLPFinal person2
git clone https://github.com/Byting-HYQ/gui-grounding person5
```

4 号代码如果不是 Git 仓库，上传到：

```text
$PROJECT_ROOT/team_repos/person4-code
```

当前最终脚本已经把 4 号 Coarse-to-Fine 和 5 号 Retry 的核心逻辑迁移进 `src/final_experiments/infer_final_qwen3vl.py`，所以正式跑 Final 不需要直接调用 4 号或 5 号原脚本。

## 3. 环境安装

如果 ModelScope 镜像里有 conda：

```bash
cd "$REPO_DIR"
bash scripts/install_env.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen3vl
export PY=python
```

如果你使用平台预装 Python 或自己创建的 venv，只要保证：

```bash
$PY - <<'PY'
import torch, transformers, peft
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("peft:", peft.__version__)
PY
```

能正常输出即可。

## 4. 模型和数据

### 4.1 模型

模型目录需要放在：

```text
$MODEL_DIR
```

也就是：

```text
$PROJECT_ROOT/models/Qwen3-VL-4B-Instruct
```

可以用 `hf download`、ModelScope 下载工具，或直接上传已有模型缓存。只要 `AutoProcessor.from_pretrained($MODEL_DIR)` 和 `Qwen3VLForConditionalGeneration.from_pretrained($MODEL_DIR)` 能加载即可。

### 4.2 训练数据

从 1 号仓库复制最终训练和测试 jsonl：

```bash
cd "$REPO_DIR"
bash scripts/final/prepare_final_data.sh
```

会生成：

```text
$PROJECT_ROOT/data/final/D1-Hard_train_qwen.jsonl
$PROJECT_ROOT/data/final/screenspot_eval.jsonl
```

### 4.3 图片目录

训练 B1 时，`IMAGE_ROOT` 要指向 SeeClick-Hard 训练图片目录。训练样本中常见图片名类似：

```text
51161.jpg
```

如果你沿用之前整理方式，通常是：

```bash
export IMAGE_ROOT=$PROJECT_ROOT/data/rico_imgs/combined
```

推理 B0/B1/Final 时，`IMAGE_ROOT` 要切换为 ScreenSpot 图片目录。推荐：

```bash
export SCREENSPOT_IMAGE_ROOT=$PROJECT_ROOT/data/screenspot_hf/images
export IMAGE_ROOT=$SCREENSPOT_IMAGE_ROOT
```

注意：1 号的 `screenspot_eval.jsonl` 是完整 1272 条版本，和你之前 3 号旧的 881 条文件不同。最终实验必须统一使用 1 号这份 1272 条 ScreenSpot。

图片路径检查：

```bash
cd "$REPO_DIR"
export IMAGE_ROOT=$PROJECT_ROOT/data/rico_imgs/combined
export SCREENSPOT_IMAGE_ROOT=$PROJECT_ROOT/data/screenspot_hf/images
bash scripts/final/check_final_data_images.sh
```

## 5. 训练 B1

```bash
cd "$REPO_DIR"

export IMAGE_ROOT=$PROJECT_ROOT/data/rico_imgs/combined

bash scripts/final/train_final_b1_lora.sh \
  --check-image-limit 20
```

默认训练设置：

```text
MAX_STEPS=500
SAVE_STEPS=250
GRADIENT_ACCUMULATION_STEPS=8
MAX_PIXELS=401408
```

如需改训练步数：

```bash
MAX_STEPS=625 SAVE_STEPS=250 bash scripts/final/train_final_b1_lora.sh
```

输出：

```text
$PROJECT_ROOT/checkpoints/final/b1_hard_point
```

## 6. 跑三组推理

推理前切换到 ScreenSpot 图片目录：

```bash
export IMAGE_ROOT=$PROJECT_ROOT/data/screenspot_hf/images
```

### B0：原始 Qwen3-VL 零样本

```bash
cd "$REPO_DIR"
bash scripts/final/infer_final_b0.sh
```

输出：

```text
$PROJECT_ROOT/outputs/predictions/final/b0_screenspot.jsonl
```

### B1：Hard + Point LoRA

```bash
cd "$REPO_DIR"
bash scripts/final/infer_final_b1.sh
```

输出：

```text
$PROJECT_ROOT/outputs/predictions/final/b1_screenspot.jsonl
```

### Final：B1 + Coarse-to-Fine + Retry

```bash
cd "$REPO_DIR"
bash scripts/final/infer_final_full.sh
```

输出：

```text
$PROJECT_ROOT/outputs/predictions/final/final_c2f_retry_screenspot.jsonl
```

Final 会比 B0/B1 慢很多，因为默认每条样本会执行多次 Coarse-to-Fine：

```text
RETRY_SAMPLES=3
MAX_RETRIES=1
```

调试时建议先跑小样本：

```bash
bash scripts/final/infer_final_full.sh --limit 20
```

如果耗时太长，可以降低重试成本：

```bash
RETRY_SAMPLES=2 MAX_RETRIES=0 bash scripts/final/infer_final_full.sh
```

## 7. 评测和汇总

```bash
cd "$REPO_DIR"
bash scripts/final/evaluate_final_experiments.sh
```

输出：

```text
$PROJECT_ROOT/outputs/metrics/final/b0_screenspot_metrics.json
$PROJECT_ROOT/outputs/metrics/final/b1_screenspot_metrics.json
$PROJECT_ROOT/outputs/metrics/final/final_c2f_retry_screenspot_metrics.json
$PROJECT_ROOT/outputs/metrics/final/final_results_summary.md
$PROJECT_ROOT/outputs/metrics/final/final_results_summary.csv
```

## 8. 需要保存的文件

实验完成后至少保存：

```text
$PROJECT_ROOT/checkpoints/final/b1_hard_point
$PROJECT_ROOT/outputs/predictions/final
$PROJECT_ROOT/outputs/metrics/final
```

建议打包：

```bash
cd "$PROJECT_ROOT"
tar -czf outputs/final_experiments_results_$(date +%Y%m%d_%H%M).tar.gz \
  outputs/predictions/final \
  outputs/metrics/final \
  checkpoints/final/b1_hard_point
```
