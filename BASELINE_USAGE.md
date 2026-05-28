# Baseline Training and Evaluation Usage

本文档记录当前 `SimpleCNN1D` baseline 的训练和评估脚本用法。

当前只说明已有脚本行为，不引入 VMD、小波、滤波、智能体或工具筛选。

## 训练脚本

训练脚本路径：

- `scripts/train_baseline.py`

默认训练命令：

```bash
python scripts/train_baseline.py
```

默认行为：

- 使用完整 `data/splits/train_indices.npy` 训练。
- 使用完整 `data/splits/val_indices.npy` 验证并选择 best checkpoint。
- 使用 `CrossEntropyLoss`。
- 使用 `Adam` 优化器。
- `--device auto` 时自动选择 `cuda` 或 `cpu`。
- 保存 best checkpoint 到 `checkpoints/simple_cnn_best.pt`。
- 保存训练日志到 `results/baseline/train_log.csv`。

## train_baseline.py 参数支持情况

| 参数 | 当前是否支持 | 说明 |
| --- | --- | --- |
| `--epochs` | 支持 | 训练轮数，默认 `20`。 |
| `--batch-size` | 支持 | batch size，默认 `256`。 |
| `--learning-rate` | 支持 | Adam 学习率，默认 `1e-3`。 |
| `--weight-decay` | 支持 | Adam weight decay，默认 `0.0`。 |
| `--device` | 支持 | 可选 `auto`、`cpu`、`cuda`，默认 `auto`。 |
| `--max-train-samples` | 支持 | 默认 `None`，设置后用固定 seed 从 train split 随机抽样。 |
| `--max-val-samples` | 支持 | 默认 `None`，设置后用固定 seed 从 val split 随机抽样。 |
| `--num-workers` | 支持 | DataLoader worker 数，默认 `0`。 |
| `--checkpoint-path` | 支持 | best checkpoint 保存路径，默认 `checkpoints/simple_cnn_best.pt`。 |
| `--log-path` | 暂不支持 | 当前通过 `--results-dir` 间接控制日志目录，日志文件名固定为 `train_log.csv`。 |

当前训练脚本还支持以下额外参数：

| 参数 | 说明 |
| --- | --- |
| `--data-path` | 原始 RadioML pkl 数据路径，默认 `data/raw/RML2016.10a_dict.pkl`。 |
| `--split-meta-path` | split 元数据路径，默认 `data/splits/split_meta.json`。 |
| `--train-indices` | train indices 路径，默认 `data/splits/train_indices.npy`。 |
| `--val-indices` | val indices 路径，默认 `data/splits/val_indices.npy`。 |
| `--results-dir` | baseline 结果目录，默认 `results/baseline`。 |
| `--dropout` | SimpleCNN1D dropout，默认 `0.3`。 |
| `--seed` | 随机种子，默认 `42`。 |
| `--normalize` | 是否对单个 IQ 样本做均值方差归一化，默认不启用。 |

## 训练示例

完整训练：

```bash
python scripts/train_baseline.py --epochs 20 --batch-size 256
```

指定 CPU：

```bash
python scripts/train_baseline.py --device cpu --epochs 20
```

指定 CUDA：

```bash
python scripts/train_baseline.py --device cuda --epochs 20
```

如果指定 `--device cuda` 但当前环境不可用，脚本会直接报错。

快速 smoke test：

```bash
python scripts/train_baseline.py ^
  --device cpu ^
  --epochs 1 ^
  --batch-size 8 ^
  --max-train-samples 16 ^
  --max-val-samples 16 ^
  --checkpoint-path checkpoints/smoke_simple_cnn_best.pt ^
  --results-dir results/baseline_smoke
```

Linux 或 macOS shell 可写成：

```bash
python scripts/train_baseline.py \
  --device cpu \
  --epochs 1 \
  --batch-size 8 \
  --max-train-samples 16 \
  --max-val-samples 16 \
  --checkpoint-path checkpoints/smoke_simple_cnn_best.pt \
  --results-dir results/baseline_smoke
```

抽样说明：

- 默认不抽样，使用完整 train/val split。
- 设置 `--max-train-samples` 后，训练集会用 `--seed` 固定随机抽样。
- 设置 `--max-val-samples` 后，验证集会用 `--seed + 1` 固定随机抽样。
- 抽样只作用于运行时 Dataset，不会修改或重新生成 split indices 文件。

## 评估脚本

评估脚本路径：

- `scripts/evaluate_baseline.py`

默认评估命令：

```bash
python scripts/evaluate_baseline.py
```

默认行为：

- 加载 `checkpoints/simple_cnn_best.pt`。
- 默认评估 `val` 和 `test`。
- 保存结果到 `results/baseline/`。

## evaluate_baseline.py 支持参数

| 参数 | 说明 |
| --- | --- |
| `--data-path` | 原始 RadioML pkl 数据路径，默认 `data/raw/RML2016.10a_dict.pkl`。 |
| `--split-meta-path` | split 元数据路径，默认 `data/splits/split_meta.json`。 |
| `--checkpoint-path` | checkpoint 路径，默认 `checkpoints/simple_cnn_best.pt`。 |
| `--results-dir` | 评估结果保存目录，默认 `results/baseline`。 |
| `--batch-size` | 评估 batch size，默认 `512`。 |
| `--num-workers` | DataLoader worker 数，默认 `0`。 |
| `--normalize` | 是否使用与训练一致的单样本归一化。 |
| `--splits` | 指定评估 split，可选 `train`、`val`、`test`，默认 `val test`。 |
| `--train-indices` | train indices 路径。 |
| `--val-indices` | val indices 路径。 |
| `--test-indices` | test indices 路径。 |

示例：

```bash
python scripts/evaluate_baseline.py --splits val test
```

只评估 test：

```bash
python scripts/evaluate_baseline.py --splits test
```

指定 checkpoint：

```bash
python scripts/evaluate_baseline.py --checkpoint-path checkpoints/simple_cnn_best.pt
```

## 评估输出

默认保存目录：

- `results/baseline/`

主要输出文件：

- `val_predictions.csv`
- `test_predictions.csv`
- `val_metrics_overall.json`
- `test_metrics_overall.json`
- `val_accuracy_by_snr.csv`
- `test_accuracy_by_snr.csv`
- `val_accuracy_by_modulation.csv`
- `test_accuracy_by_modulation.csv`
- `val_snr_accuracy_curve.csv`
- `test_snr_accuracy_curve.csv`
- `metrics_summary.json`

逐样本预测 CSV 字段：

- `sample_id`
- `true_label`
- `true_modulation`
- `true_snr`
- `pred_label`
- `pred_modulation`
- `correct`
- `top1_confidence`
- `top2_confidence`
- `confidence_margin`
- `entropy`

## 推荐运行顺序

1. 训练 baseline：

```bash
python scripts/train_baseline.py --epochs 20 --batch-size 256
```

2. 评估 val/test：

```bash
python scripts/evaluate_baseline.py --splits val test
```

3. 查看结果：

```text
results/baseline/
```

## 注意事项

- 如果训练时使用了 `--normalize`，评估时也应使用 `--normalize`。
- `--device auto` 会自动优先使用 CUDA。
- `--max-train-samples` 和 `--max-val-samples` 只用于快速调试或 smoke test，不建议用于正式 baseline 结果。
- 当前评估脚本暂未加入 `--device` 或抽样参数。
- 当前训练脚本不支持直接指定 `--log-path`，日志固定保存为 `results/baseline/train_log.csv`，可通过 `--results-dir` 改变所在目录。
