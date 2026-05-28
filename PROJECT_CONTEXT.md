# Project Context

## 项目方向

项目暂定题目方向：

基于智能体多工具调度的自适应信号预处理调制识别方法。

核心任务是自动调制识别。输入为 RadioML2016.10A 或类似 RadioML 数据集中的 IQ 信号，常见单样本形状为 `[2, 128]`，输出为调制类型。

当前研究主线不是让智能体直接完成分类，而是先建立可复现的深度学习 baseline，再分析传统信号处理工具是否能对特定样本、特定 SNR 或特定调制类型带来条件性增益。只有在确认工具确实存在可利用增益后，再进一步研究工具调度策略，最后才考虑接入 LangChain 智能体框架。

## 当前阶段目标

当前最重要目标是建立一个最小可复现实验流程，作为后续所有工具增益分析和调度研究的基线。

本阶段只做 baseline，不写智能体，不引入复杂 VMD 大框架，不重构旧代码，不做大而全的工程化系统。

## 实验顺序

1. 实现 RadioML 数据加载。
2. 按 `modulation + SNR` 进行分层划分，生成 `train / val / test`。
3. 实现 `SimpleCNN1D` 调制分类网络。
4. 训练 baseline 分类器。
5. 在 `val / test` 上保存逐样本预测结果。
6. 输出整体精度、按 SNR 精度、按调制类型精度，以及 SNR-accuracy 曲线数据。
7. 将所有 baseline 结果保存到 `results/baseline/`。

## Baseline 预测结果字段

逐样本预测结果至少包含以下字段：

- `sample_id`
- `true_mod`
- `true_snr`
- `pred_mod`
- `correct`
- `top1_confidence`
- `top2_confidence`
- `confidence_margin`
- `entropy`

这些字段用于后续分析哪些样本可能受益于传统信号处理工具，例如低置信度样本、高熵样本、特定 SNR 区间样本、特定调制类别样本或容易混淆的类别对。

## 关键输出指标

baseline 阶段需要输出以下指标：

- overall accuracy
- accuracy by SNR
- accuracy by modulation
- SNR-accuracy 曲线数据

建议保存为结构化文件，例如：

- `results/baseline/val_predictions.csv`
- `results/baseline/test_predictions.csv`
- `results/baseline/metrics_overall.json`
- `results/baseline/accuracy_by_snr.csv`
- `results/baseline/accuracy_by_modulation.csv`
- `results/baseline/snr_accuracy_curve.csv`
- `results/baseline/train_log.csv`
- `results/baseline/best_model.pt`

## 当前不做的内容

当前阶段明确不做以下内容：

- 不写 LangChain 智能体主流程。
- 不让智能体直接根据手工特征分类。
- 不把传统信号处理工具直接接入训练主线。
- 不搭建 VMD 或其他复杂信号分解大框架。
- 不重构所有旧代码。
- 不做多模型融合。
- 不做自动工具调度。
- 不做论文级完整系统包装。

这些内容保留到 baseline 稳定之后，再根据验证结果逐步加入。

## 后续研究路线建议

baseline 建立后，建议按以下顺序推进：

1. 固定数据划分、模型结构、训练配置和随机种子，保证 baseline 可复现。
2. 分析 baseline 在不同 SNR 和 modulation 上的错误分布。
3. 选择少量传统预处理工具，例如滤波、归一化、去噪或频域增强，单独验证它们对分类结果的影响。
4. 对比工具前后逐样本预测变化，识别工具的条件性增益。
5. 如果工具在部分样本上有稳定增益，再设计轻量级工具选择策略。
6. 最后再考虑将工具选择策略包装成 LangChain 智能体或多工具调度系统。

## 论文实验逻辑

论文中较稳妥的实验叙事可以是：

1. 深度学习模型作为基础调制识别器。
2. 传统信号处理工具不直接替代分类器，而是作为可选预处理操作。
3. 不同工具对不同 SNR、不同调制类型、不同置信度样本的作用不同。
4. 因此需要一种自适应工具调度机制。
5. 智能体框架可作为工具调度与实验编排层，而不是直接承担调制分类任务。

这个叙事可以避免“智能体直接分类效果差”的风险，同时保留智能体多工具调度作为研究亮点。

## 工作日志

### 2026-05-27 数据划分索引生成

本次完成 RadioML2016.10A 数据集的最小可复现实验划分准备。

已新增脚本：

- `scripts/create_radioml_splits.py`

脚本功能：

- 读取原始数据 `data/raw/RML2016.10a_dict.pkl`。
- 不复制原始 IQ 数据。
- 按 `modulation + SNR` 组合进行分层划分。
- 每个组合内部使用固定随机种子 `seed=42` 按 `60% / 20% / 20%` 划分为 `train / val / test`。
- 只保存索引文件和划分元数据。
- 在 `split_meta.json` 中记录全局 `sample_id` 的构造规则：按调制类型排序，再按 SNR 排序，再按组内样本序号分配。

已生成文件：

- `data/splits/train_indices.npy`
- `data/splits/val_indices.npy`
- `data/splits/test_indices.npy`
- `data/splits/split_meta.json`

划分结果：

- total samples: `220000`
- train samples: `132000`
- val samples: `44000`
- test samples: `44000`
- groups: `220`
- 每个 `(modulation, SNR)` 组合内样本数为 `1000`，划分为 `600 / 200 / 200`。

校验结果：

- `train / val / test` 三组索引互不重叠。
- 三组索引合并后覆盖全量样本索引范围 `0..219999`。
- 后续 Dataset 应根据这些 indices 在运行时从原始 pkl 文件读取样本，不生成 `train.npy / val.npy / test.npy` 三份完整数据。

补充实现：

- `src/amc/data/radioml_dataset.py`
- `src/amc/data/__init__.py`
- `src/amc/__init__.py`

`RadioMLDataset` 当前行为：

- 加载原始 `RadioML2016.10A` pkl 文件。
- 加载指定 split 的 indices 文件。
- 根据 `split_meta.json` 中的全局索引规则，将 `sample_id` 映射回 `(modulation, SNR, local_idx)`。
- 在运行时从原始 pkl 中读取对应 IQ 样本。
- 返回字段包括 `x`、`label`、`sample_id`、`modulation`、`snr`。

已验证：

- 使用 `data/splits/val_indices.npy` 初始化 Dataset 后，样本数量为 `44000`。
- 单个样本 `x` 的形状为 `[2, 128]`。

### 2026-05-27 Dataset 映射规则修正

本次修正 `RadioMLDataset` 中 `sample_id` 到 `(modulation, snr, local_idx)` 的映射逻辑。

修改文件：

- `src/amc/data/radioml_dataset.py`

修改内容：

- Dataset 显式读取 `split_meta.json` 中的 `index_policy.modulation_order` 和 `index_policy.snr_order`。
- 按上述顺序重新构造运行时 `group_table`。
- 每组 `num_samples` 从原始 `raw_data[(modulation, snr)].shape[0]` 获取，不写死为 `1000`。
- 保留 `split_meta.json` 中的 `groups`，用于一致性校验。
- 如果重新构造的 `group_table` 与 `meta["groups"]` 中的 `start_index`、`end_index` 或 `num_samples` 不一致，会抛出清晰的 `ValueError`。
- 新增 `resolve_sample_id(sample_id)` 方法，统一完成全局样本索引到 `(modulation, snr, local_idx)` 的解析。

本次未重新生成以下文件：

- `data/splits/train_indices.npy`
- `data/splits/val_indices.npy`
- `data/splits/test_indices.npy`
- `data/splits/split_meta.json`

验证结果：

- 使用 `data/splits/test_indices.npy` 初始化 Dataset。
- 随机抽取 `12` 个 `sample_id`。
- 每个样本均可正确解析到 `(modulation, snr, local_idx)`。
- 每个解析结果均能从原始 pkl 中读取到形状为 `[2, 128]` 的 IQ 样本。

### 2026-05-27 SimpleCNN1D baseline 实现

本次完成最小 baseline 分类器代码，不涉及 VMD、小波、滤波、智能体或工具筛选。

新增模型文件：

- `src/amc/models/simple_cnn.py`
- `src/amc/models/__init__.py`

模型行为：

- `SimpleCNN1D` 输入形状为 `[B, 2, 128]`。
- 输出形状为 `[B, 11]` logits。
- 网络结构为轻量 1D CNN，包含 `Conv1d`、`BatchNorm1d`、`ReLU`、`MaxPool1d`、`AdaptiveAvgPool1d` 和全连接分类头。

新增训练脚本：

- `scripts/train_baseline.py`

训练脚本行为：

- 使用 `data/splits/train_indices.npy` 训练。
- 使用 `data/splits/val_indices.npy` 选择 best checkpoint。
- loss 使用 `CrossEntropyLoss`。
- optimizer 使用 `Adam`。
- 自动选择 `cuda` 或 `cpu`。
- 默认保存 best model 到 `checkpoints/simple_cnn_best.pt`。
- 默认保存训练日志到 `results/baseline/train_log.csv`。

新增评估脚本：

- `scripts/evaluate_baseline.py`

评估脚本行为：

- 默认评估 `val` 和 `test`。
- 输出 overall accuracy。
- 输出 accuracy by SNR。
- 输出 accuracy by modulation。
- 保存逐样本预测 CSV。
- 保存目录为 `results/baseline/`。

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

轻量验证结果：

- `SimpleCNN1D(torch.randn(4, 2, 128))` 输出形状为 `[4, 11]`。
- `scripts/train_baseline.py --help` 可正常运行。
- `scripts/evaluate_baseline.py --help` 可正常运行。
- 使用 `RadioMLDataset` 取 `8` 个验证样本，可以正常完成 logits、softmax、top2 confidence 和 entropy 计算。
- `python -m compileall src scripts` 通过。
