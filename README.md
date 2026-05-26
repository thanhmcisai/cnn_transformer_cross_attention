# WiT: Wood Species Identification with Query-Guided Cross-Attention

This repository contains the experimental code for a wood species identification study using a hybrid CNN-Transformer architecture with query-guided cross-attention (QGCA). It is organized to support reproducible dataset splitting, multi-seed model training, evaluation, ablation studies, and supplementary analyses for manuscript submission.

Supported datasets: BD11, BFS46, PCA11, WRD21, VN26, and VN99.

## Overview

The proposed WiT model combines local anatomical representations from a CNN backbone with global contextual representations from a Transformer encoder. A query-guided cross-attention module fuses these representations before classification. The repository includes baseline models and controlled architectural variants to quantify the contribution of each component.

The experimental pipeline provides:

- Leakage-aware train/validation/test splitting based on feature clustering.
- Multi-seed training with validation macro-F1 early stopping.
- Baseline, ablation, and proposed-model comparisons.
- Bootstrap confidence intervals for reported metrics.
- Supplementary analyses for complexity, corruption robustness, cross-dataset transfer, and attention/token concentration.

## Repository Structure

```text
configs/
  dataset.yaml          Dataset paths and split settings
  experiments.yaml      Model registry, training settings, and output paths
src/
  datasets/             Feature extraction, threshold analysis, CSV split creation
  models/               Baseline, ablation, and WiT model definitions
  train/                Multi-seed training entrypoint
  evaluation/           Checkpoint evaluation on CSV splits
  analysis/             Complexity, robustness, transfer, and attention metrics
  visualization/        Confusion matrix, embedding, and attention-map utilities
```

Generated artifacts are excluded from version control:

```text
data/
features/
checkpoints/
results/
pretrained_weights/
```

## Installation

Use Python 3 explicitly. On some systems, `python` still points to Python 2.

```bash
conda create -n wit-wood python=3.10
conda activate wit-wood
python3 -m pip install -r requirements.txt
```

Install a PyTorch build matching your CUDA version if the default package is not suitable for your machine.

## Data Layout

Place each dataset under `data/raw/` using one of the following layouts:

```text
data/raw/{dataset}/{species}/{image}
```

or:

```text
data/raw/{dataset}/{train,val,test}/{species}/{image}
```

Dataset names and paths are configured in `configs/dataset.yaml` and `configs/experiments.yaml`.

## Reproducible Pipeline

Run commands from the repository root.

### 1. Feature Extraction

```bash
python3 -m src.datasets.extract_features
```

Output:

```text
features/{dataset}_features.npz
```

### 2. Threshold Analysis

```bash
python3 -m src.datasets.threshold_analysis
```

Outputs:

```text
results/threshold_analysis/threshold_analysis.csv
results/threshold_analysis/threshold_analysis_species.csv
```

### 3. CSV Split Generation

```bash
python3 -m src.datasets.split_from_features --threshold 0.10
```

Outputs:

```text
data/raw/{dataset}/train.csv
data/raw/{dataset}/val.csv
data/raw/{dataset}/test.csv
```

### 4. Multi-Seed Training

```bash
python3 -m src.train.train_multiseed
```

The default model list and random seeds are defined in `configs/experiments.yaml`.

Checkpoints:

```text
checkpoints/{dataset}_{model}_s{seed}.pt
```

Result tables:

```text
results/results_raw.csv
results/results_summary.csv
```

Example subset run:

```bash
python3 -m src.train.train_multiseed \
  --datasets pca11 \
  --models A1_CNN A2_Transformer A6_Par_CrossAttn WiT \
  --seeds 42
```

### 5. Evaluation

```bash
python3 -m src.evaluation.evaluate_csv --seed 42
```

Outputs are saved under:

```text
results/evaluation/
```

## Models

The active model registry is defined in `configs/experiments.yaml`.

Baselines:

- EfficientNet-B0
- DenseNet121
- CoAtNet-0
- MobileViT-S
- SwinV2-Base

Ablation and proposed models:

- `A1_CNN`: CNN-only local feature baseline.
- `A2_Transformer`: Transformer-only global feature baseline.
- `A3_Seq_Concat`: sequential CNN-Transformer with concatenated pooling.
- `A4_Seq_Single`: sequential CNN-Transformer with a single pooled representation.
- `A5_Par_Concat`: parallel CNN/Transformer fusion by concatenation.
- `A6_Par_CrossAttn`: parallel fusion with cross-attention.
- `WiT`: proposed sequential CNN-Transformer model with QGCA fusion.

## Supplementary Analyses

Computational complexity:

```bash
python3 -m src.analysis.complexity_benchmark
```

Deployment runtime, throughput, and optional ONNX CPU latency:

```bash
python3 -m src.analysis.runtime_benchmark --models A1_CNN A2_Transformer A6_Par_CrossAttn WiT
python3 -m src.analysis.runtime_benchmark --models WiT --onnx
```

WiT architecture sensitivity:

```bash
python3 -m src.analysis.sensitivity_analysis \
  --models WiT_depth1 WiT_depth2 WiT WiT_depth6 WiT_global_query
```

VN99 threshold sensitivity:

```bash
python3 -m src.datasets.vn99_threshold_sensitivity
```

Corruption robustness:

```bash
python3 -m src.analysis.corruption_robustness \
  --models A1_CNN A2_Transformer A6_Par_CrossAttn WiT
```

Cross-dataset transfer:

```bash
python3 -m src.analysis.feature_transfer_eval --models A1_CNN WiT --k-values 1 3 5 7
```

Attention and token concentration metrics:

```bash
python3 -m src.analysis.attention_metrics \
  --models A1_CNN A2_Transformer WiT \
  --limit-batches 10
```

## Reproducibility

- Default seeds: `42`, `123`, and `456`.
- Training monitor: validation macro-F1.
- Model hyperparameters and output paths are configured in `configs/experiments.yaml`.
- Summary tables report mean, standard deviation, and bootstrap 95% confidence intervals.
- Data, checkpoints, and generated results are not committed to the repository.

## Citation

If this repository supports your research, please cite the associated manuscript:

```bibtex
@article{macong2026witwood,
  title   = {Wood Species Identification via a Hybrid CNN-Transformer with Query-Guided Cross-Attention},
  author  = {Ma-Cong, Thanh and Coauthors},
  journal = {IEEE Access},
  year    = {2026}
}
```

## Contact

Thanh Ma-Cong  
Email: thanhmc.isai@gmail.com

---

## License

This repository is released for research use. Please follow the original dataset licenses and terms of use when using the data.
