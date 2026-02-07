# CNN–Transformer Cross-Attention for Wood Species Recognition

This repository implements a complete experimental pipeline for the paper:

> **A CNN–Transformer Cross-Attention Network for Multiscale Wood Species Recognition**

The project integrates:

- CNN-based local feature extraction
- Transformer-based global feature modeling
- Cross-attention fusion
- Comprehensive experiments with baselines and ablation models (A1–A7)

It supports:

- Dataset preprocessing with clustering-based splitting
- Training and evaluation
- Visualization (loss, accuracy, confusion matrix, t-SNE/UMAP, attention maps)

---

## 📁 Project Structure

```

cnn_transformer_cross_attention/
│
├── configs/
│ ├── dataset.yaml
│ └── experiments.yaml
│
├── src/
│ ├── datasets/
│ │ ├── download.py
│ │ ├── split_by_clustering.py
│ │ └── data_statistics.py
│ │
│ ├── models/
│ │ └── models.py
│ │
│ ├── train/
│ │ ├── train.py
│ │ └── engine.py
│ │
│ ├── evaluation/
│ │ └── evaluate.py
│ │
│ ├── visualization/
│ │ ├── plot_loss.py
│ │ ├── plot_accuracy.py
│ │ ├── confusion_matrix.py
│ │ ├── tsne_umap.py
│ │ └── attention_map.py
│ │
│ └── utils/
│ │ └── io.py
│
├── data/
│ ├── raw/
│ ├── processed/
│ ├── splits/
│ └── stats/
│
├── results/
│
├── requirements.txt
└── README.md

```

---

## 📊 Supported Datasets

The pipeline supports the following datasets:

- **BD11**
- **BFS46**
- **PCA11**
- **WRD21**
- **VN26**
- **VN99**

Dataset configuration is defined in:

```

configs/dataset.yaml

```

Each dataset is automatically:

1. Downloaded (if public link is provided)
2. Split using clustering-based strategy
3. Organized into:

```

processed/{dataset}/train/{class_name}/
processed/{dataset}/val/{class_name}/
processed/{dataset}/test/{class_name}/

```

---

## ⚙️ Installation

### 1. Clone repository

```bash
git clone https://github.com/yourname/cnn-transformer-cross-attention.git
cd cnn-transformer-cross-attention
```

### 2. Create environment

```bash
conda create -n woodnet python=3.9
conda activate woodnet
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Install PyTorch with CUDA if available:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 🧪 Dataset Preprocessing

### 1. Download datasets

```bash
python src/datasets/download.py
```

### 2. Clustering-based split

```bash
python src/datasets/split_by_clustering.py
```

This generates:

```
data/processed/{dataset}/train/
data/processed/{dataset}/test/
data/splits/{dataset}_train_split_map.csv
data/splits/{dataset}_test_split_map.csv
```

### 3. Dataset statistics

```bash
python src/datasets/data_statistics.py
```

Statistics saved in:

```
data/stats/dataset_statistics.json
```

---

## 🏋️ Training

Training is driven by:

```
configs/experiments.yaml
```

Run training:

```bash
python src/train/train.py
```

This will:

- Train all baseline and proposed models
- Save best checkpoint to:

```
results/{dataset}_{model}_nomix/best.pth
```

- Save training logs:

```
results/{dataset}_{model}_nomix_history.json
```

---

## 📈 Evaluation

Run evaluation:

```bash
python src/evaluation/evaluate.py
```

This computes:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Outputs:

```
results/evaluation/evaluation_results.json
results/evaluation/*_confusion_matrix.png
```

---

## 📊 Visualization

### A. Loss & Accuracy curves

```bash
python src/visualization/plot_loss.py
python src/visualization/plot_accuracy.py
```

Outputs:

```
results/plots/
```

---

### B. Confusion Matrix

```bash
python src/visualization/confusion_matrix.py
```

Outputs:

```
results/confusion_matrices/
```

---

### C. t-SNE / UMAP feature embedding

```bash
python src/visualization/tsne_umap.py
```

Outputs:

```
results/embedding/
```

---

### D. Attention Map (Grad-CAM)

```bash
python src/visualization/attention_map.py
```

Outputs:

```
results/attention/
```

---

## 🧠 Models

### Baseline models

- ConvNeXtV2
- EfficientNet
- DenseNet121
- CoAtNet
- Swin Transformer V2 (Tiny, Base)

### Proposed models (A1–A7)

- A1: Local CNN only
- A2: Global Transformer only
- A3–A4: Sequential fusion
- A5: Dual-branch fusion
- A6: Cross-attention fusion
- A7: Final proposed network (multi-stage cross-attention)

---

## 🔁 Reproducibility

All experiments are controlled via:

```
configs/dataset.yaml
configs/experiments.yaml
```

No hyperparameters are hardcoded in code files.

Random seeds are fixed for reproducibility.

---

## 📦 Using Pretrained Weights for Evaluation & Visualization

This repository supports evaluation and visualization directly from **pretrained weights** without retraining models.

All pretrained weights are defined in the configuration file:

```
configs/experiments.yaml
```

---

### 📁 Directory Structure for Pretrained Weights

Please organize pretrained weights as follows:

```
pretrained_weights/
├── vn26/
│   ├── swinv2_tiny_window8_256.pth
│   ├── swinv2_base_window16_256.pth
│   └── convnextv2_tiny.pth
├── vn99/
│   ├── swinv2_tiny_window8_256.pth
│   └── densenet121.pth
└── ...
```

---

### ⚙️ Configuration in `experiments.yaml`

Specify pretrained weights using the following format:

```yaml
weights:
  root_dir: "pretrained_weights"
  mapping:
    vn26:
      swinv2_tiny_window8_256: "vn26/swinv2_tiny_window8_256.pth"
      swinv2_base_window16_256: "vn26/swinv2_base_window16_256.pth"
      convnextv2_tiny: "vn26/convnextv2_tiny.pth"

    vn99:
      swinv2_tiny_window8_256: "vn99/swinv2_tiny_window8_256.pth"
      densenet121: "vn99/densenet121.pth"
```

Each entry maps:

```
<dataset_name> → <model_name> → weight_file_path
```

---

### ▶️ Run Evaluation Using Pretrained Weights

After placing pretrained weights and updating `experiments.yaml`, run:

```bash
python src/evaluation/evaluate.py
```

Outputs will be saved to:

```
results/evaluation/
├── evaluation_results.json
└── *_confusion_matrix.png
```

---

### 📊 Visualization with Pretrained Weights

All visualization scripts automatically load weights from `experiments.yaml`.

#### Confusion Matrix

```bash
python src/visualization/confusion_matrix.py
```

Output:

```
results/visualization/confusion_matrix/
```

---

#### t-SNE / UMAP Embedding

```bash
python src/visualization/tsne_umap.py
```

Output:

```
results/visualization/embedding/
```

---

#### Attention Map (Grad-CAM)

```bash
python src/visualization/attention_map.py
```

Output:

```
results/visualization/attention/
```

---

### 🔁 Reproducibility

By using pretrained weights defined in `experiments.yaml`, all evaluation and visualization results are:

- Fully reproducible
- Independent of the training process
- Consistent across different machines

No file paths are hardcoded in the codebase.
All paths are managed through configuration files.

---

### ⚠️ Notes

- Make sure the number of classes in the pretrained model matches the dataset.
- Pretrained weights must be saved using:

  ```python
  torch.save(model.state_dict(), "model_name.pth")
  ```

- If a pretrained weight is missing, the script will automatically skip that model and display a warning.

---

## 📜 Citation

If you use this code in your research, please cite:

```bibtex
@article{yourpaper2026woodnet,
  title={A CNN–Transformer Cross-Attention Network for Multiscale Wood Species Recognition},
  author={Your Name and Coauthors},
  journal={IEEE Access},
  year={2026}
}
```

---

## 📧 Contact

For questions or collaboration:

- Author: Thanh Ma-Cong
- Email: [thanhmc.isai@gmail.com](mailto:thanhmc.isai@gmail.com)

---

## ⭐ Acknowledgments

This project uses:

- PyTorch
- TIMM library
- Scikit-learn
- UMAP-learn

We thank the authors of these tools and the dataset providers.

---

## ⚠️ License

This repository is released for research purposes only.
Please respect dataset licenses when using the data.
