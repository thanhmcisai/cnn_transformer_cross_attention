# ============================
# Project Makefile
# ============================

PYTHON=python3
CONFIG_DATA=configs/dataset.yaml
CONFIG_EXP=configs/experiments.yaml

RESULTS_DIR=results
VIS_DIR=results/visualization

# ============================
# Help
# ============================
help:
	@echo "Available commands:"
	@echo "  make setup              Install dependencies"
	@echo "  make split_data         Split datasets using clustering"
	@echo "  make extract_features  Extract features for clustering"
	@echo "  make threshold         Analyze clustering thresholds"
	@echo "  make train              Train all models"
	@echo "  make evaluate           Evaluate all models"
	@echo "  make complexity         Run complexity benchmark"
	@echo "  make runtime            Run deployment runtime benchmark"
	@echo "  make confusion_matrix   Generate confusion matrices"
	@echo "  make attention_map      Generate Grad-CAM maps"
	@echo "  make tsne               Generate t-SNE plots"
	@echo "  make plots              Plot accuracy & loss curves"
	@echo "  make visualize          Run all visualization scripts"
	@echo "  make all                Full pipeline (split → train → eval → visualize)"

# ============================
# Setup
# ============================
setup:
	$(PYTHON) -m pip install -r requirements.txt

# ============================
# Data Processing
# ============================
split_data:
	$(PYTHON) -m src.datasets.split_from_features

extract_features:
	$(PYTHON) -m src.datasets.extract_features

threshold:
	$(PYTHON) -m src.datasets.threshold_analysis

# ============================
# Training
# ============================
train:
	$(PYTHON) -m src.train.train_multiseed

# ============================
# Evaluation
# ============================
evaluate:
	$(PYTHON) -m src.evaluation.evaluate_csv

complexity:
	$(PYTHON) -m src.analysis.complexity_benchmark

runtime:
	$(PYTHON) -m src.analysis.runtime_benchmark

sensitivity:
	$(PYTHON) -m src.analysis.sensitivity_analysis

vn99_threshold:
	$(PYTHON) -m src.datasets.vn99_threshold_sensitivity

hardest_class:
	$(PYTHON) -m src.analysis.hardest_class_analysis

corruption:
	$(PYTHON) -m src.analysis.corruption_robustness

transfer:
	$(PYTHON) -m src.analysis.feature_transfer_eval

attention_metrics:
	$(PYTHON) -m src.analysis.attention_metrics

# ============================
# Visualization
# ============================
confusion_matrix:
	$(PYTHON) src/visualization/confusion_matrix.py

attention_map:
	$(PYTHON) src/visualization/attention_map.py

tsne:
	$(PYTHON) src/visualization/tsne_umap.py

plots:
	$(PYTHON) src/visualization/plot_accuracy.py
	$(PYTHON) src/visualization/plot_loss.py

visualize: confusion_matrix attention_map tsne plots

# ============================
# Full Pipeline
# ============================
all: extract_features threshold split_data train evaluate visualize

# ============================
# Clean
# ============================
clean:
	rm -rf $(RESULTS_DIR)
