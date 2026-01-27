# ============================
# Project Makefile
# ============================

PYTHON=python
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
	@echo "  make train              Train all models"
	@echo "  make evaluate           Evaluate all models"
	@echo "  make complexity         Run complexity benchmark"
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
	pip install -r requirements.txt

# ============================
# Data Processing
# ============================
split_data:
	$(PYTHON) src/datasets/split_by_clustering.py

# ============================
# Training
# ============================
train:
	$(PYTHON) src/train/train.py

# ============================
# Evaluation
# ============================
evaluate:
	$(PYTHON) src/evaluation/evaluate.py

complexity:
	$(PYTHON) src/evaluation/complexity_benchmark.py

hardest_class:
	$(PYTHON) src/evaluation/hardest_class_analysis.py

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
all: split_data train evaluate visualize

# ============================
# Clean
# ============================
clean:
	rm -rf $(RESULTS_DIR)

