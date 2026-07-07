# Medical Image Classification — Portfolio Showcase

> **Personal ML case study inspired by coursework** — histopathology image classification comparing Random Forest, MLP, and hand-crafted CNN on a 9-class medical imaging dataset. I refactored the original notebook into a reproducible Python package with CLI, cached tuning, Grad-CAM interpretability, and structured error analysis. Best hand-crafted CNN achieves **90.1% test accuracy** (CLI baseline: RF GridSearch, MLP `--cpu-only`, CNN on GPU).

| Model | Test Accuracy | Training Cost | Best Hyperparameters |
|-------|---------------|---------------|----------------------|
| Random Forest (PCA) | 65.7% | Low | `n_estimators=250`, `max_depth=20`, `max_features=sqrt` (GridSearch default; `--no-tune` fallback) |
| MLP (PCA) | 69.3% | Medium | `units=256`, `relu`, `dropout=0.2`, `lr=0.1` |
| **CNN (hand-crafted)** | **90.1%** | High | `filters=(32,64)`, `kernels=(3,5)`, `dense=64`, `dropout=0.3` |

**Interview one-liner:** *"I built a hand-crafted CNN that hit 90% on 28×28 histopathology tiles, then added Grad-CAM and structured error analysis to show where the model looks and which tissue classes it confuses — the kind of interpretability medical imaging teams expect."*

---

## Problem & Motivation

Automated classification of histopathological tissue images supports computer-aided diagnosis (CAD) in digital pathology. This project systematically compares classical machine learning and deep learning approaches to identify the most effective model for multi-class tissue classification.

**Goals:**
- Compare model families on identical data and metrics
- Tune hyperparameters with structured search (RF GridSearch; optional CNN keras-tuner)
- Evaluate generalization on a held-out test set
- Add interpretability (Grad-CAM) and error analysis for portfolio depth

---

## Dataset

| Split | Samples | Format |
|-------|---------|--------|
| Training | 32,000 | 28×28 RGB (`uint8`, 0–255) |
| Test | 8,000 | 28×28 RGB |

- **9 classes** — tissue types including adipose, tumor, muscle, glandular, stromal, and background
- **Balanced** class distribution (no resampling required)
- **Preprocessing:** pixel normalization to [0, 1]; PCA (95% variance, ~330 components) for RF/MLP only

![Sample images per class](assets/sample_images.png)

![Class distribution](assets/class_distribution.png)

---

## Approach Overview

```mermaid
flowchart LR
    Data[NumPy arrays] --> Norm[Normalize 0-1]
    Norm --> Split[80/20 train/val split]
    Split --> PCA[PCA for RF and MLP]
    Split --> CNNPath[Raw tensors for CNN]
    PCA --> RF[Random Forest]
    PCA --> MLP[MLP]
    CNNPath --> CNN[CNN]
    RF --> Eval[Metrics and confusion matrices]
    MLP --> Eval
    CNN --> Eval
    CNN --> GradCAM[Grad-CAM]
    Eval --> ErrorAnalysis[Error analysis]
```

Three model families share the same evaluation pipeline (accuracy, F1, precision/recall, confusion matrices) but differ in feature representation:

| Model | Input features | Key idea |
|-------|----------------|----------|
| Random Forest | PCA-reduced vectors | Ensemble of decision trees; fast, interpretable baseline |
| MLP | PCA-reduced vectors | Non-linear fully connected network |
| CNN | Raw 28×28×3 images | Spatial convolutions capture texture and structure |

---

## Methods

### Random Forest

- **Theory:** Bagged ensemble of decision trees with random feature subsets at each split
- **Tuning:** `GridSearchCV` with 3-fold stratified CV over `n_estimators`, `max_depth`, `max_features` (cached to `outputs/tuning/rf/best_params.json`)
- **Fallback (`--no-tune`):** 250 trees, `max_depth=20`, `sqrt` features
- **Strengths:** Fast training, feature importance, robust baseline
- **Limitations:** Struggles with visually similar tissue classes after PCA compression

### MLP (Multi-Layer Perceptron)

- **Architecture:** 2 hidden layers (256 units, ReLU) + dropout 0.2 on PCA features
- **Optimizer:** SGD with learning rate 0.1
- **Tuning:** Fixed best-known params from notebook search (CLI has no MLP tuner)
- **Result:** 69.3% test accuracy — moderate performance, lacks spatial awareness
- **Apple Silicon:** run with `--cpu-only`; GPU + mixed precision breaks SGD at `lr=0.1`

### CNN (Convolutional Neural Network)

- **Architecture:** Conv2D(32,3) → Conv2D(64,5) → MaxPool → GlobalAvgPool → Dense(64) → Softmax
- **Regularization:** L2 (1e-4), BatchNorm, Dropout (0.3), EarlyStopping, ReduceLROnPlateau
- **Tuning:** Fixed `BEST_CNN_PARAMS` by default; optional `--tune` runs keras-tuner (not used for reported 90.1%)
- **Result:** **90.1% test accuracy** — best at capturing spatial patterns in tissue images (GPU recommended)

---

## Results

![Model comparison](assets/model_comparison.png)

| Model | Accuracy | F1 (weighted) | Interpretability | Speed |
|-------|----------|---------------|------------------|-------|
| PCA + RF | 0.657 | 0.649 | High | Fast |
| PCA + MLP | 0.693 | 0.685 | Medium | Medium |
| **CNN** | **0.901** | **0.901** | Low | Slow |

**Key findings:**
- CNN significantly outperforms classical and fully-connected approaches on raw image tensors
- RF is a useful interpretable baseline but confuses visually similar classes (2, 6, 7)
- MLP bridges the gap but cannot match CNN without spatial feature learning
- RF GridSearch is cached; MLP/CNN use fixed params unless `--tune` (CNN only)

Confusion matrices and training curves are generated automatically when running the CLI (see `outputs/figures/`).

![Random Forest confusion matrix](assets/confusion_matrix_rf.png)

![MLP confusion matrix](assets/confusion_matrix_mlp.png)

![CNN confusion matrix](assets/confusion_matrix_cnn.png)

### Grad-CAM Interpretability

Grad-CAM highlights spatial regions the model uses for classification — useful for clinical trust and debugging.

![CNN Grad-CAM](assets/gradcam_cnn.png)

Generated automatically after CNN training (`medimg-train --model cnn` or `--model portfolio`).

---

## Error Analysis

Which classes confuse each other? Where do RF and CNN disagree?

![CNN misclassified examples](assets/error_misclassified_cnn.png)

![RF vs CNN failure overlap](assets/error_analysis_rf_cnn.png)

**Findings (typical patterns from confusion matrices):**
- **Classes 2, 6, and 7** are the most frequently confused pairs across RF and CNN — visually similar stromal/glandular tissue textures
- **RF-only failures:** PCA compression loses spatial detail; RF misclassifies texture-heavy classes that CNN resolves with convolutions
- **CNN-only failures:** Rare edge cases with ambiguous boundaries or mixed tissue at tile edges
- **Both wrong:** Hardest samples — often class 2 ↔ 6 or 6 ↔ 7 swaps

Error analysis runs automatically when RF and CNN are trained in the same session (`--model portfolio`, `--model all`, or cached predictions from prior runs). Full report: `outputs/reports/error_analysis.txt`.

---

## Key Takeaways (Transferable Skills)

- End-to-end ML pipeline: data loading → preprocessing → model training → evaluation → artifact export
- Hyperparameter tuning strategies: GridSearch (RF, cached), fixed params (MLP/CNN), optional keras-tuner (`--tune` for CNN)
- Medical imaging specifics: PCA for tabular models vs. raw tensors for CNNs
- Model interpretability: Grad-CAM for spatial attribution
- Structured error analysis: confusion pairs, failure-mode comparison across model families
- Reproducible project structure: Python package, CLI entry points, versioned dependencies

---

## Tech Stack

- **Python 3.12** · NumPy · Pandas · scikit-learn · TensorFlow/Keras · keras-tuner · Matplotlib · Seaborn

---

## How to Reproduce

```bash
# Setup (requires Python 3.10–3.12 for TensorFlow)
./scripts/setup_venv.sh          # or: pip install -e ".[dev,macos]" on Apple Silicon
source .venv/bin/activate

# Quick smoke test (Phase 1 baseline — unchanged)
medimg-train --model all --quick

# Full training — recommended on Apple Silicon (MLP needs CPU; CNN uses GPU)
medimg-train --model rf
medimg-train --model mlp --cpu-only
medimg-train --model cnn

# Portfolio: RF + CNN with Grad-CAM and error analysis
medimg-train --model portfolio

# Skip RF GridSearch; use fixed params
medimg-train --model rf --no-tune

# Optional keras-tuner search (different architecture than baseline 90.1% config)
medimg-train --model cnn --tune
```

See the [project README](../README.md) for full directory layout.

---

## References

- Keras Team. Keras API reference. https://keras.io/api/
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436–444.
- Hornik, K., Stinchcombe, M., & White, H. (1989). Multilayer feedforward networks are universal approximators. *Neural Networks*, 2(5), 359–366.
- Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV*.

---

*Personal ML case study inspired by COMP5318/4318 coursework (University of Sydney, 2024). Original notebook and PDF report are archived in [`archive/`](../archive/). Engineering refactor, CLI, Grad-CAM, and error analysis were added independently for portfolio presentation.*
