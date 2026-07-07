# Medical Image ML Algorithm

Histopathology image classification project comparing **Random Forest**, **MLP**, and **CNN** models on a 9-class dataset (28×28 RGB).

> **Portfolio presentation:** [docs/SHOWCASE.md](docs/SHOWCASE.md)

## Project Structure

```
├── archive/                    # Original course submission (reference only)
│   ├── notebooks/              # Jupyter notebook
│   └── report.pdf
├── data/Assignment2Data/       # Training and test NumPy arrays
├── docs/
│   ├── SHOWCASE.md             # English portfolio write-up
│   └── assets/                 # Figures for showcase
├── outputs/                    # Training artifacts (gitignored)
├── scripts/
│   └── run_all.sh              # One-command training
├── src/medical_image_ml/       # Python package
├── pyproject.toml
└── requirements.txt
```

## Setup

TensorFlow requires **Python 3.10–3.12**. On macOS with Homebrew:

```bash
brew install python@3.12
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

## Run

```bash
# Train all models (quick smoke test)
medimg-train --model all --quick

# Full training with best-known hyperparameters
medimg-train --model all

# Individual models
medimg-train --model rf
medimg-train --model mlp
medimg-train --model cnn

# Optional CNN hyperparameter search
medimg-train --model cnn --tune

# Shell wrapper
./scripts/run_all.sh --quick
```

Outputs (metrics, reports, figures) are written to `outputs/`. Key figures are also copied to `docs/assets/` for the showcase page.

## Models & Reported Results

| Model | Test Accuracy |
|-------|---------------|
| Random Forest (PCA) | ~64% |
| MLP (PCA) | ~69% |
| CNN | ~93% |

## Dependencies

- numpy, pandas, scikit-learn, tensorflow, keras-tuner, matplotlib, seaborn

See `requirements.txt` and `pyproject.toml` for version constraints.

## Archive

The original COMP5318 assignment notebook and PDF report live in `archive/`. Active development uses the `medical_image_ml` Python package.
