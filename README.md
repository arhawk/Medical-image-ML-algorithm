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

TensorFlow requires **Python 3.10–3.12** (`requires-python = ">=3.10,<3.13"` in `pyproject.toml`). Use the setup script (recommended):

```bash
./scripts/setup_venv.sh
source .venv/bin/activate
```

Or manually on macOS with Homebrew:

```bash
brew install python@3.12
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev,macos]"
```

A `.python-version` file pins **3.12** for pyenv/asdf.

### Troubleshooting

If `python3 --version` shows **3.13 or 3.14**, do **not** use that interpreter for this project — TensorFlow will fail to install or import. Create the virtualenv explicitly with `python3.12` (see above). After activation, `python --version` should report 3.12.x.

## Run

```bash
# Train all models (quick smoke test)
medimg-train --model all --quick

# Full training (MLP 30 epochs, CNN 40 epochs, final fit on train+val)
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

### Performance (Apple Silicon)

On macOS (Apple Silicon), use **TensorFlow 2.18.x** with **tensorflow-metal 1.2.0** — TF 2.21 is not yet compatible with the Metal plugin on Python 3.12. GPU support is included in default dependencies on darwin; or install explicitly:

```bash
pip install -e ".[macos]"
```

Training auto-tunes runtime settings: `tf.data` prefetch, mixed precision when a GPU is present, and `--batch-size 0` (default) picks 256 on GPU / 128 on 8+ CPU cores. At startup you should see e.g. `1 GPU(s), batch_size=256, mixed_precision=True`. Use `--cpu-only` to force CPU. To set batch size manually: `medimg-train --model cnn --batch-size 128`.

**Note:** Random Forest (scikit-learn) always runs on CPU. MLP and CNN use the Apple Metal GPU when `tensorflow-metal` is installed.

RF GridSearch results are cached at `outputs/tuning/rf/best_params.json` after the first run. Subsequent `medimg-train --model rf` calls reuse the cache. Pass `--retune-rf` to force a new GridSearch.

## Models & Reported Results

| Model | Test Accuracy |
|-------|---------------|
| Random Forest (PCA) | 65.7% |
| MLP (PCA) | 69.3% |
| CNN | 90.1% |

*From full CLI run (`medimg-train --model all`), July 2026.*

## Dependencies

- numpy, pandas, scikit-learn, tensorflow, keras-tuner, matplotlib, seaborn

See `requirements.txt` and `pyproject.toml` for version constraints.

## Archive

The original COMP5318 assignment notebook and PDF report live in `archive/`. Active development uses the `medical_image_ml` Python package.
