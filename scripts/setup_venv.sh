#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if command -v python3.12 >/dev/null 2>&1; then
  PYTHON=python3.12
elif [ -x "/opt/homebrew/opt/python@3.12/bin/python3.12" ]; then
  PYTHON="/opt/homebrew/opt/python@3.12/bin/python3.12"
else
  echo "Python 3.12 is required for TensorFlow. Install with: brew install python@3.12" >&2
  exit 1
fi

echo "Using $($PYTHON --version)"

if [ ! -d ".venv" ]; then
  "$PYTHON" -m venv .venv
fi

.venv/bin/pip install -U pip setuptools wheel
.venv/bin/pip install -e ".[dev,macos]"

echo ""
echo "Checking TensorFlow GPU (Apple Metal)..."
.venv/bin/python - <<'PY'
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print(f"TensorFlow {tf.__version__}, GPUs: {len(gpus)}")
if gpus:
    print("GPU training ready (MLP/CNN will use Metal).")
else:
    print("No GPU detected. Run: pip install tensorflow-metal")
PY

echo "Done. Activate with: source .venv/bin/activate"
