#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

QUICK_FLAG=""
if [[ "${1:-}" == "--quick" ]]; then
  QUICK_FLAG="--quick"
fi

if [[ -x "$ROOT/.venv/bin/medimg-train" ]]; then
  "$ROOT/.venv/bin/medimg-train" --model all $QUICK_FLAG
elif command -v medimg-train >/dev/null 2>&1; then
  medimg-train --model all $QUICK_FLAG
else
  python -m medical_image_ml.cli --model all $QUICK_FLAG
fi
