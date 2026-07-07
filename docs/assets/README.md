# Generated figures for docs/SHOWCASE.md

Populated automatically when running `medimg-train`. Key assets:

- **Baseline:** `medimg-train --model rf`, `medimg-train --model mlp --cpu-only`, `medimg-train --model cnn`
- **Portfolio (Grad-CAM + error analysis):** `medimg-train --model portfolio` (RF + CNN on GPU)

On Apple Silicon, run MLP with `--cpu-only`; CNN benefits from GPU (~90% vs ~80% on CPU).
