# Data Quality Assessment — DPS Neural Network

Physics-constrained neural network (DPS/PINN) for curve data quality assessment.

## Project Structure

```
├── main.py                  # Single-file entry point (everything lives here)
├── data/
│   └── data.xlsx             # Input data (observation + calculation sheets)
├── model/
│   └── dps*.pth             # Trained model checkpoints
├── pic/
│   └── point_*.png          # Output curve-comparison figures
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
```

```bash
# Default: all point IDs 0~7, train mode, figures saved to ./pic/
python main.py

# Specific points only
python main.py --ids 0 3 5

# Inference only (skip training, load existing checkpoints)
python main.py --no-train
python main.py --ids 0 1 2 --no-train
```

## Architecture Overview

`main.py` is organized into clear sections:

| Section | Lines | Description |
|---------|-------|-------------|
| Configuration | ~30 | Seeds, paths, hyperparameters, loss weights |
| Physical constraint | ~100 | Sliding-window local correlation (NumPy + PyTorch) |
| Model | ~50 | `PredictModel` — FC network (256→128→256→output) |
| Loss functions | ~45 | `dps_loss` (Data-Physics-Smoothness), `pinn_loss` |
| Training | ~35 | `train_model` — loop with best-model checkpointing |
| Metrics | ~50 | Smoothness, Physical consistency, RMSE deviation |
| Utilities | ~55 | `xy_interp`, `plot_curve`, `print_metric_table` |
| Pipeline | ~100 | `load_data`, `run_single`, `main`, CLI |

## Metrics

| Metric | Meaning | Formula |
|--------|---------|---------|
| **Smoothness** | RMS of adjacent differences | sqrt(mean((xᵢ₊₁ − xᵢ)²)) |
| **Physical** | Mean local-max correlation distance | 1 − max(corr) averaged over sliding windows |
| **Deviation** | RMSE vs observed data | sqrt(mean((data − obs)²)) |

Each metric is computed for three curves: **obs** (observed), **cal** (model baseline), **dps** (DPS output).
