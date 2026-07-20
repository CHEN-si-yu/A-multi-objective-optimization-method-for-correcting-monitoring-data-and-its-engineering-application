# Data Quality Assessment — DPS Neural Network

Physics-constrained neural network (DPS/PINN) for curve data quality assessment.

> **Paper:** *"A multi-objective optimization method for correcting monitoring data and its engineering application"*

## Prerequisites

- **Python**: 3.9+ (tested on 3.12)
- **OS**: Windows / Linux / macOS
- **Hardware**: CPU is sufficient (no GPU required). Training all 8 points takes ~5–10 minutes on a modern CPU.
- **Disk**: ~6 MB for code, data, and pre-trained models.

## Project Structure

```
├── main.py                     # Single-file entry point
├── data/
│   └── data.xlsx               # Input data (observation + calculation sheets)
├── model/
│   └── dps*.pth                # Trained model checkpoints (8 files)
├── pic/
│   └── point_*.png             # Output curve-comparison figures
├── requirements.txt            # Python dependencies (exact versions)
├── LICENSE                     # MIT License
└── README.md
```

## Quick Start (for reviewers)

### 1. Clone and install

```bash
git clone git@github.com:CHEN-si-yu/A-multi-objective-optimization-method-for-correcting-monitoring-data-and-its-engineering-application.git
cd A-multi-objective-optimization-method-for-correcting-monitoring-data-and-its-engineering-application
pip install -r requirements.txt
```

### 2. Reproduce results (inference mode — recommended)

Uses the **pre-trained model checkpoints** in `model/` to produce the exact figures and metrics
reported in the paper. No training needed.

```bash
# Generate all 8 figures + metric tables (~10 seconds)
python main.py --no-train
```

This writes `pic/point_0.png` through `pic/point_7.png` and prints a summary table.

### 3. Reproduce results (full training mode)

Train all models from scratch using the fixed random seed (results are deterministic).

```bash
# Full training: all 8 points (~5–10 minutes)
python main.py

# Specific points only
python main.py --ids 0 3 5
```

### 4. Select a specific subset

```bash
# Inference only for points 0, 1, 2
python main.py --ids 0 1 2 --no-train
```

## Expected Output

Running `python main.py --no-train` produces:

1. **Console output** — A 3×3 metric table for each point (Smoothness, Physical consistency, Deviation) comparing `obs`, `cal`, and `dps` curves:

```
===== Point 0 =====
             obs |        cal |        dps
----------------------------------------------------------------------------------------------------
   smooth     ... |        ... |        ...
     phys     ... |        ... |        ...
      dev     ... |        ... |        ...
```

2. **Summary table** — Aggregated metrics across all 8 data points.

3. **Figures** — `pic/point_*.png` — Three curves per plot: observed (black solid), model baseline (blue dotted), DPS output (red solid).

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

## Reproducibility

- **Fixed random seed** — `SEED = 1008611` (line 27 of `main.py`). Both `torch.manual_seed()` and `np.random.seed()` are set.
- **Deterministic training** — No non-deterministic CUDA ops. All results are reproducible across runs.
- **Pre-trained models** — The `model/dps*.pth` files are the exact checkpoints used to produce paper results. Use `--no-train` for bit-for-bit reproduction.
- **Exact dependency versions** — `requirements.txt` pins exact versions tested.

## Troubleshooting

**`KMP_DUPLICATE_LIB_OK` error on Windows**: Already handled (line 30 of `main.py`).

**`FileNotFoundError: data/data.xlsx`**: Ensure you run `python main.py` from the repository root directory.

**Model checkpoints missing**: If `model/*.pth` files are deleted, run `python main.py` (without `--no-train`) to re-train.

## License

MIT — see [LICENSE](LICENSE).
