#!/usr/bin/env python
"""
Data Quality Assessment via Physics-Constrained Neural Networks (DPS/PINN).

Usage:
    python main.py                          # default: all IDs 0~7, train mode
    python main.py --ids 0 3 5              # specific point IDs
    python main.py --no-train               # inference only (skip training)
    python main.py --ids 0 1 2 --no-train   # specific IDs, inference only
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt

# ============================================================================
#  Configuration
# ============================================================================

SEED = 12321
torch.manual_seed(SEED)
np.random.seed(SEED)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Paths — resolved relative to this file's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
PIC_DIR = os.path.join(BASE_DIR, "pic")
DATA_FILE = os.path.join(DATA_DIR, "data.xlsx")

# Training hyperparameters
NUM_EPOCHS = 1000
LEARNING_RATE = 0.001
BATCH_SIZE = 1

# Physical constraint parameters
WINDOW = 36
SHIFT_RANGE = 3

# Loss weights
LOSS_DATA_WEIGHT = 1.0
LOSS_PHYSICS_WEIGHT = 100.0
LOSS_SMOOTHNESS_WEIGHT = 10.0


# ============================================================================
#  Physical constraint — sliding-window local correlation
# ============================================================================

def physical_numpy(data1, data2, window=WINDOW, shift_range=SHIFT_RANGE):
    """
    NumPy vectorised implementation.
    Computes mean local max correlation between two 1-D sequences.
    """
    if len(data1.shape) > 1:
        data1 = data1.squeeze(0)
        data2 = data2.squeeze(0)

    x = np.asarray(data1, dtype=float)
    y = np.asarray(data2, dtype=float)
    T = len(x)
    L = window

    if T < L:
        return 0.0

    W_x = sliding_window_view(x, L)
    W_y = sliding_window_view(y, L)
    C = W_x.shape[0]

    W_x_mean = W_x.mean(axis=1, keepdims=True)
    W_y_mean = W_y.mean(axis=1, keepdims=True)
    Xc = W_x - W_x_mean
    Yc = W_y - W_y_mean

    X_ss = np.sum(Xc * Xc, axis=1)
    Y_ss = np.sum(Yc * Yc, axis=1)

    S = shift_range
    shift_num = 2 * S + 1
    corr_mat = np.full((C, shift_num), -np.inf, dtype=float)

    for s in range(-S, S + 1):
        idx = s + S
        if s >= 0:
            xs, ys = Xc[0:C - s], Yc[s:C]
            xs_ss, ys_ss = X_ss[0:C - s], Y_ss[s:C]
            row_slice = slice(0, C - s)
        else:
            k = -s
            xs, ys = Xc[k:C], Yc[0:C - k]
            xs_ss, ys_ss = X_ss[k:C], Y_ss[0:C - k]
            row_slice = slice(k, C)

        num = np.sum(xs * ys, axis=1)
        denom = np.sqrt(xs_ss * ys_ss)
        corr = np.zeros_like(num)
        valid = denom > 0
        corr[valid] = num[valid] / denom[valid]
        corr_mat[row_slice, idx] = corr

    max_corr = np.max(corr_mat, axis=1)
    valid_rows = max_corr != -np.inf
    if not np.any(valid_rows):
        return 0.0

    P = np.sum(1.0 - max_corr[valid_rows])
    count = np.sum(valid_rows)
    return float(P) / count if count > 0 else 0.0


def physical_torch(data1, data2, window=WINDOW, shift_range=SHIFT_RANGE):
    """
    PyTorch tensor implementation — GPU-friendly and autograd-compatible.
    """
    if data1.dim() > 1:
        data1 = data1.squeeze(0)
    if data2.dim() > 1:
        data2 = data2.squeeze(0)

    assert data1.dim() == 1 and data2.dim() == 1, "Only 1-D sequences are supported"

    if not torch.is_floating_point(data1):
        data1 = data1.float()
    if not torch.is_floating_point(data2):
        data2 = data2.float()
    if data1.dtype != data2.dtype:
        data2 = data2.to(dtype=data1.dtype)

    device = data1.device
    dtype = data1.dtype
    x, y = data1, data2
    T = x.shape[0]
    L = window

    if T < L:
        return x.new_tensor(0.0)

    W_x = x.unfold(dimension=0, size=L, step=1)
    W_y = y.unfold(dimension=0, size=L, step=1)
    C = W_x.shape[0]

    W_x_mean = W_x.mean(dim=1, keepdim=True)
    W_y_mean = W_y.mean(dim=1, keepdim=True)
    Xc = W_x - W_x_mean
    Yc = W_y - W_y_mean

    X_ss = torch.sum(Xc * Xc, dim=1)
    Y_ss = torch.sum(Yc * Yc, dim=1)

    S = shift_range
    shift_num = 2 * S + 1
    corr_mat = torch.full((C, shift_num), -float("inf"), dtype=dtype, device=device)

    for s in range(-S, S + 1):
        idx = s + S
        if s >= 0:
            xs, ys = Xc[0:C - s], Yc[s:C]
            xs_ss, ys_ss = X_ss[0:C - s], Y_ss[s:C]
            row_slice = slice(0, C - s)
        else:
            k = -s
            xs, ys = Xc[k:C], Yc[0:C - k]
            xs_ss, ys_ss = X_ss[k:C], Y_ss[0:C - k]
            row_slice = slice(k, C)

        num = torch.sum(xs * ys, dim=1)
        denom = torch.sqrt(xs_ss * ys_ss)
        corr = torch.zeros_like(num)
        valid = denom > 0
        corr[valid] = num[valid] / denom[valid]
        corr_mat[row_slice, idx] = corr

    max_corr, _ = torch.max(corr_mat, dim=1)
    valid_rows = max_corr != -float("inf")
    if not torch.any(valid_rows):
        return x.new_tensor(0.0)

    P = torch.sum(1.0 - max_corr[valid_rows])
    count = valid_rows.sum()
    if count.item() == 0:
        return x.new_tensor(0.0)
    return P / count


def physical(data1, data2, window=WINDOW, shift_range=SHIFT_RANGE):
    """
    Unified dispatcher — auto-selects NumPy or PyTorch based on input type.

    Returns the mean local-max correlation distance (lower = more similar).
    """
    if isinstance(data1, torch.Tensor) or isinstance(data2, torch.Tensor):
        return physical_torch(data1, data2, window=window, shift_range=shift_range)
    else:
        return physical_numpy(data1, data2, window=window, shift_range=shift_range)


# ============================================================================
#  Neural network model
# ============================================================================

class PredictModel(nn.Module):
    """
    Fully-connected network for curve prediction.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(inplace=True),
        )
        self.net = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
        )
        self.output_layer = nn.Linear(128, output_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, data):
        x = data.float()
        x = self.input_layer(x)
        x = self.net(x)
        x = self.output_layer(x)
        return x


# ============================================================================
#  Loss functions
# ============================================================================

def dps_loss(model, inputs):
    """
    DPS (Data-Physics-Smoothness) composite loss.

        loss = data_loss + physics_loss * w_p + smoothness_loss * w_s
    """
    observed, model_curve = torch.split(inputs, inputs.shape[-1] // 2, dim=-1)
    outputs = model(inputs)

    loss_data = torch.mean((observed - outputs) ** 2)
    loss_physics = physical(outputs, model_curve)
    loss_smoothness = torch.mean(torch.diff(outputs, dim=1) ** 2)

    return (
        loss_data * LOSS_DATA_WEIGHT
        + loss_physics * LOSS_PHYSICS_WEIGHT
        + loss_smoothness * LOSS_SMOOTHNESS_WEIGHT
    )


def pinn_loss(model, inputs):
    """
    PINN (Physics-Informed Neural Network) loss.

    Constrains the gradient of the output curve to match the target curve
    via automatic differentiation.

        loss = data_loss + grad_loss * 30
    """
    observed, target_curve = torch.split(inputs, inputs.shape[-1] // 2, dim=-1)

    t = torch.tensor(
        np.arange(len(observed[0])), dtype=torch.float32, requires_grad=True
    ).unsqueeze(1)
    output = model(t)

    first_deriv = torch.autograd.grad(
        outputs=output, inputs=t,
        grad_outputs=torch.ones_like(output),
        create_graph=True, retain_graph=True,
    )[0]

    target_grad = (target_curve[:, 2:] - target_curve[:, :-2]) / 2.0
    output_grad = first_deriv[1:-1]

    loss_data = torch.mean((observed.squeeze() - output.squeeze()) ** 2)
    loss_grad = torch.mean((target_grad.squeeze() - output_grad.squeeze()) ** 2)

    return loss_data + 30 * loss_grad


# ============================================================================
#  Training
# ============================================================================

def train_model(model, loss_fn, optimizer, data_loader, num_epochs, save_path):
    """
    Training loop with best-model checkpointing.

    Returns the model (loaded with best weights) and per-epoch loss history.
    """
    loss_history = []
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch in data_loader:
            inputs = batch[0]
            optimizer.zero_grad()
            loss = loss_fn(model, inputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)

        avg_loss = epoch_loss / len(data_loader.dataset)
        loss_history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.6f}, Best: {best_loss:.6f}")

    model.load_state_dict(torch.load(save_path))
    print(f"  Best model loaded (loss: {best_loss:.6f})")
    return model, np.array(loss_history)


# ============================================================================
#  Metrics
# ============================================================================

def compute_smoothness(data_obs, data_cal, data_dps):
    """
    Smoothness indicator: RMS of adjacent first-differences.
    Lower = smoother.
    """

    def _smoothness(data):
        out = []
        for series in data:
            s = 0.0
            for i in range(len(series) - 1):
                if np.isnan(series[i]) or np.isnan(series[i + 1]):
                    continue
                s += (series[i + 1] - series[i]) ** 2
            out.append(np.sqrt(s / (len(series) - 1)))
        return out

    return _smoothness(data_obs), _smoothness(data_cal), _smoothness(data_dps)


def compute_physical(data_obs, data_cal, data_dps):
    """
    Physical consistency: mean local-max correlation distance to the model curve.
    Lower = more physically consistent.
    """
    phys_obs = np.array([physical(data_obs[i], data_cal[i]) for i in range(len(data_obs))])
    phys_cal = np.array([physical(data_cal[i], data_cal[i]) for i in range(len(data_cal))])
    phys_dps = np.array([physical(data_dps[i], data_cal[i]) for i in range(len(data_dps))])
    return phys_obs, phys_cal, phys_dps


def compute_deviation(data_obs, data_cal, data_dps):
    """
    Deviation: RMSE relative to the observed data.
    Lower = closer to observations.
    """

    def _rmse(data, target):
        m1, m2 = ~np.isnan(data), ~np.isnan(target)
        mask = m1 & m2
        return np.sqrt(np.mean((data[mask] - target[mask]) ** 2))

    dev_obs = np.array([_rmse(data_obs[i], data_obs[i]) for i in range(len(data_obs))])
    dev_cal = np.array([_rmse(data_cal[i], data_obs[i]) for i in range(len(data_cal))])
    dev_dps = np.array([_rmse(data_dps[i], data_obs[i]) for i in range(len(data_dps))])
    return dev_obs, dev_cal, dev_dps


# ============================================================================
#  Utilities
# ============================================================================

def xy_interp(X, Y, x_list):
    """
    Linear interpolation at given x-coordinates.

    Returns a list of interpolated y-values.
    """
    if len(X) != len(Y):
        raise ValueError("X and Y must have the same length")
    if len(X) < 2:
        raise ValueError("X and Y must have at least 2 points for interpolation")

    result = []
    for x in x_list:
        for i in range(len(X) - 1):
            if X[i] <= x <= X[i + 1]:
                x0, y0 = X[i], Y[i]
                x1, y1 = X[i + 1], Y[i + 1]
                result.append(y0 + (y1 - y0) * (x - x0) / (x1 - x0))
                break
        else:
            if x > max(X):
                result.append(Y[np.argmax(X)])
            elif x < min(X):
                result.append(Y[np.argmin(X)])
    return result


def plot_curve(observe, model_curve, dps, save_path=None):
    """
    Plot three curves: observed vs model vs DPS output.

    If `save_path` is provided the figure is saved to disk; otherwise shown.
    """
    plt.figure()
    plt.plot(observe, "-", label="observe", linewidth=1.2, color="black")
    plt.plot(model_curve, ":", label="model", linewidth=1.2, color="blue")
    plt.plot(dps, "-", label="dps", linewidth=1.2, color="red")
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()




# ============================================================================
#  Core pipeline
# ============================================================================

def load_data():
    """Load Excel data and interpolate calculation onto observation date grid."""
    data_obs = pd.read_excel(DATA_FILE, sheet_name="observation").to_numpy()
    data_cal = pd.read_excel(DATA_FILE, sheet_name="calculation").to_numpy()

    date_obs = data_obs[:, 0]
    date_cal = data_cal[:, 0]

    UV_obs = data_obs[:, 1:]
    UV_cal = data_cal[:, 1:]
    n_cols = UV_cal.shape[1]

    UV_cal_interp = np.zeros((len(date_obs), n_cols))
    for i in range(n_cols):
        UV_cal_interp[:, i] = np.interp(date_obs, date_cal, UV_cal[:, i])

    return UV_obs, UV_cal_interp


def run_single(point_id, UV_obs, UV_cal_interp, is_train=True):
    """
    Run the full pipeline for a single data point:
    prep data -> train/load model -> inference -> metrics.
    """
    # ---- Prepare inputs ----
    model_curve = UV_cal_interp.T[[point_id], :]
    observed = UV_obs.T[[point_id], :]

    data_inputs = np.hstack((observed, model_curve))
    inputs_tensor = torch.tensor(data_inputs, dtype=torch.float32)

    # ---- Model ----
    model = PredictModel(input_dim=inputs_tensor.shape[-1], output_dim=observed.shape[-1])
    os.makedirs(MODEL_DIR, exist_ok=True)
    ckpt_path = os.path.join(MODEL_DIR, f"dps{point_id}.pth")

    if not os.path.exists(ckpt_path) or is_train:
        dataset = torch.utils.data.TensorDataset(inputs_tensor)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        model, _ = train_model(model, dps_loss, opt, loader, NUM_EPOCHS, save_path=ckpt_path)
        torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path))
    dps_output = model(inputs_tensor).detach().numpy()

    loss_val = dps_loss(model, inputs_tensor)
    print(f"  [Point {point_id}] Final dps_loss: {loss_val.item():.6f}")

    # ---- Plot & save ----
    os.makedirs(PIC_DIR, exist_ok=True)
    for i in range(len(observed)):
        save_path = os.path.join(PIC_DIR, f"point_{point_id}.png")
        plot_curve(observed[i], model_curve[i], dps_output[i], save_path=save_path)
        print(f"  [Point {point_id}] Figure saved: {save_path}")



def main(point_ids=None, is_train=True):
    """
    Main entry point — iterate over all specified point IDs.

    Parameters
    ----------
    point_ids : list[int] | None
        Point indices to process. Defaults to 0..7.
    is_train : bool
        If True, (re)train models; otherwise load existing checkpoints only.
    """
    if point_ids is None:
        point_ids = list(range(8))

    print("Loading data...")
    UV_obs, UV_cal_interp = load_data()
    print(f"Done — {UV_cal_interp.shape[1]} columns loaded, "
          f"processing {len(point_ids)} point(s)\n")

    for pid in point_ids:
        print(f"\n{'=' * 60}")
        print(f"  Processing Point {pid}")
        print(f"{'=' * 60}")
        run_single(pid, UV_obs, UV_cal_interp, is_train=is_train)
    print(f"\nAll figures saved to: {PIC_DIR}")


# ============================================================================
#  CLI entry point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data Quality Assessment — DPS Neural-Network Curve Fitting & Metrics"
    )
    parser.add_argument(
        "--ids", type=int, nargs="+", default=None,
        help="Point indices to process (default: 0 1 2 3 4 5 6 7)",
    )

    args = parser.parse_args()

    main(point_ids=args.ids, is_train= False)
