from __future__ import annotations

from typing import Sequence, Dict, Any
import math


def _to_list(values) -> list[float]:
    if values is None:
        return []
    try:
        # Torch tensor
        import torch  # type: ignore

        if isinstance(values, torch.Tensor):
            return values.detach().cpu().view(-1).tolist()
    except Exception:
        pass

    # Numpy array
    try:
        import numpy as np  # type: ignore

        if isinstance(values, np.ndarray):
            return values.reshape(-1).tolist()
    except Exception:
        pass

    # Generic sequence
    if isinstance(values, Sequence):
        # Flatten one-level nested sequences if needed
        flat: list[float] = []
        for v in values:
            if isinstance(v, Sequence) and not isinstance(v, (str, bytes)):
                flat.extend(_to_list(v))
            else:
                flat.append(float(v))
        return flat

    return [float(values)]


def compute_regression_metrics(y_true: Any, y_pred: Any) -> Dict[str, float]:
    """Compute RMSE, MAE, R2, and MSE for regression.

    Accepts torch tensors, numpy arrays, or sequences.
    """
    y_true_list = _to_list(y_true)
    y_pred_list = _to_list(y_pred)

    n = min(len(y_true_list), len(y_pred_list))
    if n == 0:
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "mse": float("nan"),
        }

    y_true_vals = y_true_list[:n]
    y_pred_vals = y_pred_list[:n]

    # Errors
    sq_errors = [(p - t) * (p - t) for p, t in zip(y_pred_vals, y_true_vals)]
    abs_errors = [abs(p - t) for p, t in zip(y_pred_vals, y_true_vals)]

    mse = sum(sq_errors) / n
    rmse = math.sqrt(mse)
    mae = sum(abs_errors) / n

    mean_true = sum(y_true_vals) / n
    ss_tot = sum((t - mean_true) * (t - mean_true) for t in y_true_vals)
    ss_res = sum((t - p) * (t - p) for p, t in zip(y_pred_vals, y_true_vals))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else float("nan")

    return {"rmse": rmse, "mae": mae, "r2": r2, "mse": mse}
