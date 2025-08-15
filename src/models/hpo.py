"""
Bayesian hyperparameter optimization utilities for GNN models.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Callable

from bayes_opt import BayesianOptimization
from loguru import logger

from .cgcnn.train import train_cgcnn
from .megnet.train import train_megnet
from .schnet.train import train_schnet
from .mpnn.train import train_mpnn


MODEL_TRAINERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "cgcnn": train_cgcnn,
    "megnet": train_megnet,
    "schnet": train_schnet,
    "mpnn": train_mpnn,
}


def _coerce_params(model_name: str, params: Dict[str, float]) -> Dict[str, Any]:
    """Convert BO float params to proper dtypes for each model.

    - Cast integer-like params
    - Keep floats for lr, dropout, weight_decay
    """
    int_keys_common = {"epochs", "batch_size", "hidden_channels"}
    if model_name in {"cgcnn", "megnet", "mpnn"}:
        int_keys = int_keys_common | {"num_layers"}
    elif model_name == "schnet":
        int_keys = int_keys_common | {"num_interactions"}
    else:
        int_keys = int_keys_common

    coerced: Dict[str, Any] = {}
    for k, v in params.items():
        if k in int_keys:
            coerced[k] = int(round(v))
        else:
            coerced[k] = float(v)
    return coerced


def optimize_model(
    model_name: str,
    data_path: str,
    bounds: Dict[str, Any],
    init_points: int = 5,
    n_iter: int = 10,
    tmp_dir: str | None = None,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Run Bayesian Optimization for a given model.

    Returns a dict with keys: best_params, best_score.
    """
    if model_name not in MODEL_TRAINERS:
        raise ValueError(f"Unknown model for HPO: {model_name}")

    trainer = MODEL_TRAINERS[model_name]

    # Prepare parameter bounds for bayesian-optimization
    pbounds: Dict[str, tuple] = {}
    for key, rng in (bounds.get("bounds", {}) or {}).items():
        if not isinstance(rng, (list, tuple)) or len(rng) != 2:
            continue
        pbounds[key] = (float(rng[0]), float(rng[1]))

    logger.info(
        f"Starting Bayesian Optimization for {model_name} with {init_points} init and {n_iter} iter"
    )

    trial_counter = {"n": 0}

    def objective(**kwargs) -> float:
        trial_counter["n"] += 1
        t = trial_counter["n"]
        params = _coerce_params(model_name, kwargs)
        # Force short training for trials
        params["epochs"] = 25

        # Route model artifacts into a per-trial directory to avoid clobbering
        save_root = tmp_dir or os.path.join("models", model_name, "hpo_trials")
        model_save_path = os.path.join(save_root, f"trial_{t:03d}")

        os.makedirs(model_save_path, exist_ok=True)

        logger.info(f"[{model_name}][HPO] Trial {t} with params: {params}")

        # Train and capture best validation loss from history
        history = trainer(
            data_path=data_path, model_save_path=model_save_path, **params
        )
        best_val = history.get("best_val_loss")
        if best_val is None and "val_loss" in history and history["val_loss"]:
            best_val = min(history["val_loss"])  # fallback

        if best_val is None:
            logger.warning("No validation loss found; returning poor score")
            return -1e9

        score = -float(best_val)  # maximize negative val loss
        logger.info(
            f"[{model_name}][HPO] Trial {t} best val loss: {best_val:.6f} -> score {score:.6f}"
        )
        return score

    optimizer = BayesianOptimization(f=objective, pbounds=pbounds, verbose=2)

    optimizer.maximize(init_points=int(init_points), n_iter=int(n_iter))

    best_params = optimizer.max["params"] if optimizer.max else {}
    best_score = optimizer.max["target"] if optimizer.max else float("nan")

    best_params = _coerce_params(model_name, best_params)

    logger.info(f"[{model_name}] HPO complete. Best score: {best_score:.6f}")
    logger.info(f"[{model_name}] Best params: {best_params}")

    return {"best_params": best_params, "best_score": best_score}
