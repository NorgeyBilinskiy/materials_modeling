from typing import Dict, Any

from loguru import logger

from .data_preprocessing import prepare_datasets_from_cif
from .config import Config
from .models.hpo import optimize_model
from .models import (
    train_cgcnn,
    train_megnet,
    train_schnet,
    train_mpnn,
)


def train_and_validate_all_models(config: Config = None) -> Dict[str, Any]:
    """Prepare datasets, then train and validate all models.

    Args:
        config: Optional Config object. If None, creates a new one.

    Returns mapping model_name -> training history dict.
    """
    logger.info("Preparing datasets from CIF files...")
    paths = prepare_datasets_from_cif()
    data_path = str(paths.get("data_path"))

    logger.info("Starting training for all models...")
    histories: Dict[str, Any] = {}

    # Load hyperparameter limits and BO config
    if config is None:
        config = Config()

    hpo_cfg = config.get_model_hparam_limits()
    bo_settings = hpo_cfg.get("bayes_optimization", {"init_points": 5, "n_iter": 10})
    model_bounds = hpo_cfg.get("models", {})

    # Load reproducibility config
    reproducibility_config = config.get_reproducibility_config()
    random_seed = reproducibility_config.get("random_seed", 42)

    # Load models config for training epochs
    models_config = config.get_models_config()
    final_training_epochs = models_config.get("training_epochs", {}).get(
        "final_training", 500
    )

    try:
        if model_bounds.get("cgcnn", {}).get("enabled", True):
            hpo_res = optimize_model(
                "cgcnn",
                data_path=data_path,
                bounds=model_bounds.get("cgcnn", {}),
                init_points=bo_settings.get("init_points", 5),
                n_iter=bo_settings.get("n_iter", 10),
                random_seed=random_seed,
            )
            best = hpo_res.get("best_params", {})
            # Train best model longer
            best["epochs"] = final_training_epochs
            best["random_seed"] = random_seed
            histories["cgcnn"] = train_cgcnn(data_path=data_path, **best)
        else:
            histories["cgcnn"] = train_cgcnn(
                data_path=data_path, random_seed=random_seed
            )
    except Exception as e:
        logger.error(f"CGCNN training failed: {e}")
        histories["cgcnn"] = {"error": str(e)}

    try:
        if model_bounds.get("megnet", {}).get("enabled", True):
            hpo_res = optimize_model(
                "megnet",
                data_path=data_path,
                bounds=model_bounds.get("megnet", {}),
                init_points=bo_settings.get("init_points", 5),
                n_iter=bo_settings.get("n_iter", 10),
                random_seed=random_seed,
            )
            best = hpo_res.get("best_params", {})
            best["epochs"] = final_training_epochs
            best["random_seed"] = random_seed
            histories["megnet"] = train_megnet(data_path=data_path, **best)
        else:
            histories["megnet"] = train_megnet(
                data_path=data_path, random_seed=random_seed
            )
    except Exception as e:
        logger.error(f"MEGNet training failed: {e}")
        histories["megnet"] = {"error": str(e)}

    try:
        if model_bounds.get("schnet", {}).get("enabled", True):
            hpo_res = optimize_model(
                "schnet",
                data_path=data_path,
                bounds=model_bounds.get("schnet", {}),
                init_points=bo_settings.get("init_points", 5),
                n_iter=bo_settings.get("n_iter", 10),
                random_seed=random_seed,
            )
            best = hpo_res.get("best_params", {})
            best["epochs"] = final_training_epochs
            best["random_seed"] = random_seed
            histories["schnet"] = train_schnet(data_path=data_path, **best)
        else:
            histories["schnet"] = train_schnet(
                data_path=data_path, random_seed=random_seed
            )
    except Exception as e:
        logger.error(f"SchNet training failed: {e}")
        histories["schnet"] = {"error": str(e)}

    try:
        if model_bounds.get("mpnn", {}).get("enabled", True):
            hpo_res = optimize_model(
                "mpnn",
                data_path=data_path,
                bounds=model_bounds.get("mpnn", {}),
                init_points=bo_settings.get("init_points", 5),
                n_iter=bo_settings.get("n_iter", 10),
                random_seed=random_seed,
            )
            best = hpo_res.get("best_params", {})
            best["epochs"] = final_training_epochs
            best["random_seed"] = random_seed
            histories["mpnn"] = train_mpnn(data_path=data_path, **best)
        else:
            histories["mpnn"] = train_mpnn(data_path=data_path, random_seed=random_seed)
    except Exception as e:
        logger.error(f"MPNN training failed: {e}")
        histories["mpnn"] = {"error": str(e)}

    logger.info("Training complete for all models")
    return histories


__all__ = ["train_and_validate_all_models"]
