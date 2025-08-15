import math
import warnings
from typing import Dict, List, Tuple, Any
from pathlib import Path

import torch
import numpy as np
from loguru import logger
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.io.cif import CifParser
from pymatgen.ext.matproj import MPRester
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from torch_geometric.data import Data

from .config import Config


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


warnings.filterwarnings(
    "ignore",
    message=r"^Issues encountered while parsing CIF",
    category=UserWarning,
)


def create_graph_features(
    structure: Structure, cutoff: float = None, min_edges: int = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create graph features from a pymatgen Structure.

    Args:
        structure: Pymatgen Structure object
        cutoff: Radius for neighbor search (in Angstroms). If None, uses default from config.
        min_edges: Minimum number of edges. If None, uses default from config.

    Returns:
        x: Node features matrix [num_nodes, F]. Column 0 holds atomic number (float).
        edge_index: Edge indices shape [2, num_edges]
        edge_attr: Edge features matrix [num_edges, E] with distance and RBFs
        pos: Atomic cartesian positions shape [num_nodes, 3]
    """
    if cutoff is None or min_edges is None:
        try:
            cfg = Config()
            preprocessing_config = cfg.get_preprocessing_config()
            graph_config = preprocessing_config.get("graph_features", {})
            cutoff = cutoff or graph_config.get("neighbor_cutoff", 5.0)
            min_edges = min_edges or graph_config.get("min_edges", 1)
        except Exception as e:
            logger.warning(f"Failed to load config, using defaults: {e}")
            cutoff = cutoff or 5.0
            min_edges = min_edges or 1

    # Configuration for graph features
    rbf_num = 32
    rbf_gamma = None
    try:
        cfg = Config()
        preprocessing_config = cfg.get_preprocessing_config()
        graph_cfg = preprocessing_config.get("graph_features", {})
        rbf_cfg = graph_cfg.get("rbf", {})
        rbf_num = int(rbf_cfg.get("num_gaussians", 32))
        rbf_gamma = rbf_cfg.get("gamma", None)
    except Exception:
        pass

    # Node features: atomic descriptors
    node_features: List[List[float]] = []
    for site in structure.sites:
        Z = float(site.specie.Z)
        el: Element = Element.from_Z(int(Z))
        # Collect a compact set of stable descriptors
        group = (
            float(el.group) if getattr(el, "group", None) is not None else float("nan")
        )
        period = float(el.row) if getattr(el, "row", None) is not None else float("nan")
        en = float(el.X) if el.X is not None else float("nan")
        cov_r = (
            float(el.atomic_radius) if el.atomic_radius is not None else float("nan")
        )
        mass = float(el.atomic_mass) if el.atomic_mass is not None else float("nan")
        mende = float(el.mendeleev_no) if el.mendeleev_no is not None else float("nan")
        node_features.append([Z, group, period, en, cov_r, mass, mende])

    # Replace NaNs with per-feature means, then remaining with zeros
    x_np = np.array(node_features, dtype=np.float32)
    col_means = np.nanmean(x_np, axis=0)
    inds = np.where(np.isnan(x_np))
    x_np[inds] = np.take(col_means, inds[1])
    x_np = np.nan_to_num(x_np, nan=0.0, posinf=0.0, neginf=0.0)
    x = torch.tensor(x_np, dtype=torch.float32)

    # Positions
    pos = torch.tensor(np.array(structure.cart_coords, dtype=np.float32))

    # Build edges using neighbor search within cutoff
    edge_src: List[int] = []
    edge_dst: List[int] = []
    edge_dist: List[float] = []

    # Use simple neighbor search via get_neighbors
    for i, site in enumerate(structure.sites):
        neighbors = structure.get_neighbors(site, r=cutoff)
        for n in neighbors:
            j = n.index
            if i == j:
                continue
            d = float(n.nn_distance)
            edge_src.append(i)
            edge_dst.append(j)
            edge_dist.append(d)

    if len(edge_src) < min_edges:
        num_nodes = len(structure.sites)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_src.append(i)
                    edge_dst.append(j)
                    # Euclidean distance
                    d = float(np.linalg.norm(pos[i].numpy() - pos[j].numpy()))
                    edge_dist.append(d)

        logger.debug(
            f"Created {len(edge_src)} edges for structure with {num_nodes} nodes (fallback to fully connected)"
        )

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)

    # Edge attributes: distance + RBF expansion
    if len(edge_dist) == 0:
        edge_attr = torch.zeros((0, 1 + rbf_num), dtype=torch.float32)
    else:
        dists = np.array(edge_dist, dtype=np.float32)
        # RBF centers from 0 to cutoff
        centers = np.linspace(0.0, float(cutoff), num=rbf_num, dtype=np.float32)
        if rbf_gamma is None:
            # A reasonable default so neighboring centers overlap
            delta = centers[1] - centers[0] if rbf_num > 1 else max(1.0, float(cutoff))
            rbf_gamma_val = float(1.0 / (2 * (delta**2)))
        else:
            rbf_gamma_val = float(rbf_gamma)
        # Compute Gaussian RBFs
        rbf = np.exp(-rbf_gamma_val * (dists[:, None] - centers[None, :]) ** 2)
        edge_attr_np = np.concatenate([dists[:, None], rbf], axis=1)
        edge_attr = torch.tensor(edge_attr_np, dtype=torch.float32)

    return x, edge_index, edge_attr, pos


def _read_cif(cif_path: Path) -> Structure:
    parser = CifParser(str(cif_path))
    # Use modern API; primitive=True to match older get_structures default
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Issues encountered while parsing CIF",
            module="pymatgen.io.cif",
        )
        structures = parser.parse_structures(primitive=True)
    if not structures:
        raise ValueError(f"No structures parsed from {cif_path}")
    return structures[0]


def _fetch_formation_energy(api_key: str, material_id: str) -> float:
    """Fetch formation energy per atom for given Materials Project ID.
    Returns NaN if not available.
    """
    try:
        with MPRester(api_key) as mpr:
            summaries = mpr.summary.search(material_ids=[material_id])
            if summaries:
                entry = summaries[0]
                logger.debug(f"Processing entry for {material_id}: {type(entry)}")

                value = getattr(entry, "formation_energy_per_atom", None)
                if value is not None:
                    logger.info(
                        f"Found formation_energy_per_atom for {material_id}: {value} eV/atom"
                    )
                    return float(value)

                if isinstance(entry, dict):
                    value = entry.get("formation_energy_per_atom")
                    if value is not None:
                        logger.info(
                            f"Found formation_energy_per_atom (dict) for {material_id}: {value} eV/atom"
                        )
                        return float(value)

                value = getattr(entry, "energy_per_atom", None)
                if value is not None:
                    logger.info(
                        f"Using energy_per_atom (fallback) for {material_id}: {value} eV/atom"
                    )
                    return float(value)

                if isinstance(entry, dict):
                    value = entry.get("energy_per_atom")
                    if value is not None:
                        logger.info(
                            f"Using energy_per_atom (dict fallback) for {material_id}: {value} eV/atom"
                        )
                        return float(value)

                logger.warning(f"No energy values found for {material_id} in any field")
                return float("nan")
            else:
                logger.warning(f"No summaries found for {material_id}")
                return float("nan")
    except Exception as e:
        logger.warning(f"Failed to fetch formation energy for {material_id}: {e}")
    return float("nan")


def _material_formula_from_dirname(dirname: str) -> str:
    # Directories are saved as lower-case material formulas (e.g. 'nacl')
    # Return a pretty version with capitalization guesses
    # Keep original lower-case tokenization; comparing in lower-case later
    return dirname


def _create_scaler(config: Dict[str, Any]) -> Any:
    """Create a scaler based on configuration.

    Args:
        config: Configuration dictionary with scaling parameters

    Returns:
        Fitted scaler object
    """
    scaling_config = config.get("target_scaling", {})
    method = scaling_config.get("scaling_method", "standard")

    if method == "standard":
        return StandardScaler()
    elif method == "minmax":
        feature_range = scaling_config.get("minmax", {}).get("feature_range", [0, 1])
        return MinMaxScaler(feature_range=tuple(feature_range))
    elif method == "robust":
        quantile_range = scaling_config.get("robust", {}).get(
            "quantile_range", [25.0, 75.0]
        )
        return RobustScaler(quantile_range=tuple(quantile_range))
    else:
        logger.warning(f"Unknown scaling method: {method}, using StandardScaler")
        return StandardScaler()


def _apply_scaling(
    data_samples: List[Any], scaler: Any, config: Dict[str, Any]
) -> List[Any]:
    """Apply scaling to target values in data samples.

    Args:
        data_samples: List of PyG Data objects
        scaler: Fitted scaler object
        config: Configuration dictionary

    Returns:
        List of scaled PyG Data objects
    """
    if not data_samples:
        return data_samples

    scaling_config = config.get("target_scaling", {})
    if not scaling_config.get("scale_formation_energy", True):
        logger.info("Target scaling disabled, returning original data")
        return data_samples

    targets = [sample.y.item() for sample in data_samples]

    targets_scaled = scaler.fit_transform(np.array(targets).reshape(-1, 1)).flatten()

    for i, sample in enumerate(data_samples):
        sample.y = torch.tensor(targets_scaled[i], dtype=torch.float32)

    logger.info(
        f"Applied {scaler.__class__.__name__} scaling to {len(data_samples)} samples"
    )
    logger.info(
        f"Target range: {targets_scaled.min():.4f} to {targets_scaled.max():.4f}"
    )

    return data_samples


def prepare_datasets_from_cif() -> Dict[str, Path]:
    """Prepare PyG datasets from CIFs saved in temporary_data/cif_structures.

    Splitting strategy:
    - Samples whose formula is listed in Config.get_predict_materials_list() are reserved.
      Their structures are explicitly assigned to validation/predict sets in predict_material.yaml.
    - Remaining samples are split into train/test according to configuration (default: 80/20).
    - For NaCl: structures are explicitly assigned - no automatic splitting.
    - Target values (formation energy) can be scaled according to configuration.

    Returns:
        Dict with keys 'data_path' (root temporary_data), 'processed_dir', and paths to saved tensors.
    """
    cfg = Config()
    temp_dir = Path(cfg.get_temporary_data_path())
    cif_root = temp_dir / "cif_structures"
    processed_dir = temp_dir / "processed"
    _ensure_dir(processed_dir)

    if not cif_root.exists():
        logger.error(f"CIF directory not found: {cif_root}")
        return {
            "data_path": temp_dir,
            "processed_dir": processed_dir,
        }

    predict_config = cfg.get_predict_materials_config()
    preprocessing_config = cfg.get_preprocessing_config()
    api_key = cfg.get_material_project_info()

    regular_config = preprocessing_config.get("data_splitting", {}).get(
        "regular_materials", {}
    )

    train_ratio = regular_config.get("train_ratio", 0.8)
    test_ratio = regular_config.get("test_ratio", 0.2)

    logger.info(
        f"Data splitting ratios - Regular: train={train_ratio:.1%}, test={test_ratio:.1%}"
    )
    logger.info(
        "Reserved materials: structures assigned explicitly in predict_material.yaml"
    )

    train_samples: List[Data] = []
    val_samples: List[Data] = []
    test_samples: List[Data] = []
    predict_samples: List[Data] = []

    for material_dir in sorted([p for p in cif_root.iterdir() if p.is_dir()]):
        material_formula_lc = material_dir.name.lower()

        # Check if this material is in predict list
        predict_material_config = None
        for config in predict_config:
            if config["formula"].lower() == material_formula_lc:
                predict_material_config = config
                break

        is_reserved = predict_material_config is not None

        # For NaCl: only preferred structures go to predict, others go to train/val/test
        if material_formula_lc == "nacl" and predict_material_config:
            # This is NaCl with preferred structures
            pass  # Will be handled in the filtering logic below
        elif material_formula_lc == "nacl":
            # This is NaCl but not in predict list - treat as regular material
            is_reserved = False

        cif_files = sorted(material_dir.glob("*.cif"))
        if not cif_files:
            continue

        material_samples: List[Data] = []

        if is_reserved and predict_material_config.get("structures"):
            structures_config = predict_material_config["structures"]
            validation_ids = [
                str(s).lower() for s in structures_config.get("validation", [])
            ]
            predict_ids = [str(s).lower() for s in structures_config.get("predict", [])]

            for cif_fp in cif_files:
                material_id = cif_fp.stem.lower()

                try:
                    structure = _read_cif(cif_fp)
                    x, edge_index, edge_attr, pos = create_graph_features(structure)

                    y_value = float("nan")
                    if api_key:
                        y_value = _fetch_formation_energy(api_key, cif_fp.stem)

                    if math.isnan(y_value):
                        logger.warning(
                            f"Target (formation energy) missing for {cif_fp.stem}; skipping sample"
                        )
                        continue

                    data_obj = Data(
                        x=x,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        pos=pos,
                        y=torch.tensor(y_value, dtype=torch.float32),
                    )
                    data_obj.material = material_formula_lc
                    data_obj.material_id = cif_fp.stem

                    if material_id in validation_ids:
                        val_samples.append(data_obj)
                        logger.info(
                            f"Explicitly assigned {cif_fp.stem} to VALIDATION set"
                        )
                    elif material_id in predict_ids:
                        predict_samples.append(data_obj)
                        logger.info(f"Explicitly assigned {cif_fp.stem} to PREDICT set")
                    else:
                        current_train_count = len(
                            [
                                s
                                for s in train_samples
                                if s.material == material_formula_lc
                            ]
                        )
                        if (
                            current_train_count
                            < len(
                                [
                                    f
                                    for f in cif_files
                                    if f.stem.lower()
                                    not in validation_ids + predict_ids
                                ]
                            )
                            * train_ratio
                        ):
                            train_samples.append(data_obj)
                            logger.info(f"Added unassigned {cif_fp.stem} to train set")
                        else:
                            test_samples.append(data_obj)
                            logger.info(f"Added unassigned {cif_fp.stem} to test set")

                except Exception as e:
                    logger.warning(f"Failed to process {cif_fp}: {e}")
                    continue

            # Skip the main processing loop for reserved materials
            continue
        else:
            # For non-reserved materials, process normally
            filtered_cif_files = cif_files

        for cif_fp in filtered_cif_files:
            try:
                structure = _read_cif(cif_fp)
                x, edge_index, edge_attr, pos = create_graph_features(structure)

                # Parse material_id from filename (e.g., mp-12345.cif)
                material_id = cif_fp.stem
                y_value = float("nan")
                if api_key:
                    y_value = _fetch_formation_energy(api_key, material_id)

                if math.isnan(y_value):
                    logger.warning(
                        f"Target (formation energy) missing for {material_id}; skipping sample"
                    )
                    continue

                data_obj = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    pos=pos,
                    y=torch.tensor(y_value, dtype=torch.float32),
                )
                data_obj.material = material_formula_lc
                data_obj.material_id = material_id
                material_samples.append(data_obj)
            except Exception as e:
                logger.warning(f"Failed to process {cif_fp}: {e}")
                continue

        if not material_samples:
            continue

        if is_reserved:
            logger.info(
                f"Reserved material {material_formula_lc}: structures assigned explicitly"
            )
        else:
            split_idx = max(1, int(train_ratio * len(material_samples)))
            train_samples.extend(material_samples[:split_idx])
            test_samples.extend(material_samples[split_idx:])

            train_ids = [s.material_id for s in material_samples[:split_idx]]
            test_ids = [s.material_id for s in material_samples[split_idx:]]
            logger.info(
                f"Regular material {material_formula_lc}: {len(train_ids)} to train ({train_ids}), {len(test_ids)} to test ({test_ids})"
            )

    if preprocessing_config.get("target_scaling", {}).get(
        "scale_formation_energy", True
    ):
        logger.info("Applying target scaling to all datasets...")

        scaler = _create_scaler(preprocessing_config)
        train_samples = _apply_scaling(train_samples, scaler, preprocessing_config)
        val_samples = _apply_scaling(val_samples, scaler, preprocessing_config)
        test_samples = _apply_scaling(test_samples, scaler, preprocessing_config)
        predict_samples = _apply_scaling(predict_samples, scaler, preprocessing_config)

        scaler_path = processed_dir / "target_scaler.pkl"
        import pickle

        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        logger.info(f"Saved target scaler to {scaler_path}")
    else:
        logger.info("Target scaling disabled, using original values")

    # Save tensors
    def _save_list(tensors: List[Data], name: str) -> Path:
        path = processed_dir / f"{name}.pt"
        if tensors:
            torch.save(tensors, str(path))
            logger.info(f"Saved {len(tensors)} samples to {path}")
        else:
            logger.warning(
                f"No samples to save for {name}; creating empty list at {path}"
            )
            torch.save([], str(path))
        return path

    train_path = _save_list(train_samples, "train")
    val_path = _save_list(val_samples, "val")
    test_path = _save_list(test_samples, "test")
    predict_path = _save_list(predict_samples, "predict")

    return {
        "data_path": temp_dir,
        "processed_dir": processed_dir,
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "predict": predict_path,
    }


__all__ = [
    "prepare_datasets_from_cif",
    "create_graph_features",
    "_create_scaler",
    "_apply_scaling",
]
