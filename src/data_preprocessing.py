from pathlib import Path
from typing import Dict, List, Tuple

import os
import math
import warnings
import torch
import numpy as np
from loguru import logger
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from pymatgen.ext.matproj import MPRester

from .config import Config


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# Suppress benign CIF parsing warnings from pymatgen
warnings.filterwarnings(
    "ignore",
    message=r"^Issues encountered while parsing CIF",
    category=UserWarning,
)


def create_graph_features(structure: Structure, cutoff: float = 5.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create graph features from a pymatgen Structure.

    Returns:
        x: Node features (atomic numbers) shape [num_nodes, 1] (dtype long)
        edge_index: Edge indices shape [2, num_edges]
        edge_attr: Edge features (distances) shape [num_edges, 1]
        pos: Atomic cartesian positions shape [num_nodes, 3]
    """
    # Node features: atomic numbers
    atomic_numbers = [site.specie.Z for site in structure.sites]
    x = torch.tensor(atomic_numbers, dtype=torch.long).unsqueeze(-1)

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
            # Skip self-loops just in case
            if i == j:
                continue
            d = float(n.nn_distance)
            edge_src.append(i)
            edge_dst.append(j)
            edge_dist.append(d)

    if not edge_src:
        # In extremely small structures, fall back to fully connected (excluding self)
        num_nodes = len(structure.sites)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_src.append(i)
                    edge_dst.append(j)
                    # Euclidean distance
                    d = float(np.linalg.norm(pos[i].numpy() - pos[j].numpy()))
                    edge_dist.append(d)

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_dist, dtype=torch.float32).unsqueeze(-1)

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
            # Summary returns rich objects. Try getting a single summary by id
            summaries = mpr.summary.search(material_ids=[material_id])
            if summaries:
                entry = summaries[0]
                # Try common field names
                value = getattr(entry, 'formation_energy_per_atom', None)
                if value is None and isinstance(entry, dict):
                    value = entry.get('formation_energy_per_atom')
                if value is None:
                    # Try energy_per_atom if formation energy is unavailable
                    value = getattr(entry, 'energy_per_atom', None)
                    if value is None and isinstance(entry, dict):
                        value = entry.get('energy_per_atom')
                return float(value) if value is not None else float('nan')
    except Exception as e:
        logger.warning(f"Failed to fetch formation energy for {material_id}: {e}")
    return float('nan')


def _material_formula_from_dirname(dirname: str) -> str:
    # Directories are saved as lower-case material formulas (e.g. 'nacl')
    # Return a pretty version with capitalization guesses
    # Keep original lower-case tokenization; comparing in lower-case later
    return dirname


def prepare_datasets_from_cif() -> Dict[str, Path]:
    """Prepare PyG datasets from CIFs saved in temporary_data/cif_structures.

    Splitting strategy:
    - Samples whose formula is listed in Config.get_predict_materials_list() are reserved.
      They are split 50/50 into validation and prediction sets.
    - Remaining samples are split into train/test as 80/20.

    Returns:
        Dict with keys 'data_path' (root temporary_data), 'processed_dir', and paths to saved tensors.
    """
    cfg = Config()
    temp_dir = Path(cfg.get_temporary_data_path())
    cif_root = temp_dir / 'cif_structures'
    processed_dir = temp_dir / 'processed'
    _ensure_dir(processed_dir)

    if not cif_root.exists():
        logger.error(f"CIF directory not found: {cif_root}")
        return {
            'data_path': temp_dir,
            'processed_dir': processed_dir,
        }

    predict_materials = [m.lower() for m in cfg.get_predict_materials_list()]
    api_key = cfg.get_material_project_info()

    # Collect all samples
    from torch_geometric.data import Data
    train_samples: List[Data] = []
    val_samples: List[Data] = []
    test_samples: List[Data] = []
    predict_samples: List[Data] = []

    for material_dir in sorted([p for p in cif_root.iterdir() if p.is_dir()]):
        material_formula_lc = material_dir.name.lower()
        is_reserved = material_formula_lc in predict_materials

        cif_files = sorted(material_dir.glob('*.cif'))
        if not cif_files:
            continue

        material_samples: List[Data] = []
        for cif_fp in cif_files:
            try:
                structure = _read_cif(cif_fp)
                x, edge_index, edge_attr, pos = create_graph_features(structure)

                # Parse material_id from filename (e.g., mp-12345.cif)
                material_id = cif_fp.stem
                y_value = float('nan')
                if api_key:
                    y_value = _fetch_formation_energy(api_key, material_id)

                if math.isnan(y_value):
                    logger.warning(f"Target (formation energy) missing for {material_id}; skipping sample")
                    continue

                data_obj = Data(
                    x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos,
                    y=torch.tensor(y_value, dtype=torch.float32)
                )
                # Attach metadata for later aggregation
                data_obj.material = material_formula_lc
                data_obj.material_id = material_id
                material_samples.append(data_obj)
            except Exception as e:
                logger.warning(f"Failed to process {cif_fp}: {e}")
                continue

        if not material_samples:
            continue

        if is_reserved:
            # Split reserved samples into val and predict 50/50
            split_idx = max(1, len(material_samples) // 2)
            val_samples.extend(material_samples[:split_idx])
            predict_samples.extend(material_samples[split_idx:])
        else:
            # Add to pool for later train/test split
            # Simple split per material 80/20
            split_idx = max(1, int(0.8 * len(material_samples)))
            train_samples.extend(material_samples[:split_idx])
            test_samples.extend(material_samples[split_idx:])

    # Save tensors
    def _save_list(tensors: List[Data], name: str) -> Path:
        path = processed_dir / f"{name}.pt"
        if tensors:
            torch.save(tensors, str(path))
            logger.info(f"Saved {len(tensors)} samples to {path}")
        else:
            logger.warning(f"No samples to save for {name}; creating empty list at {path}")
            torch.save([], str(path))
        return path

    train_path = _save_list(train_samples, 'train')
    val_path = _save_list(val_samples, 'val')
    test_path = _save_list(test_samples, 'test')
    predict_path = _save_list(predict_samples, 'predict')

    return {
        'data_path': temp_dir,
        'processed_dir': processed_dir,
        'train': train_path,
        'val': val_path,
        'test': test_path,
        'predict': predict_path,
    }


__all__ = [
    'prepare_datasets_from_cif',
    'create_graph_features',
]


