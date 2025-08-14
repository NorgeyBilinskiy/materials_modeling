from typing import Dict, List

from loguru import logger

from .data_preprocessing import _read_cif
from .models import (
    predict_cgcnn,
    predict_megnet,
    predict_schnet,
    predict_mpnn,
)

from .config import Config
from pathlib import Path


def run_predictions_for_selected_materials() -> Dict[str, Dict[str, float]]:
    """Run predictions on CIFs for materials from prediction list.

    Returns mapping model_name -> material_formula -> average predicted value across CIFs.
    """
    cfg = Config()
    predict_materials = [m.lower() for m in cfg.get_predict_materials_list()]
    cif_root = Path(cfg.get_temporary_data_path()) / "cif_structures"

    if not predict_materials:
        logger.warning("No materials selected for prediction in config")
        return {}

    if not cif_root.exists():
        logger.error(f"CIF directory not found: {cif_root}")
        return {}

    def _predict_for_material(material_dir: Path) -> Dict[str, float]:
        from pymatgen.core import Structure

        structures: List[Structure] = []
        for cif_fp in sorted(material_dir.glob("*.cif")):
            try:
                structures.append(_read_cif(cif_fp))
            except Exception as e:
                logger.warning(f"Failed to read CIF {cif_fp}: {e}")
        if not structures:
            return {}

        # Use multi-structure predictors if available; else fall back to single
        values: Dict[str, float] = {}
        try:
            from src.models.cgcnn.predict import (
                predict_multiple_structures as cgcnn_multi,
            )

            preds = cgcnn_multi("models/cgcnn/best_model.pth", structures)
            if preds:
                values["cgcnn"] = float(sum(preds) / len(preds))
        except Exception as e:
            try:
                values["cgcnn"] = float(predict_cgcnn())
            except Exception as ee:
                logger.error(f"CGCNN prediction failed: {e or ee}")

        try:
            from src.models.megnet.predict import (
                predict_multiple_structures as megnet_multi,
            )

            preds = megnet_multi("models/megnet/best_model.pth", structures)
            if preds:
                values["megnet"] = float(sum(preds) / len(preds))
        except Exception as e:
            try:
                values["megnet"] = float(predict_megnet())
            except Exception as ee:
                logger.error(f"MEGNet prediction failed: {e or ee}")

        try:
            from src.models.schnet.predict import (
                predict_multiple_structures as schnet_multi,
            )

            preds = schnet_multi("models/schnet/best_model.pth", structures)
            if preds:
                values["schnet"] = float(sum(preds) / len(preds))
        except Exception as e:
            try:
                values["schnet"] = float(predict_schnet())
            except Exception as ee:
                logger.error(f"SchNet prediction failed: {e or ee}")

        try:
            from src.models.mpnn.predict import (
                predict_multiple_structures as mpnn_multi,
            )

            preds = mpnn_multi("models/mpnn/best_model.pth", structures)
            if preds:
                values["mpnn"] = float(sum(preds) / len(preds))
        except Exception as e:
            try:
                values["mpnn"] = float(predict_mpnn())
            except Exception as ee:
                logger.error(f"MPNN prediction failed: {e or ee}")

        return values

    results: Dict[str, Dict[str, float]] = {
        "cgcnn": {},
        "megnet": {},
        "schnet": {},
        "mpnn": {},
    }

    for material in predict_materials:
        material_dir = cif_root / material
        if not material_dir.exists():
            logger.warning(f"No CIFs found for prediction material: {material}")
            continue
        vals = _predict_for_material(material_dir)
        for model_name, val in vals.items():
            results.setdefault(model_name, {})[material] = val

    return results


__all__ = ["run_predictions_for_selected_materials"]
