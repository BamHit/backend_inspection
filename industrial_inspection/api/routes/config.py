"""Routes pour la configuration."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from industrial_inspection.api.dependencies import (
    get_camera_config,
    get_detection_config,
    get_model_config,
)
from industrial_inspection.config.camera_config import CameraConfig
from industrial_inspection.config.detection_config import DetectionConfig
from industrial_inspection.config.model_config import ModelConfig
from industrial_inspection.ml.model_loader import ModelSingleton

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/config", tags=["Configuration"])


@router.get("/model")
async def get_model_config_route(config: ModelConfig = Depends(get_model_config)):
    """Retourne la configuration du modèle."""
    try:
        model = ModelSingleton.get_model()
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            "config": config.model_dump(),
            "model_stats": {
                "total_parameters": num_params,
                "trainable_parameters": num_trainable,
                "device": str(ModelSingleton.get_device()),
            },
        }
    except Exception as err:
        logger.error(f"Error getting model config: {err}")
        raise HTTPException(status_code=500, detail=str(err)) from err


@router.get("/camera")
async def get_camera_config_route(config: CameraConfig = Depends(get_camera_config)):
    """Retourne la configuration de la caméra."""
    return {"config": config.model_dump()}


@router.get("/detection")
async def get_detection_config_route(
    config: DetectionConfig = Depends(get_detection_config),
):
    """Retourne la configuration de détection."""
    return {"config": config.model_dump()}


@router.get("/classes")
async def get_classes(config: ModelConfig = Depends(get_model_config)):
    """Retourne la liste des classes disponibles."""
    return {
        "num_classes": config.num_classes,
        "classes": config.class_names,
        "architecture": config.architecture,
        "input_size": config.input_size,
    }


@router.post("/threshold")
async def set_threshold(threshold: float):
    """Modifie le seuil de confiance."""
    if not 0.0 <= threshold <= 1.0:
        raise HTTPException(status_code=400, detail="Threshold must be between 0.0 and 1.0")

    config = ModelSingleton.get_config()
    old_threshold = config.confidence_threshold
    config.confidence_threshold = threshold

    logger.info(f"Threshold updated: {old_threshold:.2f} → {threshold:.2f}")

    return {
        "status": "success",
        "old_threshold": old_threshold,
        "new_threshold": threshold,
        "message": f"Threshold updated to {threshold}",
    }


@router.get("/threshold")
async def get_threshold(config: ModelConfig = Depends(get_model_config)):
    """Retourne le seuil de confiance actuel."""
    return {
        "threshold": config.confidence_threshold,
        "description": "Minimum probability to accept a prediction",
    }
