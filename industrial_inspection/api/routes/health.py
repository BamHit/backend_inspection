"""Route pour le health check."""

import logging

from fastapi import APIRouter

from industrial_inspection.camera.singleton import CameraSingleton
from industrial_inspection.ml.model_loader import ModelSingleton

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check():
    """Health check de l'API."""
    try:
        # Vérifier la caméra
        camera_status = "unknown"
        try:
            camera = CameraSingleton.get_instance()
            camera_status = "initialized" if camera.is_initialized else "not_initialized"
        except Exception:
            camera_status = "error"

        # Vérifier le modèle
        model_status = "unknown"
        try:
            model = ModelSingleton.get_model()
            model_status = "loaded" if model is not None else "not_loaded"
        except Exception:
            model_status = "error"

        overall_status = (
            "healthy" if camera_status == "initialized" and model_status == "loaded" else "degraded"
        )

        return {
            "status": overall_status,
            "camera": camera_status,
            "model": model_status,
            "message": "API is running",
        }

    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return {
            "status": "unhealthy",
            "camera": "unknown",
            "model": "unknown",
            "message": str(e),
        }


@router.get("/")
async def root():
    """Informations sur l'API."""
    try:
        config = ModelSingleton.get_config()
        device = ModelSingleton.get_device()

        return {
            "status": "running",
            "application": "Industrial Inspection API",
            "version": "2.0.0",
            "architecture": config.architecture,
            "num_classes": config.num_classes,
            "classes": config.class_names,
            "confidence_threshold": config.confidence_threshold,
            "device": str(device),
            "input_size": config.input_size,
        }
    except Exception:
        return {
            "status": "running",
            "application": "Industrial Inspection API",
            "version": "2.0.0",
            "message": "Model not loaded yet",
        }
