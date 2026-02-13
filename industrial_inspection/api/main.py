"""Point d'entrée FastAPI."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from industrial_inspection.api.routes import camera, config, export, health, prediction
from industrial_inspection.camera.singleton import CameraSingleton
from industrial_inspection.config.camera_config import CameraConfig
from industrial_inspection.config.model_config import ModelConfig
from industrial_inspection.ml.model_loader import ModelSingleton

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Industrial Inspection API",
    description="API de classification multi-classe pour inspection industrielle",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.on_event("startup")
async def startup_event():
    """Initialise les ressources au démarrage."""
    logger.info("=" * 70)
    logger.info("INITIALISATION DE L'API")
    logger.info("=" * 70)

    # Charger la caméra
    try:
        camera_config = CameraConfig.default()
        camera = CameraSingleton.get_instance(camera_config)
        camera.initialize()
        logger.info("✅ Caméra initialisée")
    except Exception as e:
        logger.error(f"❌ Erreur caméra: {e}")

    # Charger le modèle
    try:
        model_config = ModelConfig.default()
        ModelSingleton.get_model(model_config)
        logger.info("✅ Modèle chargé")
    except Exception as e:
        logger.error(f"❌ Erreur modèle: {e}")

    logger.info("=" * 70)


@app.on_event("shutdown")
async def shutdown_event():
    """Libère les ressources à l'arrêt."""
    logger.info("Arrêt de l'API - Libération des ressources")
    # Libérer la caméra si nécessaire


# Inclure les routes
app.include_router(camera.router)
app.include_router(prediction.router)
app.include_router(config.router)
app.include_router(health.router)
app.include_router(export.router)
