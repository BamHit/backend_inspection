"""Dépendances FastAPI (Dependency Injection)."""

from industrial_inspection.camera.controller import CameraController
from industrial_inspection.camera.singleton import CameraSingleton
from industrial_inspection.config.camera_config import CameraConfig
from industrial_inspection.config.detection_config import DetectionConfig
from industrial_inspection.config.model_config import ModelConfig


def get_camera() -> CameraController:
    """Dépendance pour obtenir l'instance de la caméra."""
    return CameraSingleton.get_instance()


def get_camera_config() -> CameraConfig:
    """Dépendance pour obtenir la config caméra."""
    return CameraConfig.default()


def get_model_config() -> ModelConfig:
    """Dépendance pour obtenir la config modèle."""
    return ModelConfig.default()


def get_detection_config() -> DetectionConfig:
    """Dépendance pour obtenir la config détection."""
    return DetectionConfig.default()
