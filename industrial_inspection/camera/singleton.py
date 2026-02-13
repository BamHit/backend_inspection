"""Singleton pour la gestion globale de la caméra."""


from industrial_inspection.camera.controller import CameraController
from industrial_inspection.config.camera_config import CameraConfig


class CameraSingleton:
    """Singleton pour l'instance globale de la caméra."""

    _instance: CameraController | None = None

    @classmethod
    def get_instance(cls, config: CameraConfig | None = None) -> CameraController:
        """Retourne ou crée l'instance unique de la caméra."""
        if cls._instance is None:
            if config is None:
                config = CameraConfig.default()
            cls._instance = CameraController(config)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Réinitialise le singleton (utile pour les tests)."""
        if cls._instance is not None:
            # Libérer la caméra si nécessaire
            cls._instance = None
