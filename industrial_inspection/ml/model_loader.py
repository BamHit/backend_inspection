"""Chargement et singleton pour le modèle."""

import logging

import torch
import torch.nn as nn

from industrial_inspection.config.model_config import ModelConfig
from industrial_inspection.exceptions import ModelException
from industrial_inspection.ml.model_builder import build_model

logger = logging.getLogger(__name__)


class ModelSingleton:
    """Singleton pour l'instance globale du modèle."""

    _model: nn.Module | None = None
    _config: ModelConfig | None = None
    _device: torch.device | None = None

    @classmethod
    def get_model(
        cls,
        config: ModelConfig | None = None,
        device: torch.device | None = None,
        force_reload: bool = False,
    ) -> nn.Module:
        """Retourne ou charge l'instance unique du modèle.

        Args:
            config: Configuration du modèle.
            device: Device PyTorch (cpu ou cuda).
            force_reload: Forcer le rechargement.

        Returns:
            Modèle chargé et prêt pour l'inférence.
        """
        if cls._model is None or force_reload:
            if config is None:
                config = ModelConfig.default()
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            logger.info(f"Loading model: {config.architecture}")
            logger.info(f"Device: {device}")

            # Construire le modèle
            model = build_model(config)

            # Charger les poids
            model_path = config.model_path
            if not model_path.exists():
                raise ModelException(f"Model file not found: {model_path}")

            logger.info(f"Loading weights from {model_path}")
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)

            model.to(device)
            model.eval()

            cls._model = model
            cls._config = config
            cls._device = device

            logger.info("Model loaded successfully")

        return cls._model

    @classmethod
    def get_config(cls) -> ModelConfig:
        """Retourne la configuration actuelle."""
        if cls._config is None:
            raise ModelException("Model not loaded yet")
        return cls._config

    @classmethod
    def get_device(cls) -> torch.device:
        """Retourne le device actuel."""
        if cls._device is None:
            raise ModelException("Model not loaded yet")
        return cls._device

    @classmethod
    def reset(cls) -> None:
        """Réinitialise le singleton (utile pour les tests)."""
        cls._model = None
        cls._config = None
        cls._device = None
