"""Configuration pytest et fixtures globales."""

import numpy as np
import pytest

from industrial_inspection.config.camera_config import CameraConfig
from industrial_inspection.config.detection_config import DetectionConfig
from industrial_inspection.config.model_config import ModelConfig


def pytest_addoption(parser):
    """
    Déclare l'option --with-camera pour Pytest.
    Sans cette fonction, Pytest renvoie une erreur s'il voit cet argument.
    """
    parser.addoption(
        "--with-camera",
        action="store_true",
        default=False,
        help="Exécuter les tests nécessitant une caméra réelle",
    )


@pytest.fixture(scope="session")
def config(request):
    """
    Permet d'accéder à l'objet de configuration de pytest (request.config).
    Nécessaire pour que le marker @pytest.mark.skipif puisse évaluer la condition.
    """
    return request.config


@pytest.fixture
def camera_config():
    """Fixture pour la configuration caméra."""
    return CameraConfig.default()


@pytest.fixture
def model_config():
    """Fixture pour la configuration modèle."""
    return ModelConfig.default()


@pytest.fixture
def detection_config():
    """Fixture pour la configuration détection."""
    return DetectionConfig.default()


@pytest.fixture
def dummy_image():
    """Fixture pour une image factice."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def dummy_mask():
    """Fixture pour un masque factice."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 255  # Carré au centre
    return mask
