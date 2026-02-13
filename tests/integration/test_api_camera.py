"""Tests d'intégration pour les routes caméra."""

import pytest
from fastapi.testclient import TestClient

from industrial_inspection.api.main import app


@pytest.fixture
def client():
    """Client de test FastAPI."""
    return TestClient(app)


def test_get_camera_status(client):
    """Test de l'endpoint GET /camera/status."""
    response = client.get("/camera/status")
    assert response.status_code == 200
    data = response.json()
    assert "is_initialized" in data
    assert "camera_model" in data


# Note: Les tests nécessitant une vraie caméra doivent être marqués
@pytest.mark.skipif(
    "not config.getoption('--with-camera')", reason="Requires actual camera hardware"
)
def test_capture_image(client):
    """Test de l'endpoint POST /camera/capture."""
    response = client.post("/camera/capture")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "image_base64" in data
