"""Contracts pour la caméra."""


from pydantic import BaseModel


class CameraStatus(BaseModel):
    """Statut de la caméra."""

    is_initialized: bool
    camera_model: str
    resolution: str
    error: str | None = None

    class Config:
        arbitrary_types_allowed = True


class CaptureResult(BaseModel):
    """Résultat d'une capture."""

    success: bool
    image_base64: str | None = None
    width: int | None = None
    height: int | None = None
    resolution_match: bool = False
    error: str | None = None

    class Config:
        arbitrary_types_allowed = True
