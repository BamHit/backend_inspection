"""Configuration de la caméra via Pydantic."""

from pathlib import Path

from pydantic import BaseModel


class ExposureConfig(BaseModel):
    """Configuration de l'exposition."""

    auto: bool = False
    value: float = -5


class FocusConfig(BaseModel):
    """Configuration du focus."""

    auto: bool = False
    value: int = 384


class WhiteBalanceConfig(BaseModel):
    """Configuration de la balance des blancs."""

    r: float = 0.95
    g: float = 1.00
    b: float = 1.05


class ProcessingConfig(BaseModel):
    """Configuration du traitement d'image."""

    brightness: int = 28
    contrast: int = 32
    saturation: int = 32
    sharpness: int = 32
    gamma: int = 32


class ResolutionConfig(BaseModel):
    """Configuration de la résolution."""

    width: int = 8000
    height: int = 6000


class NotesConfig(BaseModel):
    """Notes et métadonnées."""

    optimal_distance_cm: int = 85
    field_of_view_m2: float = 1.0
    DFOV: int = 79
    focus_optimized_for: str = "Holes with chamfer"
    default_lens: str = "EFL 4.71mm, F/NO: 1.79 6P"


class CameraConfig(BaseModel):
    """Configuration complète de la caméra."""

    camera_model: str = "Supertek ST10-48MPAF"
    cam_index: int = 1
    resolution: ResolutionConfig
    fourcc: str = "MJPG"
    fps: int = 2
    exposure: ExposureConfig
    focus: FocusConfig
    white_balance: WhiteBalanceConfig
    gain: int = 0
    processing: ProcessingConfig
    warmup_time: float = 1.0
    warmup_frames: int = 1
    notes: NotesConfig

    @classmethod
    def from_file(cls, config_path: Path) -> "CameraConfig":
        """Charge la configuration depuis un fichier JSON."""
        with open(config_path, encoding="utf-8") as f:
            return cls.model_validate_json(f.read())

    @classmethod
    def default(cls) -> "CameraConfig":
        """Retourne la configuration par défaut."""
        default_path = Path(__file__).parent / "files" / "camera-config.json"
        return cls.from_file(default_path)
