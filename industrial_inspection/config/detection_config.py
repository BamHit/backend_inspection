"""Configuration des paramètres de détection."""

from pathlib import Path

from pydantic import BaseModel


class DetectionConfig(BaseModel):
    """Configuration des seuils et paramètres de vérification qualité."""

    # Seuils de vérification
    alignment_distance_threshold: int = 20  # pixels
    alignment_angle_threshold: float = 5.0  # degrés
    illumination_tolerance: float = 0.15
    lighting_uniformity_tolerance: float = 0.2
    frame_occupation_tolerance: float = 0.15
    sharpness_tolerance: float = 0.25

    # Paramètres de segmentation
    hsv_lower_bound: list[int] = [0, 0, 50]
    hsv_upper_bound: list[int] = [180, 50, 255]

    # Pondération sigmoïde
    sigmoid_steepness: float = 10.0
    sigmoid_center_offset: float = 0.2

    @classmethod
    def from_file(cls, config_path: Path) -> "DetectionConfig":
        """Charge la configuration depuis un fichier JSON."""
        with open(config_path, encoding="utf-8") as f:
            return cls.model_validate_json(f.read())

    @classmethod
    def default(cls) -> "DetectionConfig":
        """Retourne la configuration par défaut."""
        default_path = Path(__file__).parent / "files" / "detection-config.json"
        if default_path.exists():
            return cls.from_file(default_path)
        return cls()
