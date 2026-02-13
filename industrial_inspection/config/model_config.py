"""Configuration du modèle ML via Pydantic."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class ModelConfig(BaseModel):
    """Configuration du modèle de classification."""

    # Architecture
    architecture: Literal[
        "small_cnn",
        "medium_cnn",
        "large_cnn",
        "residual_cnn",
        "compact_cnn",
        "efficientnet_b0",
        "resnet18",
    ] = "large_cnn"

    # Paramètres
    num_classes: int = 6
    input_size: int = 128
    confidence_threshold: float = 0.1

    # Classes
    class_names: list[str] = [
        "Thread",
        "Hole",
        "Nuts",
        "Push_In_Rivet",
        "Rivets",
        "Foreign_Object",
    ]

    # Chemin du modèle
    model_path: Path = Path("model_large_cnn_6classes_130126.pth")

    @classmethod
    def from_file(cls, config_path: Path) -> "ModelConfig":
        """Charge la configuration depuis un fichier JSON."""
        with open(config_path, encoding="utf-8") as f:
            return cls.model_validate_json(f.read())

    @classmethod
    def default(cls) -> "ModelConfig":
        """Retourne la configuration par défaut."""
        default_path = Path(__file__).parent / "files" / "model-config.json"
        if default_path.exists():
            return cls.from_file(default_path)
        return cls()  # Fallback sur valeurs par défaut
