"""Exceptions personnalisées pour l'application."""

from dataclasses import dataclass


@dataclass
class IndustrialInspectionException(Exception):
    """Exception de base pour toute l'application."""

    message: str

    def __str__(self) -> str:
        return self.message


@dataclass
class CameraException(IndustrialInspectionException):
    """Exception liée à la caméra."""

    pass


@dataclass
class ModelException(IndustrialInspectionException):
    """Exception liée au modèle ML."""

    pass


@dataclass
class QualityCheckException(IndustrialInspectionException):
    """Exception liée aux vérifications qualité."""

    pass


@dataclass
class ConfigurationException(IndustrialInspectionException):
    """Exception liée à la configuration."""

    pass
