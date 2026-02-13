"""Vérification de l'illumination."""

import cv2
import numpy as np
import numpy.typing as npt

from industrial_inspection.config.detection_config import DetectionConfig
from industrial_inspection.contracts.quality import QualityCheckResult
from industrial_inspection.quality_checks.base_check import QualityCheck


class IlluminationCheck(QualityCheck):
    """Vérification de la correspondance d'illumination."""

    def __init__(self, config: DetectionConfig):
        self.config = config

    @property
    def name(self) -> str:
        return "Illumination"

    def check(
        self,
        image_ref: npt.NDArray[np.uint8],
        image_test: npt.NDArray[np.uint8],
        mask_ref: npt.NDArray[np.uint8],
        mask_test: npt.NDArray[np.uint8],
    ) -> QualityCheckResult:
        """Vérifie que l'illumination est similaire."""

        gray_ref = cv2.cvtColor(image_ref, cv2.COLOR_BGR2GRAY)
        gray_test = cv2.cvtColor(image_test, cv2.COLOR_BGR2GRAY)

        mean_ref = np.mean(gray_ref[mask_ref > 0])
        mean_test = np.mean(gray_test[mask_test > 0])

        diff = abs(mean_ref - mean_test) / mean_ref
        tolerance = self.config.illumination_tolerance

        if diff > tolerance:
            return QualityCheckResult(
                check_name=self.name,
                passed=False,
                message=f"Different brightness - Offset: {diff * 100:.1f}% (max: {tolerance * 100:.0f}%)",
                value=diff,
                threshold=tolerance,
            )

        return QualityCheckResult(check_name=self.name, passed=True, message="Illumination OK")
