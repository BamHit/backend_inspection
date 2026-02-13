"""Vérification de la netteté."""

import cv2
import numpy as np
import numpy.typing as npt

from industrial_inspection.config.detection_config import DetectionConfig
from industrial_inspection.contracts.quality import QualityCheckResult
from industrial_inspection.quality_checks.base_check import QualityCheck


class SharpnessCheck(QualityCheck):
    """Vérification de la correspondance de netteté."""

    def __init__(self, config: DetectionConfig):
        self.config = config

    @property
    def name(self) -> str:
        return "Sharpness"

    def check(
        self,
        image_ref: npt.NDArray[np.uint8],
        image_test: npt.NDArray[np.uint8],
        mask_ref: npt.NDArray[np.uint8],
        mask_test: npt.NDArray[np.uint8],
    ) -> QualityCheckResult:
        """Vérifie que la netteté est similaire."""

        gray_ref = cv2.cvtColor(image_ref, cv2.COLOR_BGR2GRAY)
        gray_test = cv2.cvtColor(image_test, cv2.COLOR_BGR2GRAY)

        lap_ref = cv2.Laplacian(gray_ref, cv2.CV_64F).var()
        lap_test = cv2.Laplacian(gray_test, cv2.CV_64F).var()

        diff = abs(lap_ref - lap_test) / lap_ref
        tolerance = self.config.sharpness_tolerance

        if diff > tolerance:
            return QualityCheckResult(
                check_name=self.name,
                passed=False,
                message=f"Different sharpness - Offset: {diff * 100:.1f}% (max: {tolerance * 100:.0f}%)",
                value=diff,
                threshold=tolerance,
            )

        return QualityCheckResult(check_name=self.name, passed=True, message="Sharpness OK")
