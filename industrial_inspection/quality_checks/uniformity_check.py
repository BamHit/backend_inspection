"""Vérification de l'uniformité de l'éclairage."""

import cv2
import numpy as np
import numpy.typing as npt

from industrial_inspection.config.detection_config import DetectionConfig
from industrial_inspection.contracts.quality import QualityCheckResult
from industrial_inspection.quality_checks.base_check import QualityCheck


class UniformityCheck(QualityCheck):
    """Vérification de l'uniformité de l'éclairage."""

    def __init__(self, config: DetectionConfig):
        self.config = config

    @property
    def name(self) -> str:
        return "Lighting Uniformity"

    def check(
        self,
        image_ref: npt.NDArray[np.uint8],
        image_test: npt.NDArray[np.uint8],
        mask_ref: npt.NDArray[np.uint8],
        mask_test: npt.NDArray[np.uint8],
    ) -> QualityCheckResult:
        """Vérifie que l'uniformité de l'éclairage est similaire."""

        gray_ref = cv2.cvtColor(image_ref, cv2.COLOR_BGR2GRAY)
        gray_test = cv2.cvtColor(image_test, cv2.COLOR_BGR2GRAY)

        std_ref = np.std(gray_ref[mask_ref > 0])
        std_test = np.std(gray_test[mask_test > 0])

        diff = abs(std_ref - std_test) / std_ref
        tolerance = self.config.lighting_uniformity_tolerance

        if diff > tolerance:
            return QualityCheckResult(
                check_name=self.name,
                passed=False,
                message=f"Inconsistent lighting uniformity - Offset: {diff * 100:.1f}% (max: {tolerance * 100:.0f}%)",
                value=diff,
                threshold=tolerance,
            )

        return QualityCheckResult(
            check_name=self.name, passed=True, message="Lighting uniformity OK"
        )
