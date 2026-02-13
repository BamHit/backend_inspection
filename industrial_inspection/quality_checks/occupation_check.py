"""Vérification de l'occupation de la frame."""

import numpy as np
import numpy.typing as npt

from industrial_inspection.config.detection_config import DetectionConfig
from industrial_inspection.contracts.quality import QualityCheckResult
from industrial_inspection.quality_checks.base_check import QualityCheck


class OccupationCheck(QualityCheck):
    """Vérification de l'occupation de la frame (zoom/échelle)."""

    def __init__(self, config: DetectionConfig):
        self.config = config

    @property
    def name(self) -> str:
        return "Frame Occupation"

    def check(
        self,
        image_ref: npt.NDArray[np.uint8],
        image_test: npt.NDArray[np.uint8],
        mask_ref: npt.NDArray[np.uint8],
        mask_test: npt.NDArray[np.uint8],
    ) -> QualityCheckResult:
        """Vérifie que l'occupation de la frame est similaire."""

        h_ref, w_ref = mask_ref.shape
        h_test, w_test = mask_test.shape

        area_ref = np.sum(mask_ref > 0) / (h_ref * w_ref)
        area_test = np.sum(mask_test > 0) / (h_test * w_test)

        diff = abs(area_ref - area_test) / area_ref
        tolerance = self.config.frame_occupation_tolerance

        if diff > tolerance:
            return QualityCheckResult(
                check_name=self.name,
                passed=False,
                message=f"Different zoom - Offset: {diff * 100:.1f}% (max: {tolerance * 100:.0f}%)",
                value=diff,
                threshold=tolerance,
            )

        return QualityCheckResult(check_name=self.name, passed=True, message="Scale OK")
