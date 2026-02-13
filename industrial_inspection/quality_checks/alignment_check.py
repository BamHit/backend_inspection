"""Vérification de l'alignement."""

import numpy as np
import numpy.typing as npt

from industrial_inspection.config.detection_config import DetectionConfig
from industrial_inspection.contracts.quality import QualityCheckResult
from industrial_inspection.image_processing.geometry import compute_pca_and_centroid
from industrial_inspection.quality_checks.base_check import QualityCheck


class AlignmentCheck(QualityCheck):
    """Vérification de l'alignement de la pièce."""

    def __init__(self, config: DetectionConfig):
        self.config = config

    @property
    def name(self) -> str:
        return "Alignment"

    def check(
        self,
        image_ref: npt.NDArray[np.uint8],
        image_test: npt.NDArray[np.uint8],
        mask_ref: npt.NDArray[np.uint8],
        mask_test: npt.NDArray[np.uint8],
    ) -> QualityCheckResult:
        """Vérifie l'alignement entre référence et test."""

        centroid_ref, axes_ref, _ = compute_pca_and_centroid(mask_ref)
        centroid_test, axes_test, _ = compute_pca_and_centroid(mask_test)

        if centroid_ref is None or centroid_test is None or axes_ref is None or axes_test is None:
            return QualityCheckResult(
                check_name=self.name, passed=False, message="Segmentation failed"
            )

        # Conversion explicite en array pour satisfaire le typage strict
        c_ref: npt.NDArray[np.float64] = np.array(centroid_ref, dtype=np.float64)
        c_test: npt.NDArray[np.float64] = np.array(centroid_test, dtype=np.float64)
        a_ref: npt.NDArray[np.float64] = axes_ref
        a_test: npt.NDArray[np.float64] = axes_test

        # Vérifier la distance
        dist = float(np.sqrt((c_ref[0] - c_test[0]) ** 2 + (c_ref[1] - c_test[1]) ** 2))

        dist_threshold = self.config.alignment_distance_threshold

        if dist > dist_threshold:
            return QualityCheckResult(
                check_name=self.name,
                passed=False,
                message=f"Misaligned part - Distance: {dist:.1f}px (max: {dist_threshold}px)",
                value=dist,
                threshold=dist_threshold,
            )

        # Vérifier l'angle
        angle_ref = np.degrees(np.arctan2(a_ref[0, 1], a_ref[0, 0]))
        angle_test = np.degrees(np.arctan2(a_test[0, 1], a_test[0, 0]))
        diff_angle = float(abs(angle_ref - angle_test))

        if diff_angle > 180:
            diff_angle = 360 - diff_angle

        angle_threshold = self.config.alignment_angle_threshold

        if diff_angle > angle_threshold:
            return QualityCheckResult(
                check_name=self.name,
                passed=False,
                message=f"Rotation detected - Angle: {diff_angle:.1f}° (max: {angle_threshold}°)",
                value=diff_angle,
                threshold=angle_threshold,
            )

        return QualityCheckResult(check_name=self.name, passed=True, message="Alignment OK")
