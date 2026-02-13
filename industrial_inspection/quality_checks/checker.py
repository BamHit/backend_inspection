"""Orchestrateur de vérifications qualité."""


import numpy as np
import numpy.typing as npt

from industrial_inspection.config.detection_config import DetectionConfig
from industrial_inspection.contracts.quality import QualityCheckResult, QualityReport
from industrial_inspection.quality_checks.alignment_check import AlignmentCheck
from industrial_inspection.quality_checks.base_check import QualityCheck
from industrial_inspection.quality_checks.illumination_check import IlluminationCheck
from industrial_inspection.quality_checks.occupation_check import OccupationCheck
from industrial_inspection.quality_checks.sharpness_check import SharpnessCheck
from industrial_inspection.quality_checks.uniformity_check import UniformityCheck


class QualityChecker:
    """Orchestre toutes les vérifications qualité."""

    def __init__(self, config: DetectionConfig):
        self.checks: list[QualityCheck] = [
            AlignmentCheck(config),
            IlluminationCheck(config),
            OccupationCheck(config),
            SharpnessCheck(config),
            UniformityCheck(config),
        ]

    def run_all_checks(
        self,
        image_ref: npt.NDArray[np.uint8],
        image_test: npt.NDArray[np.uint8],
        mask_ref: npt.NDArray[np.uint8],
        mask_test: npt.NDArray[np.uint8],
    ) -> QualityReport:
        """Exécute toutes les vérifications."""

        results: list[QualityCheckResult] = []

        for check in self.checks:
            result = check.check(image_ref, image_test, mask_ref, mask_test)
            results.append(result)

        all_passed = all(r.passed for r in results)

        failed_checks = [r for r in results if not r.passed]
        if failed_checks:
            summary = " & ".join([r.message for r in failed_checks])
        else:
            summary = "All quality checks passed"

        return QualityReport(all_passed=all_passed, checks=results, summary=summary)
