"""Pondération des masques."""

import cv2
import numpy as np
import numpy.typing as npt

from industrial_inspection.config.detection_config import DetectionConfig


def apply_sigmoid_weighting(
    mask: npt.NDArray[np.uint8], config: DetectionConfig
) -> npt.NDArray[np.float32]:
    """Applique une pondération sigmoïde basée sur la distance au bord.

    Args:
        mask: Masque binaire.
        config: Configuration contenant les paramètres de sigmoïde.

    Returns:
        Masque pondéré avec valeurs entre 0 et 1.
    """
    if np.sum(mask) == 0:
        return mask.astype(np.float32)

    # Distance transform
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_norm = dist / np.max(dist)

    # Sigmoïde
    k = config.sigmoid_steepness
    offset = config.sigmoid_center_offset
    weighted_mask = 1 / (1 + np.exp(-k * (dist_norm - offset)))

    return weighted_mask.astype(np.float32)
