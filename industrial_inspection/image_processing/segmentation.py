"""Fonctions de segmentation d'images."""

import cv2
import numpy as np
import numpy.typing as npt

from industrial_inspection.config.detection_config import DetectionConfig


def segment_piece(img: npt.NDArray[np.uint8], config: DetectionConfig) -> npt.NDArray[np.uint8]:
    """Segmente une pièce métallique par seuillage HSV.

    Args:
        img: Image BGR.
        config: Configuration contenant les seuils HSV.

    Returns:
        Masque binaire de la pièce.
    """
    lower = np.array(config.hsv_lower_bound, dtype=np.uint8)
    upper = np.array(config.hsv_upper_bound, dtype=np.uint8)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_not(cv2.inRange(hsv, lower, upper))

    kernel = np.ones((1, 1), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask.astype(np.uint8)


def keep_largest_contour(mask: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Garde uniquement le contour le plus grand.

    Args:
        mask: Masque binaire.

    Returns:
        Masque avec uniquement le plus grand contour.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.zeros_like(mask)

    largest = max(contours, key=cv2.contourArea)
    mask_largest = np.zeros_like(mask)
    cv2.drawContours(mask_largest, [largest], -1, (255,), thickness=cv2.FILLED)

    return mask_largest
