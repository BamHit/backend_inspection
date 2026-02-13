"""Tests pour les vérifications qualité."""

import numpy as np

from industrial_inspection.quality_checks.alignment_check import AlignmentCheck


def test_alignment_check_identical_images(detection_config, dummy_image, dummy_mask):
    """Vérifie que des images identiques passent le test d'alignement."""
    checker = AlignmentCheck(detection_config)

    result = checker.check(dummy_image, dummy_image, dummy_mask, dummy_mask)

    assert result.passed is True
    assert result.check_name == "Alignment"


def test_alignment_check_shifted_mask(detection_config, dummy_image):
    """Vérifie que des masques décalés échouent."""
    checker = AlignmentCheck(detection_config)

    mask1 = np.zeros((100, 100), dtype=np.uint8)
    mask1[25:75, 25:75] = 255

    mask2 = np.zeros((100, 100), dtype=np.uint8)
    mask2[50:100, 50:100] = 255  # Décalé de 25px

    result = checker.check(dummy_image, dummy_image, mask1, mask2)

    assert result.passed is False
    assert "Distance" in result.message
