"""Tests pour les fonctions de segmentation."""

import numpy as np

from industrial_inspection.image_processing.segmentation import (
    keep_largest_contour,
    segment_piece,
)


def test_segment_piece_returns_mask(detection_config):
    """Vérifie que segment_piece retourne un masque binaire."""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    mask = segment_piece(img, detection_config)

    assert mask.shape == (100, 100)
    assert mask.dtype == np.uint8
    assert np.all((mask == 0) | (mask == 255))


def test_keep_largest_contour(dummy_mask):
    """Vérifie que keep_largest_contour garde le plus grand contour."""
    # Ajouter un petit contour
    mask = dummy_mask.copy()
    mask[10:15, 10:15] = 255

    result = keep_largest_contour(mask)

    # Le petit contour devrait être supprimé
    assert np.sum(result[10:15, 10:15]) == 0
    # Le grand contour devrait être conservé
    assert np.sum(result[25:75, 25:75]) > 0


def test_keep_largest_contour_empty_mask():
    """Vérifie le comportement avec un masque vide."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    result = keep_largest_contour(mask)

    assert np.sum(result) == 0
