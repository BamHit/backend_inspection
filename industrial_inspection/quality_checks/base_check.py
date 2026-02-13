"""Interface de base pour les vérifications qualité."""

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from industrial_inspection.contracts.quality import QualityCheckResult


class QualityCheck(ABC):
    """Interface abstraite pour une vérification qualité."""

    @abstractmethod
    def check(
        self,
        image_ref: npt.NDArray[np.uint8],
        image_test: npt.NDArray[np.uint8],
        mask_ref: npt.NDArray[np.uint8],
        mask_test: npt.NDArray[np.uint8],
    ) -> QualityCheckResult:
        """Effectue la vérification.

        Args:
            image_ref: Image de référence (BGR).
            image_test: Image de test (BGR).
            mask_ref: Masque de référence.
            mask_test: Masque de test.

        Returns:
            Résultat de la vérification.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Nom de la vérification."""
        pass
