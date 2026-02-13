"""Contracts pour les vérifications qualité."""


from pydantic import BaseModel


class QualityCheckResult(BaseModel):
    """Résultat d'une vérification qualité individuelle."""

    check_name: str
    """Nom du check (ex: 'Alignment', 'Illumination')"""

    passed: bool
    """True si le check est passé, False sinon"""

    message: str
    """Message décrivant le résultat (ex: 'Alignment OK' ou 'Misaligned part')"""

    value: float | None = None
    """Valeur mesurée (optionnel, ex: distance en pixels, différence en %)"""

    threshold: float | None = None
    """Seuil utilisé pour la validation (optionnel)"""


class QualityReport(BaseModel):
    """Rapport complet de qualité regroupant tous les checks."""

    all_passed: bool
    """True si tous les checks sont passés, False sinon"""

    checks: list[QualityCheckResult]
    """Liste de tous les résultats de checks"""

    summary: str
    """Résumé textuel du rapport (ex: 'All quality checks passed' ou liste des erreurs)"""
