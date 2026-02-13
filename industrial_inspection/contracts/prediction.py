"""Contracts pour les prédictions."""


from pydantic import BaseModel


class PredictionResult(BaseModel):
    """Résultat d'une prédiction."""

    predicted_class: str
    predicted_index: int
    confidence: float
    probabilities: dict[str, float]


class ROI(BaseModel):
    """Région d'intérêt."""

    x: int
    y: int
    size: int
    expected_class: str | None = None


class ROIBatch(BaseModel):
    """Batch de ROIs."""

    rois: list[ROI]
    image_base64: str


class PredictionBatch(BaseModel):
    """Batch de prédictions."""

    predictions: list[PredictionResult]
    processing_time_ms: float
