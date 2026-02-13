"""Prédiction avec preprocessing."""

import logging

import numpy as np
import torch

from industrial_inspection.contracts.prediction import PredictionResult
from industrial_inspection.image_processing.preprocessing import (
    preprocess_for_inference,
)
from industrial_inspection.ml.model_loader import ModelSingleton

logger = logging.getLogger(__name__)


def predict(image: np.ndarray) -> PredictionResult:
    """Effectue une prédiction sur une image.

    Args:
        image: Image numpy (RGB ou BGR).

    Returns:
        Résultat de la prédiction.
    """
    model = ModelSingleton.get_model()
    config = ModelSingleton.get_config()
    device = ModelSingleton.get_device()

    # Preprocessing
    tensor = preprocess_for_inference(
        image,
        target_size=(config.input_size, config.input_size),
    )
    tensor = tensor.unsqueeze(0).to(device)  # Batch dimension

    # Inférence
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    # Résultats
    predicted_idx = int(probabilities.argmax().item())
    predicted_class = config.class_names[predicted_idx]
    confidence = probabilities[predicted_idx].item()

    probs_dict = {
        class_name: prob.item()
        for class_name, prob in zip(config.class_names, probabilities, strict=False)
    }

    logger.info(f"Prediction: {predicted_class} ({confidence:.2%})")

    return PredictionResult(
        predicted_class=predicted_class,
        predicted_index=predicted_idx,
        confidence=confidence,
        probabilities=probs_dict,
    )
