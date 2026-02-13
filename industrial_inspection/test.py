# Test de validation
from contracts.prediction import PredictionResult

result = PredictionResult(
    predicted_class="Thread",
    predicted_index=0,
    confidence=0.95,
    probabilities={"Thread": 0.95, "Hole": 0.05},
)
print(result.model_dump_json(indent=2))
