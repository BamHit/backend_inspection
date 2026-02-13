"""Construction des modèles (Factory pattern)."""

import torch.nn as nn
from torchvision import models

from industrial_inspection.config.model_config import ModelConfig
from industrial_inspection.exceptions import ModelException
from industrial_inspection.ml.architectures.custom_cnns import (
    CompactCNN,
    LargeCNN,
    MediumCNN,
    ResidualCNN,
    SmallCNN,
)

try:
    from torchvision.models import EfficientNet_B0_Weights, ResNet18_Weights
except ImportError:
    EfficientNet_B0_Weights = None
    ResNet18_Weights = None


def build_model(config: ModelConfig) -> nn.Module:
    """Factory pour construire l'architecture du modèle.

    Args:
        config: Configuration du modèle.

    Returns:
        Modèle PyTorch.

    Raises:
        ModelException: Si l'architecture n'est pas supportée.
    """
    arch = config.architecture
    num_classes = config.num_classes

    match arch:
        case "small_cnn":
            return SmallCNN(num_classes=num_classes)

        case "medium_cnn":
            return MediumCNN(num_classes=num_classes)

        case "large_cnn":
            return LargeCNN(num_classes=num_classes)

        case "residual_cnn":
            return ResidualCNN(num_classes=num_classes)

        case "compact_cnn":
            return CompactCNN(num_classes=num_classes)

        case "efficientnet_b0":
            if EfficientNet_B0_Weights is None:
                raise ModelException("EfficientNet weights not available")
            model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

            last_layer = model.classifier[1]
            if isinstance(last_layer, nn.Linear):
                in_features = last_layer.in_features
                model.classifier[1] = nn.Linear(in_features, num_classes)

            return model

        case "resnet18":
            if ResNet18_Weights is None:
                raise ModelException("ResNet weights not available")
            model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model

        case _:
            raise ModelException(f"Architecture '{arch}' not supported")
