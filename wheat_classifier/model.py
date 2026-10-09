"""ResNet50 model builder for wheat disease classification."""

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

NUM_CLASSES: int = 5
CLASS_NAMES: list[str] = [
    "BlackPoint",
    "FusariumFootRot",
    "HealthyLeaf",
    "LeafBlight",
    "WheatBlast",
]


def build_model(pretrained: bool = False) -> nn.Module:
    """Build a ResNet50 with a 5-class classification head.

    Args:
        pretrained: Load ImageNet weights when True (requires internet).
                    Pass False for offline testing and CI.

    Returns:
        Model in training mode with a 5-class fully-connected output layer.
    """
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    return model


def load_model(model_path: str, device: str = "cpu") -> nn.Module:
    """Load a fine-tuned model from a .pth state-dict file.

    Args:
        model_path: Path to the .pth weights file.
        device: Torch device string, e.g. ``"cpu"`` or ``"cuda"``.

    Returns:
        Model set to eval mode with weights loaded from disk.
    """
    model = build_model(pretrained=False)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
