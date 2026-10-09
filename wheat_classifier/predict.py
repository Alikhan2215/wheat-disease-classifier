"""Inference pipeline for wheat disease classification."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import CLASS_NAMES, build_model
from .preprocess import load_and_preprocess


def predict(
    image_path: str | Path,
    model: nn.Module | None = None,
) -> dict[str, Any]:
    """Classify a wheat leaf image.

    Args:
        image_path: Path to the input image.
        model: A loaded torch model. If ``None``, a fresh random-weight
               ResNet50 is built (demo / offline mode).

    Returns:
        A dict with three keys:

        * ``"class"``         – predicted class name (str)
        * ``"confidence"``    – top-class probability (float in [0, 1])
        * ``"probabilities"`` – mapping of every class name to its probability
    """
    if model is None:
        model = build_model(pretrained=False)

    model.eval()
    tensor = load_and_preprocess(image_path)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    probs_list = probs.tolist()
    top_idx = int(probs.argmax())

    return {
        "class": CLASS_NAMES[top_idx],
        "confidence": probs_list[top_idx],
        "probabilities": dict(zip(CLASS_NAMES, probs_list)),
    }
