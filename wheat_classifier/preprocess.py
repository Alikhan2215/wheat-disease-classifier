"""Image preprocessing pipeline for ResNet50 inference."""

from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision import transforms

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_and_preprocess(image_path: str | Path) -> torch.Tensor:
    """Load an image and prepare it for ResNet50 inference.

    Applies EXIF-based orientation correction, resizes to 224 x 224,
    converts to a float32 tensor, and normalises with ImageNet statistics.

    Args:
        image_path: Path to any PIL-readable image file.

    Returns:
        A ``(1, 3, 224, 224)`` float32 tensor ready for the model.
    """
    img = Image.open(image_path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    tensor = _TRANSFORM(img)
    return tensor.unsqueeze(0)
