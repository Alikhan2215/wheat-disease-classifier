"""Tests for the image preprocessing pipeline."""

import numpy as np
import torch
from PIL import Image
from wheat_classifier.preprocess import load_and_preprocess


def _make_image(tmp_path, size=(320, 240), color=(100, 150, 200)):
    arr = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    path = tmp_path / "leaf.jpg"
    Image.fromarray(arr).save(path)
    return path


def test_output_shape(tmp_path):
    tensor = load_and_preprocess(_make_image(tmp_path))
    assert tensor.shape == (1, 3, 224, 224)


def test_output_dtype(tmp_path):
    tensor = load_and_preprocess(_make_image(tmp_path))
    assert tensor.dtype == torch.float32


def test_batch_dimension(tmp_path):
    tensor = load_and_preprocess(_make_image(tmp_path))
    assert tensor.ndim == 4


def test_normalization_applied(tmp_path):
    """White pixels exceed [0, 1] after ImageNet normalisation."""
    path = _make_image(tmp_path, color=(255, 255, 255))
    tensor = load_and_preprocess(path)
    # (1.0 - 0.485) / 0.229 ≈ 2.25 — confirms normalisation was applied
    assert tensor.max().item() > 1.0
