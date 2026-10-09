"""Tests for the predict() inference function."""

import numpy as np
from PIL import Image
from wheat_classifier.model import CLASS_NAMES, build_model
from wheat_classifier.predict import predict


def _make_image(tmp_path, size=(64, 64)):
    arr = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    path = tmp_path / "leaf.jpg"
    Image.fromarray(arr).save(path)
    return path


def test_predict_keys(tmp_path):
    model = build_model(pretrained=False)
    result = predict(_make_image(tmp_path), model=model)
    assert set(result.keys()) == {"class", "confidence", "probabilities"}


def test_predict_class_valid(tmp_path):
    model = build_model(pretrained=False)
    result = predict(_make_image(tmp_path), model=model)
    assert result["class"] in CLASS_NAMES


def test_predict_confidence_range(tmp_path):
    model = build_model(pretrained=False)
    result = predict(_make_image(tmp_path), model=model)
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_probabilities_sum(tmp_path):
    model = build_model(pretrained=False)
    result = predict(_make_image(tmp_path), model=model)
    total = sum(result["probabilities"].values())
    assert abs(total - 1.0) < 1e-5


def test_predict_probabilities_keys(tmp_path):
    model = build_model(pretrained=False)
    result = predict(_make_image(tmp_path), model=model)
    assert set(result["probabilities"].keys()) == set(CLASS_NAMES)


def test_predict_no_model_arg(tmp_path):
    """predict() must work with model=None (builds random demo model)."""
    result = predict(_make_image(tmp_path), model=None)
    assert result["class"] in CLASS_NAMES
