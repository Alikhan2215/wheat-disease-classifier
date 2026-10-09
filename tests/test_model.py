"""Tests for model construction and output shape."""

import torch
from wheat_classifier.model import CLASS_NAMES, NUM_CLASSES, build_model


def test_num_classes():
    assert NUM_CLASSES == 5


def test_class_names_count():
    assert len(CLASS_NAMES) == 5


def test_class_names_values():
    expected = {
        "BlackPoint", "FusariumFootRot", "HealthyLeaf",
        "LeafBlight", "WheatBlast",
    }
    assert set(CLASS_NAMES) == expected


def test_build_model_output_shape():
    model = build_model(pretrained=False)
    model.eval()
    dummy = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy)
    assert output.shape == (1, NUM_CLASSES)


def test_build_model_no_pretrained():
    """Model must build without any network access when pretrained=False."""
    model = build_model(pretrained=False)
    assert model is not None
