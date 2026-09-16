import pytest

from src.config import ModelConfig
from src.detectors import GruDetector, available_detectors, create_detector


def test_registry_lists_reserved_and_implemented_detectors():
    assert available_detectors() == ("aasist", "gru")


def test_registry_constructs_the_gru_detector():
    detector = create_detector(ModelConfig(name="gru"))
    assert isinstance(detector, GruDetector)
    assert detector.model.input_dim == 1024
    assert detector.model.classifier.out_features == 2


def test_registry_reports_unimplemented_and_unknown_names():
    with pytest.raises(NotImplementedError, match="aasist"):
        create_detector(ModelConfig(name="aasist"))
    with pytest.raises(KeyError, match="Unknown detector"):
        create_detector(ModelConfig(name="resnet"))
