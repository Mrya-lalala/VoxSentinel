from voxsentinel.config import ModelConfig
from voxsentinel.models import GruSpoofDetector, available_models, create_backbone


def test_reserved_models_are_visible_before_implementation():
    assert {"gru", "aasist"}.issubset(available_models())


def test_gru_registry_constructs_the_default_detector():
    model = create_backbone(ModelConfig(name="gru"))
    assert isinstance(model, GruSpoofDetector)
    assert model.input_dim == 1024
    assert model.hidden_size == 256
    assert model.classifier.out_features == 2
