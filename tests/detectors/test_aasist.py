import pytest
import torch
from torch.nn import functional as F

from src.config import ModelConfig, load_config
from src.data import EmbeddingExample, collate_examples
from src.detectors import AasistDetector, AasistSpoofDetector, available_detectors, create_detector
from src.detectors.aasist_blocks import GraphPool, HtrgGraphAttentionLayer
from src.detectors.checkpoints import load_checkpoint, restore_checkpoint, save_checkpoint
from src.detectors.standardization import fit_detector_standardizer

INPUT_DIM = 32


def _params(**overrides) -> dict:
    params = {
        "input_dim": INPUT_DIM,
        "pooled_bins": 32,
        "channels": 32,
        "latent_nodes": 16,
        "gat_dims": (64, 32),
        "pool_ratios": (0.5, 0.7, 0.5),
        "temperatures": (2.0, 2.0, 100.0),
        "feature_standardization": True,
    }
    params.update(overrides)
    return params


def _examples(count: int = 8, frames: int = 64, dim: int = INPUT_DIM, seed: int = 0) -> list[EmbeddingExample]:
    generator = torch.Generator().manual_seed(seed)
    return [EmbeddingExample(torch.randn(frames, dim, generator=generator), index % 2) for index in range(count)]


def _fitted_detector(**overrides) -> AasistDetector:
    detector = AasistDetector(AasistSpoofDetector(**_params(**overrides)))
    fit_detector_standardizer(detector, _examples(dim=detector.model.input_dim))
    return detector


def _batch(lengths: list[int], dim: int = INPUT_DIM, frames: int | None = None, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    frames = frames or max(lengths)
    features = torch.randn(len(lengths), frames, dim, generator=generator)
    padding_mask = torch.arange(frames).unsqueeze(0) >= torch.tensor(lengths).unsqueeze(1)
    return {
        "features": features,
        "valid_lengths": torch.tensor(lengths),
        "padding_mask": padding_mask,
        "labels": torch.tensor([index % 2 for index in range(len(lengths))]),
    }


def test_aasist_registry_and_defaults():
    assert available_detectors() == ("aasist", "gru")
    detector = create_detector(ModelConfig(name="aasist"))
    assert isinstance(detector, AasistDetector)
    model = detector.model
    assert (model.input_dim, model.pooled_bins, model.channels, model.latent_nodes) == (1024, 32, 32, 16)
    assert model.gat_dims == (64, 32)
    assert model.pool_ratios == (0.5, 0.7, 0.5)
    assert model.out_layer.out_features == 2


def test_aasist_config_overlay_constructs_the_head():
    config = load_config(["configs/base.yaml", "configs/aasist.yaml"])
    assert config.model.name == "aasist"
    assert config.model.parameters["pooled_bins"] == 32
    detector = create_detector(config.model)
    assert detector.model.input_dim == 1024
    assert detector.model.temperatures == (2.0, 2.0, 100.0)


def test_aasist_forward_shapes_at_real_embedding_dimension():
    detector = _fitted_detector(input_dim=1024)
    detector.eval()
    single = _batch([49], dim=1024, seed=1)
    mixed = _batch([49, 199], dim=1024, seed=2)
    with torch.no_grad():
        single_logits = detector(single).logits
        mixed_logits = detector(mixed).logits
    assert single_logits.shape == (1, 2)
    assert mixed_logits.shape == (2, 2)
    assert torch.isfinite(single_logits).all() and torch.isfinite(mixed_logits).all()


def test_aasist_requires_fitted_standardizer():
    detector = AasistDetector(AasistSpoofDetector(**_params()))
    with pytest.raises(ValueError, match="training statistics"):
        detector(_batch([64]))
    fit_detector_standardizer(detector, _examples())
    with pytest.raises(ValueError, match="Refusing to refit"):
        fit_detector_standardizer(detector, _examples())
    assert detector(_batch([64])).logits.shape == (1, 2)


def test_aasist_padding_values_do_not_change_logits():
    detector = _fitted_detector().eval()
    batch = _batch([64, 40], frames=64)
    clean = batch["features"].clone()
    noisy = batch["features"].clone()
    noisy[batch["padding_mask"]] = 77.0
    with torch.no_grad():
        clean_logits = detector({**batch, "features": clean}).logits
        positive = detector({**batch, "features": noisy}).logits
        noisy[batch["padding_mask"]] = -77.0
        negative = detector({**batch, "features": noisy}).logits
    assert torch.allclose(clean_logits, positive, atol=1e-6)
    assert torch.allclose(clean_logits, negative, atol=1e-6)


def test_aasist_single_item_matches_mixed_length_batch():
    detector = _fitted_detector().eval()
    mixed = _batch([64, 40], frames=64)
    item = {
        "features": mixed["features"][1:2, :40],
        "valid_lengths": torch.tensor([40]),
        "padding_mask": torch.zeros(1, 40, dtype=torch.bool),
    }
    with torch.no_grad():
        alone = detector(item).logits
        in_batch = detector(mixed).logits[1:2]
    assert torch.allclose(alone, in_batch, atol=1e-5)


def test_aasist_rejects_invalid_lengths_masks_and_short_inputs():
    detector = _fitted_detector()
    features = torch.randn(1, 64, INPUT_DIM)
    with pytest.raises(ValueError, match="at least 32 frames"):
        detector({"features": torch.randn(1, 16, INPUT_DIM)})
    with pytest.raises(ValueError, match="valid frames"):
        detector({"features": features, "valid_lengths": torch.tensor([20])})
    with pytest.raises(ValueError, match="right padding"):
        detector(
            {
                "features": features,
                "valid_lengths": torch.tensor([40]),
                "padding_mask": torch.tensor([[False] * 20 + [True] * 5 + [False] * 39]),
            }
        )
    with pytest.raises(ValueError, match="right padding"):
        detector({"features": features, "padding_mask": torch.tensor([[True] + [False] * 63])})
    with pytest.raises(ValueError, match="boolean"):
        detector({"features": features, "padding_mask": torch.zeros(1, 64, dtype=torch.long)})
    with pytest.raises(ValueError, match="integer"):
        detector({"features": features, "valid_lengths": torch.tensor([40.0])})
    with pytest.raises(ValueError, match="within"):
        detector({"features": features, "valid_lengths": torch.tensor([999])})
    with pytest.raises(ValueError, match="at least one valid frame"):
        detector({"features": features, "padding_mask": torch.ones(1, 64, dtype=torch.bool)})
    with pytest.raises(ValueError, match="non-finite"):
        detector({"features": torch.full((1, 64, INPUT_DIM), float("inf"))})
    with pytest.raises(ValueError, match="Expected embedding dim"):
        detector({"features": torch.randn(1, 64, INPUT_DIM + 1)})
    with pytest.raises(ValueError, match="batch, time, dim"):
        detector({"features": torch.randn(64, INPUT_DIM)})
    with pytest.raises(ValueError, match="float32"):
        detector({"features": torch.randn(1, 64, INPUT_DIM, dtype=torch.float64)})


def test_aasist_backward_updates_head_groups_and_keeps_inputs_detached():
    detector = _fitted_detector()
    batch = _batch([64, 48])
    output = detector(batch)
    loss = detector.loss(output, batch)
    loss.backward()

    model = detector.model
    for name, parameter in {
        "projection": model.projection.weight,
        "gat_s": model.gat_s.att_proj.weight,
        "gat_t": model.gat_t.proj_with_att.weight,
        "hetero_st11": model.hetero_st11.att_proj.weight,
        "hetero_st22": model.hetero_st22.proj_without_attM.weight,
        "pool_s": model.pool_s.proj.weight,
        "master1": model.master1,
        "out_layer": model.out_layer.weight,
    }.items():
        assert parameter.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(parameter.grad).all(), f"{name} has non-finite gradients"

    assert batch["features"].requires_grad is False
    assert batch["features"].grad is None
    assert model.standardizer.mean.grad is None and model.standardizer.scale.grad is None
    assert model.standardizer.mean.requires_grad is False


def test_aasist_eval_determinism_and_train_dropout_with_batch_size_one():
    detector = _fitted_detector().eval()
    batch = _batch([64])
    with torch.no_grad():
        first = detector(batch).logits
        second = detector(batch).logits
    assert torch.equal(first, second)

    detector.train()
    first = detector(batch).logits
    second = detector(batch).logits
    assert not torch.equal(first, second)
    loss = F.cross_entropy(first, batch["labels"])
    loss.backward()
    assert detector.model.gat_s.bn.running_mean is not None
    assert torch.isfinite(detector.model.projection.weight.grad).all()


def test_aasist_checkpoint_round_trip_parity(tmp_path):
    detector = _fitted_detector().eval()
    batch = _batch([64, 48])
    with torch.no_grad():
        live = detector(batch).logits

    path = tmp_path / "aasist_best.pt"
    save_checkpoint(path, detector, "aasist", _params(), epoch=2, global_step=4, metrics={"eer": 0.1})
    restored = AasistDetector(AasistSpoofDetector(**_params())).eval()
    payload = restore_checkpoint(path, restored, model_name="aasist")
    with torch.no_grad():
        reloaded = restored(batch).logits

    assert payload["epoch"] == 2
    assert torch.allclose(live, reloaded, atol=1e-6)
    assert float((live - reloaded).abs().max()) < 1e-5


def test_aasist_checkpoint_guards_wrong_model_and_transform_mismatch(tmp_path):
    detector = _fitted_detector().eval()
    path = tmp_path / "aasist.pt"
    save_checkpoint(path, detector, "aasist", _params())

    with pytest.raises(ValueError, match="not gru"):
        load_checkpoint(path, model_name="gru")

    plain = AasistDetector(AasistSpoofDetector(**_params(feature_standardization=False)))
    with pytest.raises(ValueError, match="disagree about required feature standardization"):
        restore_checkpoint(path, plain, model_name="aasist")


def test_aasist_checkpoint_rejects_missing_standardizer_state(tmp_path):
    detector = _fitted_detector().eval()
    payload = save_checkpoint(tmp_path / "aasist_full.pt", detector, "aasist", _params())
    stripped = dict(payload)
    stripped["model_state_dict"] = {
        key: value for key, value in payload["model_state_dict"].items() if not key.startswith("model.standardizer.")
    }
    path = tmp_path / "aasist_stripped.pt"
    torch.save(stripped, path)

    target = _fitted_detector().eval()
    with pytest.raises(RuntimeError, match="standardizer"):
        restore_checkpoint(path, target, strict=False, model_name="aasist")


def test_aasist_trainable_parameter_count_matches_gate1_plan():
    detector = create_detector(ModelConfig(name="aasist"))
    trainable = sum(parameter.numel() for parameter in detector.parameters() if parameter.requires_grad)
    assert trainable == 600_392


def test_aasist_graph_blocks_match_upstream_shapes():
    pool = GraphPool(0.7, 8, 0.0)
    nodes = torch.randn(2, 32, 8)
    assert pool(nodes).shape == (2, 22, 8)
    tiny = GraphPool(0.5, 8, 0.0)
    assert tiny(torch.randn(1, 3, 8)).shape == (1, 1, 8)

    layer = HtrgGraphAttentionLayer(8, 4, temperature=100.0, dropout=0.0)
    x1 = torch.randn(2, 5, 8)
    x2 = torch.randn(2, 3, 8)
    master = torch.randn(2, 1, 8)
    out1, out2, out_master = layer(x1, x2, master=master)
    assert out1.shape == (2, 5, 4)
    assert out2.shape == (2, 3, 4)
    assert out_master.shape == (2, 1, 4)


def test_aasist_detector_loss_scores_and_label_validation():
    detector = _fitted_detector().eval()
    batch = _batch([64, 48])
    with torch.no_grad():
        output = detector(batch)
    assert output.scores.shape == (2,)
    assert torch.allclose(output.scores, output.logits.softmax(dim=-1)[..., 1], atol=1e-6)
    assert torch.isfinite(detector.loss(output, batch)).all()
    with pytest.raises(ValueError, match="binary"):
        detector.loss(output, {**batch, "labels": torch.tensor([0.5, 1.0])})
