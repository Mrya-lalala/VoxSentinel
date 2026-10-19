"""AASIST training-plumbing checks on disposable fixtures.

These tests exercise the real ``src.detectors.training.fit`` loop with the real
embedding-adapted AASIST head on tiny synthetic batches.  They verify plumbing
only: updates, freezing, evaluation isolation, checkpoint fidelity and metric
bookkeeping.  Nothing here is performance evidence for a trained release, and
no test loads real audio, real caches or the encoder.
"""

from __future__ import annotations

import copy

import pytest
import torch

from src.config import ModelConfig, OptimizerConfig
from src.data import EmbeddingExample, collate_examples
from src.detectors import create_detector
from src.detectors.checkpoints import restore_checkpoint
from src.detectors.standardization import fit_detector_standardizer
from src.detectors.training import build_loss, build_optimizer, evaluate, fit

FRAMES = 40


def _config(**overrides) -> dict:
    params = {
        "input_dim": 16,
        "pooled_bins": 32,
        "channels": 4,
        "latent_nodes": 4,
        "gat_dims": [4, 2],
        "pool_ratios": [0.5, 0.7, 0.5],
        "temperatures": [2.0, 2.0, 100.0],
        "dropout_attention": 0.2,
        "dropout_pool": 0.3,
        "dropout_fusion": 0.2,
        "dropout_readout": 0.5,
        "feature_standardization": True,
        "num_classes": 2,
    }
    params.update(overrides)
    return params


def _examples(count: int, *, offset: float, seed: int, frames: int = FRAMES, dim: int = 16) -> list[EmbeddingExample]:
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(count, frames, dim, generator=generator) + offset
    return [EmbeddingExample(features[index], index % 2) for index in range(count)]


def _fitted_detector() -> object:
    detector = create_detector(ModelConfig("aasist", _config()))
    fit_detector_standardizer(detector, _examples(12, offset=0.5, seed=1))
    return detector


def _batches(examples, size: int = 4):
    return lambda epoch: (collate_examples(examples[i : i + size]) for i in range(0, len(examples), size))


def _fit(detector, *, epochs: int = 2, checkpoint_path=None, metadata=None):
    train = _examples(12, offset=1.0, seed=2)
    dev = _examples(8, offset=1.2, seed=3)
    optimizer = build_optimizer(detector, OptimizerConfig())
    payload_metadata = dict(metadata or {})
    payload_metadata.setdefault("feature_standardization", dict(detector.model.standardizer.metadata))
    return fit(
        detector,
        _batches(train),
        _batches(dev),
        optimizer,
        model_name="aasist",
        model_config=_config(),
        epochs=epochs,
        device="cpu",
        loss_fn=build_loss("cross_entropy"),
        checkpoint_path=checkpoint_path,
        monitor="eer",
        mode="min",
        threshold=0.5,
        metadata=payload_metadata,
    )


def _parameter_snapshot(detector) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in detector.named_parameters()}


def _buffer_snapshot(module) -> dict[str, torch.Tensor]:
    return {name: buffer.detach().clone() for name, buffer in module.named_buffers()}


def test_fit_trains_head_and_keeps_standardizer_fixed():
    detector = _fitted_detector()
    standardizer = detector.model.standardizer
    before = _parameter_snapshot(detector)
    buffers_before = _buffer_snapshot(standardizer)
    metadata_before = copy.deepcopy(standardizer.metadata)

    result = _fit(detector)

    assert result.best_epoch is not None
    assert all(torch.isfinite(torch.tensor(epoch.train_loss)) for epoch in result.epochs)
    assert all(torch.isfinite(torch.tensor(epoch.val_loss)) for epoch in result.epochs)

    after = _parameter_snapshot(detector)
    changed = {name for name, value in after.items() if not torch.equal(before[name], value)}
    assert {"model.projection.weight", "model.out_layer.weight", "model.master1"} <= changed

    for name, buffer in standardizer.named_buffers():
        assert torch.equal(buffer, buffers_before[name]), f"standardizer buffer {name} changed"
    assert standardizer.metadata == metadata_before
    assert not detector.training

    with pytest.raises(ValueError, match="Refusing to refit"):
        fit_detector_standardizer(detector, _examples(4, offset=0.0, seed=9))


def test_fit_requires_fitted_standardizer_and_writes_no_checkpoint(tmp_path):
    detector = create_detector(ModelConfig("aasist", _config()))
    target = tmp_path / "should_not_exist.pt"
    with pytest.raises(ValueError, match="training statistics"):
        _fit(detector, epochs=1, checkpoint_path=target)
    assert not target.exists()


def test_fit_updates_batchnorm_running_stats_but_evaluate_is_read_only():
    detector = _fitted_detector()
    bn_before = detector.model.gat_s.bn.running_mean.detach().clone()
    _fit(detector, epochs=1)
    assert not torch.equal(bn_before, detector.model.gat_s.bn.running_mean)

    parameters = _parameter_snapshot(detector)
    buffers = _buffer_snapshot(detector)
    batches = [collate_examples(_examples(6, offset=1.5, seed=5)[i : i + 3]) for i in range(0, 6, 3)]

    state = torch.random.get_rng_state().clone()
    evaluation = evaluate(detector, batches, "cpu", threshold=0.5)
    assert torch.equal(state, torch.random.get_rng_state())
    assert not detector.training
    assert evaluation.scores.shape == (6,)
    assert evaluation.loss is not None and torch.isfinite(torch.tensor(evaluation.loss))
    assert evaluation.batches == 2

    for name, parameter in _parameter_snapshot(detector).items():
        assert torch.equal(parameters[name], parameter), f"{name} changed during evaluation"
    for name, buffer in detector.named_buffers():
        assert torch.equal(buffers[name], buffer), f"buffer {name} changed during evaluation"


def test_fit_selects_earliest_minimum_development_eer():
    detector = _fitted_detector()
    result = _fit(detector, epochs=3)
    history = [epoch.metrics.eer for epoch in result.epochs]
    best = min(value for value in history if value is not None)
    assert result.best_metric == best
    assert result.best_epoch == next(index + 1 for index, value in enumerate(history) if value == best)


def test_fit_checkpoint_preserves_predictions_config_metrics_and_metadata(tmp_path):
    detector = _fitted_detector()
    metadata = {"encoder": {"checkpoint_sha256": "a" * 64, "selected_layer": None, "embedding_dim": 16,
                            "preprocessing_version": "voxsentinel-prep-2", "dtype": "float32"}}
    path = tmp_path / "aasist_best.pt"
    result = _fit(detector, epochs=3, checkpoint_path=path, metadata=metadata)
    assert path.exists()

    dev = _examples(8, offset=1.2, seed=3)
    best_epoch = result.best_epoch
    assert best_epoch is not None

    first = create_detector(ModelConfig("aasist", _config()))
    payload = restore_checkpoint(path, first, model_name="aasist")
    assert payload["model_name"] == "aasist"
    assert payload["model_config"]["feature_standardization"] is True
    assert payload["epoch"] == best_epoch
    assert payload["metadata"]["encoder"] == metadata["encoder"]
    assert payload["metadata"]["feature_standardization"]["schema"] == "voxsentinel.frame_standardizer.v1"
    assert payload["model_state_dict"]["model.standardizer._extra_state"]["schema"] == "voxsentinel.frame_standardizer.v1"
    for key in ("eer", "accuracy", "genuine_false_alarm_rate", "spoof_miss_rate"):
        assert key in payload["metrics"], f"checkpoint metrics lost {key}"

    first.eval()
    evaluation = evaluate(first, list(_batches(dev)(0)), "cpu", threshold=0.5)
    recorded = result.epochs[best_epoch - 1].metrics
    assert evaluation.metrics.eer == recorded.eer
    assert evaluation.metrics.accuracy == recorded.accuracy
    assert payload["metrics"]["eer"] == recorded.eer
    assert payload["metrics"]["accuracy"] == recorded.accuracy

    second = create_detector(ModelConfig("aasist", _config()))
    restore_checkpoint(path, second, model_name="aasist")
    second.eval()
    with torch.no_grad():
        first_logits = first(collate_examples(dev[:4])).logits
        second_logits = second(collate_examples(dev[:4])).logits
    assert torch.equal(first_logits, second_logits)
