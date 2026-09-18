"""CPU integration smoke: real frozen IndicWav2Vec -> existing GRU -> one update.

Run from the repository root: python -m scripts.verify_encoder_gru
Uses two hash-checked genuine LibriSpeech fixtures, not a detector evaluation set.
No downloads, routing changes, or trained detector checkpoint are produced.
"""
import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import soundfile as sf
import torch
import yaml

from src.backbones.indic_wav2vec import IndicWav2VecConfig, IndicWav2VecExtractor
from src.config import load_config
from src.data.batch import EmbeddingExample, collate_examples
from src.data.labels import GENUINE
from src.detectors import create_detector
from src.detectors.training import build_optimizer, train_epoch


FIXTURES = (
    ("1272-128104-0000.flac", "4e25e22555cd16e90edb0a3b49fdcf1fe652b2a1250ab643634db33895c75b41", 48000),
    ("1272-128104-0001.flac", "46a9d58622b4675b29564da2d9ba73e702241c5fa969f12c387cad4aa984276a", 64000),
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def model_digest(model):
    """Hash every encoder parameter and persistent buffer without cloning the model."""
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone-config", default="configs/backbones.yaml")
    parser.add_argument("--detector-config", nargs="+", default=["configs/base.yaml", "configs/gru.yaml"])
    parser.add_argument("--speech-dir", default="artifacts/speech")
    parser.add_argument("--output", default="artifacts/encoder-gru-smoke.json")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    encoder_config = IndicWav2VecConfig(**yaml.safe_load(Path(args.backbone_config).read_text())["indic_wav2vec"])
    detector_config = load_config(args.detector_config)
    if encoder_config.device != "cpu" or detector_config.training.device != "cpu":
        parser.error("this smoke test measures CPU execution only; use CPU configurations")
    if detector_config.model.name.lower() != "gru":
        parser.error("this smoke test requires the existing GRU detector")
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)

    rows, fixtures = [], []
    for filename, expected_hash, count in FIXTURES:
        path = Path(args.speech_dir) / filename
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        require(digest == expected_hash, f"fixture SHA-256 mismatch: {path}")
        audio, rate = sf.read(path, dtype="float32")
        require(rate == 16000 and audio.ndim == 1 and len(audio) >= count,
                f"expected mono 16 kHz speech with at least {count} samples: {path}")
        rows.append(torch.from_numpy(audio[:count].copy()))
        fixtures.append({"path": str(path), "sha256": digest, "source_samples": len(audio),
                         "used_samples": count, "sample_rate": rate, "label": GENUINE})

    print("Loading and hashing the real encoder...", flush=True)
    start = time.perf_counter()
    encoder = IndicWav2VecExtractor(encoder_config).load()
    load_seconds = time.perf_counter() - start
    model = encoder._model  # Verification introspection; not the downstream API.
    encoder_before = model_digest(model)
    detector = create_detector(detector_config.model).to("cpu")
    require(detector.model.input_dim == encoder.embedding_dim,
            "configured GRU input_dim must match the loaded encoder embedding dimension")
    optimizer = build_optimizer(detector, detector_config.optimizer)
    detector.train()

    lengths = torch.tensor([len(row) for row in rows], dtype=torch.long)
    waveforms = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=77.0)
    waveforms.requires_grad_()
    start = time.perf_counter()
    encoded = encoder.extract_batch(waveforms, lengths, sample_rate=16000)
    extraction_seconds = time.perf_counter() - start
    require(not encoded.features.requires_grad and not torch.is_inference(encoded.features),
            "features must be detached ordinary tensors usable by a trainable head")
    expected_lengths = lengths.clone()
    for kernel, stride in encoder.conv_geometry:
        expected_lengths = (expected_lengths - kernel) // stride + 1
    torch.testing.assert_close(encoded.valid_lengths, expected_lengths, rtol=0, atol=0)
    require(bool(torch.isfinite(encoded.features).all()), "non-finite encoder features")
    examples = [EmbeddingExample(encoded.features[i, :n], GENUINE)
                for i, n in enumerate(encoded.valid_lengths.tolist())]
    batch = collate_examples(examples)
    for key in ("features", "valid_lengths", "padding_mask"):
        torch.testing.assert_close(batch[key], getattr(encoded, key), rtol=0, atol=0)

    print("Checking speech/padding parity and one GRU training step...", flush=True)
    detector.eval()
    with torch.no_grad():
        start = time.perf_counter()
        logits = detector(batch).logits
        head_forward_seconds = time.perf_counter() - start
        require(tuple(logits.shape) == (2, 2) and bool(torch.isfinite(logits).all()), "invalid GRU logits")
        feature_errors, logit_errors = [], []
        for i, row in enumerate(rows):
            single = encoder.extract_batch(row[None], lengths[i:i+1], sample_rate=16000)
            feature_errors.append(float((single.features[0] - examples[i].features).abs().max()))
            torch.testing.assert_close(single.features[0], examples[i].features, atol=2e-4, rtol=2e-4)
            single_logits = detector(collate_examples([EmbeddingExample(single.features[0], GENUINE)])).logits[0]
            logit_errors.append(float((single_logits - logits[i]).abs().max()))
            torch.testing.assert_close(single_logits, logits[i], atol=2e-4, rtol=2e-4)
        changed_waveforms = waveforms.detach().clone()
        changed_waveforms[0, lengths[0]:] = -77.0
        repeated = encoder.extract_batch(changed_waveforms, lengths, sample_rate=16000)
        torch.testing.assert_close(repeated.features, encoded.features, atol=0, rtol=0)
        padded_batch = {**batch, "features": batch["features"].clone()}
        padded_batch["features"][batch["padding_mask"]] = 77.0
        padded_logits = detector(padded_batch).logits
        torch.testing.assert_close(padded_logits, logits, atol=0, rtol=0)

    head_before = {name: p.detach().clone() for name, p in detector.named_parameters()}
    start = time.perf_counter()
    epoch = train_epoch(detector, [batch], optimizer, device="cpu")
    training_step_seconds = time.perf_counter() - start
    require(epoch.steps == 1 and epoch.examples == 2, "training loop did not consume the batch")
    gradient_l2, changed_parameters = {}, []
    for name, parameter in detector.named_parameters():
        require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()),
                f"missing or non-finite head gradient: {name}")
        gradient_l2[name] = float(parameter.grad.norm())
        if not torch.equal(parameter.detach(), head_before[name]):
            changed_parameters.append(name)
    for component in ("input_norm", "projection", "gru", "classifier"):
        require(any(name.startswith(f"model.{component}.") and gradient_l2[name] > 0
                    for name in changed_parameters), f"no gradient-driven update in {component}")
    require(waveforms.grad is None, "encoder unexpectedly propagated waveform gradients")
    require(all(not p.requires_grad and p.grad is None for p in model.parameters()),
            "encoder is not frozen or received gradients")
    require(all(not module.training for module in model.modules()), "encoder entered training mode")
    encoder_after = model_digest(model)
    require(encoder_after == encoder_before, "encoder state changed after GRU update")

    report = {
        "status": "passed", "purpose": "Integration mechanics only; not spoof detection evaluation",
        "python": platform.python_version(), "platform": platform.platform(),
        "dependencies": {name: version(name) for name in ("torch", "fairseq", "numpy", "soundfile", "pytest")},
        "device": "cpu", "threads": torch.get_num_threads(), "seed": 42,
        "encoder_config": asdict(encoder_config), "detector_config": asdict(detector_config),
        "effective_smoke_training": {"epochs": 1, "batch_size": 2, "steps": 1,
                                     "seed": 42, "use_amp": False, "loss": "cross_entropy"},
        "checkpoint_sha256": encoder.checkpoint_sha256, "fixtures": fixtures,
        "features_shape": list(encoded.features.shape), "valid_lengths": encoded.valid_lengths.tolist(),
        "padding_mask_semantics": "True = ignored right-padding; lengths are embedding frames",
        "logits_shape": list(logits.shape), "loss_before_update": epoch.loss,
        "gradient_l2": gradient_l2, "updated_head_parameters": changed_parameters,
        "encoder_state_sha256_before": encoder_before, "encoder_state_sha256_after": encoder_after,
        "checks": {"encoder_state_unchanged": True, "encoder_gradients_none": True,
                   "encoder_eval_mode": True, "waveform_gradient_none": True,
                   "b1_b2_collation_exact": True, "waveform_padding_invariance_exact": True,
                   "feature_padding_invariance_exact": True,
                   "single_vs_batch_feature_max_abs_errors": feature_errors,
                   "single_vs_batch_logit_max_abs_errors": logit_errors},
        "timings_seconds": {"encoder_load_including_checkpoint_hash": load_seconds,
                            "first_unequal_batch_extraction": extraction_seconds,
                            "first_head_forward": head_forward_seconds,
                            "one_cached_feature_training_step": training_step_seconds},
        "limitations": ["Two genuine English clips from one LibriSpeech speaker; no spoof examples",
                        "One update from a randomly initialized head; no accuracy, EER, or calibration claim",
                        "CPU float32 only; timings are single observations, not a latency benchmark",
                        "Features are extracted before head training; no production audio/data pipeline tested"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {list(encoded.features.shape)} -> {list(logits.shape)}; loss={epoch.loss:.6f}")
    print(f"Encoder unchanged; {len(changed_parameters)} head tensors updated. Report: {output}")


if __name__ == "__main__":
    main()
