"""Engineering-only CUDA smoke check for the verified IndicWav2Vec Large."""

import argparse
import gc
from dataclasses import replace
from pathlib import Path
import tempfile
import wave

import numpy as np
import torch
import yaml

from src.backbones.errors import BackboneInputError, BackboneLoadError
from src.backbones.indic_wav2vec import IndicWav2VecConfig, IndicWav2VecExtractor
from src.backbones.trainable_indic_wav2vec import TrainableIndicWav2Vec


def wav_fixtures(directory):
    rows = []
    for i, length in enumerate((4_000, 1_920, 400, 4_000)):
        t = np.arange(length) / 16_000
        pcm = (8_192 * np.sin(2 * np.pi * (220 + i * 55) * t)).astype("<i2")
        path = directory / f"engineering-{i}.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setparams((1, 2, 16_000, 0, "NONE", "not compressed"))
            handle.writeframes(pcm.tobytes())
        with wave.open(str(path), "rb") as handle:
            rows.append(torch.from_numpy(
                np.frombuffer(handle.readframes(length), dtype="<i2").astype(np.float32) / 32768
            ))
    return rows


def compare(actual, expected, *, atol=0):
    assert actual.features.dtype == torch.float32 and actual.features.is_cuda
    assert actual.valid_lengths.dtype == torch.int64 and actual.valid_lengths.is_cuda
    assert actual.padding_mask.dtype == torch.bool and actual.padding_mask.is_cuda
    assert torch.equal(actual.valid_lengths, expected.valid_lengths)
    assert torch.equal(actual.padding_mask, expected.padding_mask)
    torch.testing.assert_close(actual.features, expected.features, atol=atol, rtol=0)
    return (actual.features - expected.features).abs().max().item()


def verify(config, directory):
    invalid_configs = (
        (config, "all", ValueError),
        (replace(config, checkpoint_sha256=None), "frozen", ValueError),
        (replace(config, output_layer=23), "last_two", ValueError),
        (replace(config, expected_embedding_dim=768), "last_two", ValueError),
        (replace(config, checkpoint_sha256="0" * 64), "last_two", BackboneLoadError),
    )
    for invalid_config, mode, error in invalid_configs:
        try:
            TrainableIndicWav2Vec(invalid_config, mode=mode)
        except error:
            continue
        raise AssertionError("invalid encoder configuration accepted")
    print(f"Invalid configurations rejected={len(invalid_configs)}", flush=True)
    rows = wav_fixtures(directory)
    lengths = torch.tensor([row.numel() for row in rows])
    samples = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True).cuda()
    with torch.no_grad():
        baseline = IndicWav2VecExtractor(config).extract_batch(samples, lengths)
    frozen = TrainableIndicWav2Vec(config).eval()
    actual = frozen(samples, lengths)
    parity_error = compare(actual, baseline)
    assert not actual.features.requires_grad and not torch.is_inference(actual.features)
    assert actual.features.shape == (4, 12, 1024)
    assert actual.valid_lengths.tolist() == [12, 5, 1, 12]
    assert torch.equal((~actual.padding_mask).sum(1), actual.valid_lengths)
    assert torch.count_nonzero(actual.features[actual.padding_mask]).item() == 0
    print(f"CUDA frozen parity max_abs={parity_error:.9g}, shape={tuple(actual.features.shape)}, "
          f"lengths={actual.valid_lengths.tolist()}", flush=True)
    frozen.train()
    assert all(not module.training for module in frozen.model.modules())
    assert not any(p.requires_grad for p in frozen.parameters())
    padded = torch.nn.functional.pad(samples, (0, 17), value=17)
    for i, n in enumerate(lengths.tolist()):
        padded[i, n:] = -21
    padding_error = compare(frozen(padded, lengths), actual)
    singleton_error = 0.0
    for i, row in enumerate(rows):
        single = frozen(row[None], lengths[i:i + 1])
        valid = actual.features[i, :actual.valid_lengths[i]]
        # The equal-length group changes CUDA kernel batch sizes relative to singletons.
        torch.testing.assert_close(single.features[0], valid, atol=1e-4, rtol=0)
        singleton_error = max(singleton_error, (single.features[0] - valid).abs().max().item())
    assert torch.isfinite(actual.features).all()
    invalid = [
        (samples.half(), lengths, 16_000),
        (samples[0], lengths, 16_000),
        (samples, lengths.float(), 16_000),
        (samples, lengths.clamp(max=399), 16_000),
        (samples, lengths + 4_000, 16_000),
        (samples, lengths, 8_000),
        (samples * 10, lengths, 16_000),
        (torch.full_like(samples, float("nan")), lengths, 16_000),
    ]
    for x, n, rate in invalid:
        try:
            frozen(x, n, sample_rate=rate)
        except BackboneInputError:
            continue
        raise AssertionError("invalid input accepted")
    print(f"Padding max_abs={padding_error:.9g}, singleton max_abs={singleton_error:.9g}, "
          f"invalid inputs rejected={len(invalid)}", flush=True)
    del frozen

    trainable = TrainableIndicWav2Vec(config, mode="last_two").train()
    expected_blocks = ("model.encoder.layers.22", "model.encoder.layers.23")
    assert trainable.trainable_block_names == expected_blocks
    for name, module in trainable.model.named_modules(prefix="model"):
        assert module.training == any(name == block or name.startswith(block + ".")
                                      for block in expected_blocks), name
    torch.cuda.reset_peak_memory_stats()
    output = trainable(samples, lengths)
    assert output.features.requires_grad and torch.isfinite(output.features).all()
    output.features[~output.padding_mask, :].square().mean().backward()
    trainable_names = []
    for name, param in trainable.named_parameters():
        expected = name.startswith(tuple(block + "." for block in expected_blocks))
        assert param.requires_grad == expected, name
        if expected:
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
            trainable_names.append(name)
        else:
            assert param.grad is None, name
    assert len(trainable_names) == 32
    for block in expected_blocks:
        assert any(torch.count_nonzero(p.grad).item()
                   for p in trainable.get_submodule(block).parameters()), block
    print(f"Backward finite gradient tensors={len(trainable_names)}, "
          f"peak allocated MiB={torch.cuda.max_memory_allocated() / 2**20:.1f}", flush=True)
    trainable.zero_grad(set_to_none=True)
    del output
    with torch.no_grad():
        eval_error = compare(trainable.eval()(samples, lengths), baseline)
    state_path = directory / "encoder-state.pt"
    torch.save(trainable.state_dict(), state_path)
    del trainable
    gc.collect()

    state = torch.load(str(state_path), map_location="cpu", weights_only=True, mmap=True)
    restored = TrainableIndicWav2Vec(config, mode="last_two").eval()
    for name, value in restored.state_dict().items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value.cpu(), state[name]), f"pretrained state changed: {name}"
    restored.load_state_dict(state)
    for name, value in restored.state_dict().items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value.cpu(), state[name]), f"reloaded state differs: {name}"
    with torch.no_grad():
        reload_error = compare(restored(samples, lengths), baseline)
    metadata = state["_extra_state"]
    assert metadata["checkpoint_sha256"] == config.checkpoint_sha256
    for invalid in ({**metadata, "mode": "frozen"},
                    {**metadata, "checkpoint_sha256": "0" * 64}, None):
        if invalid is None:
            del state["_extra_state"]
        else:
            state["_extra_state"] = invalid
        try:
            restored.load_state_dict(state, strict=False)
        except RuntimeError:
            continue
        raise AssertionError("invalid state metadata accepted")
    # Release the mapped file before TemporaryDirectory cleanup on Windows.
    del state
    print(f"Eval max_abs={eval_error:.9g}, reload max_abs={reload_error:.9g}, "
          f"base={config.checkpoint_sha256}, blocks={expected_blocks}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    settings = yaml.safe_load(Path("configs/backbones.yaml").read_text(encoding="utf-8"))
    config = replace(IndicWav2VecConfig(**settings["indic_wav2vec"]),
                     checkpoint_path=str(args.checkpoint), device="cuda")
    torch.manual_seed(0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    print(f"PyTorch {torch.__version__}, GPU {torch.cuda.get_device_name()}", flush=True)
    with tempfile.TemporaryDirectory() as directory:
        verify(config, Path(directory))


if __name__ == "__main__":
    main()
