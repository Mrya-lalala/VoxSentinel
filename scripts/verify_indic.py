"""Real-checkpoint verification only; no detector training or accuracy evaluation.

python -m scripts.verify_indic --audio clip1.flac clip2.flac --output artifacts/encoder-verification.json
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import resource
import statistics
import time

import numpy as np
import soundfile as sf
import torch
import yaml
from src.backbones.errors import BackboneInputError
from src.backbones.indic_wav2vec import IndicWav2VecConfig, IndicWav2VecExtractor
from src.backbones.schemas import ChunkMetadata


def model_digest(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', nargs=2, required=True)
    parser.add_argument('--config', default='configs/backbones.yaml')
    parser.add_argument('--output', required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error('--repeats must be >=2')
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    rows, fixtures = [], []
    for path in args.audio:
        audio, rate = sf.read(path, dtype='float32')
        if rate != 16000 or audio.ndim != 1 or len(audio) < 64000:
            parser.error('verification requires two mono 16 kHz speech clips of at least 4 seconds')
        rows.append(torch.from_numpy(audio))
        fixtures.append({'filename': Path(path).name, 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                         'sample_rate': rate, 'samples': len(audio)})
    config = IndicWav2VecConfig(**yaml.safe_load(Path(args.config).read_text())['indic_wav2vec'])
    if config.device != 'cpu':
        parser.error('this benchmark currently verifies CPU timing only')
    start = time.perf_counter()
    encoder = IndicWav2VecExtractor(config).load()
    load_seconds = time.perf_counter() - start
    model = encoder._model  # Verification introspection, not the consumer interface.
    before = model_digest(model)
    report = {'checkpoint_sha256': encoder.checkpoint_sha256, 'checkpoint_bytes': Path(config.checkpoint_path).stat().st_size,
        'python': platform.python_version(), 'platform': platform.platform(), 'architecture': platform.machine(),
        'torch': torch.__version__, 'numpy': np.__version__, 'device': config.device,
        'threads': torch.get_num_threads(), 'fixtures': fixtures, 'load_seconds_including_hash': load_seconds,
        'embedding_dim': encoder.embedding_dim, 'normalize': encoder.normalize, 'num_layers': encoder.num_layers,
        'conv_geometry_kernel_stride': encoder.conv_geometry, 'min_samples': encoder.min_samples,
        'frame_hop_ms': encoder.frame_hop_ms, 'parameter_count': sum(p.numel() for p in model.parameters()),
        'extractor_mode': str(model.cfg.extractor_mode), 'layer_norm_first': bool(model.cfg.layer_norm_first),
        'checks': {}, 'benchmarks': []}
    # Unequal lengths plus a duplicate length exercise both grouping and padding.
    x = torch.stack([rows[0][:64000], rows[1][:64000], rows[0][:64000]])
    lengths = torch.tensor([64000, 48000, 64000])
    x[1, 48000:] = 77
    result = encoder.extract_batch(x, lengths)
    assert result.features.shape == (3, 199, encoder.embedding_dim)
    assert result.valid_lengths.tolist() == [199, 149, 199]
    assert result.valid_lengths.dtype == torch.int64
    assert result.padding_mask.dtype == torch.bool
    assert torch.equal((~result.padding_mask).sum(1), result.valid_lengths)
    assert torch.count_nonzero(result.features[result.padding_mask]) == 0
    assert torch.isfinite(result.features).all()
    parity_errors = []
    for i, n in enumerate(lengths.tolist()):
        single = encoder.extract_batch(x[i:i+1, :n], torch.tensor([n]))
        valid = result.features[i, :single.valid_lengths[0]]
        parity_errors.append((valid - single.features[0]).abs().max().item())
        torch.testing.assert_close(valid, single.features[0], atol=2e-4, rtol=2e-4)
    changed = x.clone()
    changed[1, 48000:] = -77
    torch.testing.assert_close(result.features, encoder.extract_batch(changed, lengths).features, atol=0, rtol=0)
    report['checks']['batch_padding'] = {'shape': list(result.features.shape), 'valid_lengths': result.valid_lengths.tolist(),
                                        'single_max_abs_errors': parity_errors, 'padding_invariance_exact': True}
    # Direct upstream parity also verifies normalization is applied to each valid clip.
    source = rows[0][:16000][None]
    layer_errors = {}
    for layer in [None] + list(range(encoder.num_layers)):
        encoder.config = replace(config, output_layer=layer)
        actual = encoder.extract_batch(source, torch.tensor([16000]))
        with torch.no_grad():
            normalized = torch.nn.functional.layer_norm(source, (16000,)) if encoder.normalize else source
            reference = model.extract_features(source=normalized, padding_mask=None, mask=False, layer=layer)['x']
        torch.testing.assert_close(actual.features, reference, atol=0, rtol=0)
        layer_errors[str(layer)] = (actual.features - reference).abs().max().item()
    encoder.config = config
    report['checks']['all_layers_direct_upstream_max_abs_error'] = layer_errors
    model.train()  # Simulate accidental train mode on the frozen model.
    differentiable_input = source.clone().requires_grad_()
    with torch.inference_mode(), torch.autocast('cpu', dtype=torch.bfloat16):
        detached = encoder.extract_batch(differentiable_input, torch.tensor([16000]))
    assert not detached.features.requires_grad and not torch.is_inference(detached.features)
    assert all(not module.training for module in model.modules())
    layer = torch.nn.Linear(encoder.embedding_dim, 1)  # Disposable gradient probe only.
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)
    old_weight = layer.weight.detach().clone()
    layer(detached.features).square().mean().backward()
    assert torch.isfinite(layer.weight.grad).all() and layer.weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(layer.weight, old_weight)
    assert differentiable_input.grad is None
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    torch.testing.assert_close(detached.features, encoder.extract_batch(source, torch.tensor([16000])).features, atol=0, rtol=0)
    assert model_digest(model) == before
    report['checks']['frozen_downstream_gradient'] = {'head_updated': True, 'encoder_state_unchanged': True,
        'encoder_gradients_none': True, 'input_gradient_none': True, 'repeatability_exact': True,
        'works_under_outer_inference_mode_and_autocast': True}
    invalid = [(source[0], torch.tensor([16000])), (source.to(torch.int16), torch.tensor([16000])),
        (source, torch.tensor([0])), (source, torch.tensor([16001])), (source, torch.tensor([399])),
        (source, torch.tensor([16000.])), (source, torch.tensor([[16000]])),
        (torch.empty(0, 16000), torch.tensor([], dtype=torch.long)),
        (torch.full_like(source, float('nan')), torch.tensor([16000])),
        (torch.full_like(source, float('inf')), torch.tensor([16000])),
        (torch.full_like(source, 2), torch.tensor([16000]))]
    for waveform, counts in invalid:
        try:
            encoder.extract_batch(waveform, counts)
        except BackboneInputError:
            pass
        else:
            raise AssertionError('invalid input accepted')
    try:
        encoder.extract_batch(source, torch.tensor([16000]), sample_rate=8000)
    except BackboneInputError:
        pass
    else:
        raise AssertionError('wrong sample rate accepted')
    minimum = encoder.extract_batch(torch.zeros(1, encoder.min_samples), torch.tensor([encoder.min_samples]))
    assert minimum.valid_lengths.tolist() == [1] and torch.isfinite(minimum.features).all()
    # Silence is an acoustic input, not a risk decision or a speech-quality claim.
    silent = encoder.extract_batch(torch.zeros(1, 16000), torch.tensor([16000]))
    assert torch.isfinite(silent.features).all()
    bridge = encoder.extract(source[0].numpy(), ChunkMetadata('speech', 0, 1000))
    np.testing.assert_array_equal(bridge.features, detached.features[0].numpy())
    report['checks']['invalid_inputs_rejected'] = len(invalid) + 1
    report['checks']['minimum_length_silence_and_router_bridge'] = True
    for counts in ([48000], [64000], [64000, 48000], [64000, 64000]):
        batch = torch.stack([rows[i][:max(counts)] for i in range(len(counts))])
        valid = torch.tensor(counts)
        encoder.extract_batch(batch, valid)  # warm-up, excluded
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            encoder.extract_batch(batch, valid)
            times.append(time.perf_counter() - start)
        median = statistics.median(times)
        report['benchmarks'].append({'input_samples': counts, 'warmup_runs': 1, 'repeats': args.repeats,
            'seconds': times, 'median_seconds': median, 'min_seconds': min(times), 'max_seconds': max(times),
            'rtf_total_audio_median': median / (sum(counts) / 16000)})
    report['peak_process_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # macOS bytes
    report['status'] = 'passed'
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
