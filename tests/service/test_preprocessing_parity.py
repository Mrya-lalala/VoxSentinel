"""Offline C1/release parity gate: real decoding/resampling, captured B1 inputs."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
import soxr

from src.audio.prepare import decode_mono_16k
from src.audio.identity import TRAINED_PREPROCESSING_VERSION
from src.config import ModelConfig
from src.dataset_prep.materialize import sha256_file
from src.dataset_prep.windows import select_window
from src.detectors.checkpoints import save_checkpoint
from src.detectors.registry import create_detector
from src.scoring.predict import FilePredictor
from src.service import api
from src.service.engine import ServiceEngine, load_detector


class RecordingEncoder:
    def __init__(self, *args):
        self.inputs = []

    def load(self):
        return self

    def extract_batch(self, samples, lengths, *, sample_rate):
        assert sample_rate == 16000
        self.inputs.append((samples.clone(), lengths.clone()))
        # Waveform-dependent features make score parity sensitive to input differences.
        features = torch.stack([samples.mean(1), samples.square().mean(1)], dim=-1)[:, None, :]
        return SimpleNamespace(features=features, valid_lengths=torch.ones(1, dtype=torch.long),
                               padding_mask=torch.zeros(1, 1, dtype=torch.bool))


@pytest.fixture
def release(tmp_path, monkeypatch):
    monkeypatch.setattr('src.scoring.predict.IndicWav2VecExtractor', RecordingEncoder)
    config = {'input_dim': 2, 'hidden_size': 4}
    torch.manual_seed(7)
    head = tmp_path / 'head.pt'
    identity = {'checkpoint_sha256': 'a' * 64, 'selected_layer': None, 'embedding_dim': 2,
                'preprocessing_version': TRAINED_PREPROCESSING_VERSION, 'dtype': 'float32'}
    save_checkpoint(head, create_detector(ModelConfig('gru', config)), 'gru', config,
                    metadata={'encoder': identity})
    spec = {'schema': 'voxsentinel.inference_release.v1', 'release_id': 'test',
            'head': {'file': head.name, 'sha256': sha256_file(head)},
            'encoder': {'checkpoint_path': 'unused.pt', 'checkpoint_sha256': 'a' * 64,
                        'output_layer': None, 'expected_embedding_dim': 2, 'device': 'cpu'},
            'preprocessing_version': TRAINED_PREPROCESSING_VERSION,
            # Deliberately nondefault: tests must catch C1 ignoring the release window.
            'window': {'max_seconds': 2., 'min_seconds': 1.}, 'threshold': .37,
            'training_languages': [], 'validation_status': 'test only'}
    path = tmp_path / 'release.json'
    path.write_text(json.dumps(spec))
    monkeypatch.setenv('INFERENCE_RELEASE', str(path))
    monkeypatch.delenv('DETECTION_THRESHOLD', raising=False)
    monkeypatch.setattr(api, 'ENCODER_CHECKPOINT', 'unused.pt')
    monkeypatch.setattr(api, 'DEVICE', 'cpu')
    return path


@pytest.mark.parametrize('rate', [8000, 16000, 22050, 24000, 44100, 48000])
@pytest.mark.parametrize('kind', ['mono', 'stereo', 'overflow', 'short', 'silent', 'long'])
def test_service_release_waveform_window_and_score_parity(release, tmp_path, rate, kind):
    duration = .5 if kind == 'short' else 5.13 if kind == 'long' else 1.3
    t = np.arange(round(duration * rate)) / rate
    data = (.15 * np.sin(2 * np.pi * 173 * t)).astype(np.float32)
    if kind == 'stereo':
        data = np.stack([data, data * .3], axis=1)
    elif kind == 'overflow':
        data *= 12  # FLOAT WAV amplitudes > 1 must be attenuated, never rejected/clipped.
    elif kind == 'silent':
        data[:] = 0
    elif kind == 'long':
        data[-2 * rate:] *= 4
    audio_path = tmp_path / 'input.wav'
    sf.write(audio_path, data, rate, subtype='FLOAT')
    predictor = FilePredictor(release)
    service = api.create_engine()
    expected = predictor.predict(audio_path)
    results, metadata = service.predict(audio_path.read_bytes(), 'test')
    decoded = decode_mono_16k(audio_path)
    # Independent numerical check of the declared decoder/downmix/soxr/gain contract.
    reference = data.mean(axis=1) if data.ndim == 2 else data
    if rate != 16000:
        reference = soxr.resample(reference, rate, 16000, quality='HQ')
    peak = float(np.max(np.abs(reference)))
    if peak > 1:
        reference = (reference * (1 / peak)).astype(np.float32)
    np.testing.assert_array_equal(decoded.samples, reference)
    if kind in ('short', 'silent'):
        assert not results and metadata['chunks_used'] == 0
        assert expected['status'] == 'insufficient_audio'
        assert not service.encoder.inputs
    else:
        assert len(results) == 1
        window = select_window(decoded.samples, 16000, predictor.policy)
        assert (results[0].start_sample, results[0].end_sample) == (window.start_sample, window.end_sample)
        if kind == 'long':
            assert window.start_sample == len(decoded.samples) - 32000  # strongest flush tail
        assert torch.equal(service.encoder.inputs[0][0], predictor.encoder.inputs[0][0])
        assert torch.equal(service.encoder.inputs[0][1], predictor.encoder.inputs[0][1])
        assert results[0].score == expected['synthetic_score']
    assert service.threshold == .37 and service.threshold_source == 'release'


def test_incompatible_checkpoint_fails_before_scoring(release):
    path = release.parent / 'head.pt'
    payload = torch.load(path, weights_only=False)
    payload['metadata']['encoder']['preprocessing_version'] = 'audio-v1'
    torch.save(payload, path)
    with pytest.raises(ValueError, match='preprocessing_version'):
        load_detector(path)


def test_matching_but_unsupported_release_and_checkpoint_identity_rejected(release):
    spec = json.loads(release.read_text())
    spec['preprocessing_version'] = 'unknown-prep'
    head = release.parent / 'head.pt'
    payload = torch.load(head, weights_only=False)
    payload['metadata']['encoder']['preprocessing_version'] = 'unknown-prep'
    torch.save(payload, head)
    spec['head']['sha256'] = sha256_file(head)
    release.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='preprocessing_version'):
        FilePredictor(release)


def test_release_threshold_override_and_api_provenance(release, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv('DETECTION_THRESHOLD', '.7')
    assert api.create_engine().threshold == .7
    monkeypatch.delenv('DETECTION_THRESHOLD')
    monkeypatch.setattr(api, 'engine', api.create_engine())
    path = release.parent / 'silent.wav'
    sf.write(path, np.zeros(32000, dtype=np.float32), 16000)
    response = TestClient(api.app).post('/detect', files={'file': ('silent.wav', path.read_bytes(), 'audio/wav')})
    assert response.status_code == 200
    assert response.json()['threshold_source'] == 'release'


@pytest.mark.parametrize('threshold', [float('nan'), float('inf'), -.1, 1.1])
def test_invalid_operator_threshold_fails_startup(release, monkeypatch, threshold):
    monkeypatch.setenv('DETECTION_THRESHOLD', str(threshold))
    with pytest.raises(ValueError, match='Threshold'):
        api.create_engine()
