"""Release dispatch tests use small fixture weights, never a deployed AASIST model."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from src.audio.identity import TRAINED_PREPROCESSING_VERSION
from src.config import ModelConfig
from src.dataset_prep.materialize import sha256_file
from src.detectors.aasist import AasistDetector
from src.detectors.checkpoints import save_checkpoint
from src.detectors.registry import create_detector
from src.scoring.predict import FilePredictor
from src.service import api


class FixtureEncoder:
    def __init__(self, config):
        self.loaded = False

    def load(self):
        self.loaded = True

    def extract_batch(self, samples, lengths, *, sample_rate):
        assert sample_rate == 16000
        features = torch.stack([samples.mean(1), samples.square().mean(1)], -1)
        features = features[:, None, :].repeat(1, 8, 1)
        return SimpleNamespace(features=features, valid_lengths=torch.full((len(samples),), 8),
                               padding_mask=torch.zeros(len(samples), 8, dtype=torch.bool))


@pytest.fixture
def aasist_release(tmp_path, monkeypatch):
    monkeypatch.setattr('src.scoring.predict.IndicWav2VecExtractor', FixtureEncoder)
    config = dict(input_dim=2, pooled_bins=4, channels=4, latent_nodes=4,
                  gat_dims=[4, 2], feature_standardization=True)
    with torch.random.fork_rng():
        torch.manual_seed(17)
        detector = create_detector(ModelConfig('aasist', config)).eval()
        detector.model.standardizer.fit([SimpleNamespace(features=torch.randn(12, 2), label=0)])
    head = tmp_path / 'fixture_head.pt'
    identity = dict(checkpoint_sha256='a' * 64, selected_layer=None, embedding_dim=2,
                    preprocessing_version=TRAINED_PREPROCESSING_VERSION, dtype='float32')
    save_checkpoint(head, detector, 'aasist', config, metadata={'encoder': identity})
    spec = dict(schema='voxsentinel.inference_release.v1', release_id='aasist-fixture-only',
                head=dict(file=head.name, sha256=sha256_file(head), model_name='aasist'),
                encoder=dict(checkpoint_path='unused.pt', checkpoint_sha256='a' * 64,
                             output_layer=None, expected_embedding_dim=2, device='cpu'),
                preprocessing_version=TRAINED_PREPROCESSING_VERSION,
                window=dict(min_seconds=1., max_seconds=2.), threshold=.5,
                training_languages=[], validation_status='fixture only')
    path = tmp_path / 'release.json'
    path.write_text(json.dumps(spec))
    monkeypatch.setenv('INFERENCE_RELEASE', str(path))
    monkeypatch.delenv('DETECTION_THRESHOLD', raising=False)
    monkeypatch.setattr(api, 'ENCODER_CHECKPOINT', 'unused.pt')
    monkeypatch.setattr(api, 'DEVICE', 'cpu')
    return path, detector


def rewrite_release(path, mutate_spec=None, mutate_head=None):
    spec = json.loads(path.read_text())
    if mutate_head:
        head = path.parent / spec['head']['file']
        payload = torch.load(head, weights_only=False)
        mutate_head(payload)
        torch.save(payload, head)
        spec['head']['sha256'] = sha256_file(head)
    if mutate_spec:
        mutate_spec(spec)
    path.write_text(json.dumps(spec))


def test_aasist_release_roundtrip_and_service_parity(aasist_release, tmp_path):
    path, original = aasist_release
    predictor = FilePredictor(path)
    assert isinstance(predictor.detector, AasistDetector)
    for key, value in original.state_dict().items():
        restored = predictor.detector.state_dict()[key]
        assert torch.equal(value, restored) if isinstance(value, torch.Tensor) else value == restored
    assert not predictor.detector.training
    assert all(not p.requires_grad for p in predictor.detector.parameters())
    service = api.create_engine()
    assert service.encoder.loaded and service.detector_ready
    assert service.threshold == .5 and service.threshold_source == 'release'
    clip = tmp_path / 'fixture.wav'
    samples = (.2 * np.sin(2 * np.pi * 173 * np.arange(48000) / 16000)).astype('float32')
    sf.write(clip, samples, 16000, subtype='FLOAT')
    prediction = predictor.predict(clip)
    results, _ = service.predict(clip.read_bytes(), 'fixture')
    assert len(results) == 1
    assert results[0].score == pytest.approx(prediction['synthetic_score'], abs=1e-7)
    assert (results[0].start_sample, results[0].end_sample) == (
        prediction['window']['start_sample'], prediction['window']['end_sample'])
    assert ('spoof' if results[0].score >= service.threshold else 'genuine') == prediction['prediction']


@pytest.mark.parametrize('identity', ['unknown', None, 1, 'AASIST'])
def test_unsupported_release_identity_rejected(aasist_release, identity):
    path, _ = aasist_release
    rewrite_release(path, lambda spec: spec['head'].update(model_name=identity))
    with pytest.raises(ValueError, match='Unsupported release detector'):
        FilePredictor(path)


@pytest.mark.parametrize('omitted', [True, False])
def test_manifest_checkpoint_identity_mismatch_rejected(aasist_release, omitted):
    path, _ = aasist_release
    def mutate(spec):
        if omitted:
            spec['head'].pop('model_name')
        else:
            spec['head']['model_name'] = 'gru'
    rewrite_release(path, mutate)
    with pytest.raises(ValueError, match='not gru'):
        api.create_engine()


@pytest.mark.parametrize('defect', ['disabled', 'missing_config', 'missing_state', 'unfitted'])
def test_invalid_aasist_standardizer_fails_at_startup(aasist_release, defect):
    path, _ = aasist_release
    def mutate(payload):
        if defect == 'disabled':
            payload['model_config']['feature_standardization'] = False
        elif defect == 'missing_config':
            payload['model_config'].pop('feature_standardization')
        elif defect == 'missing_state':
            payload['model_state_dict'].pop('model.standardizer.scale')
        else:
            payload['model_state_dict']['model.standardizer.fitted'] = torch.tensor(False)
    rewrite_release(path, mutate_head=mutate)
    with pytest.raises((ValueError, RuntimeError), match='standardiz'):
        api.create_engine()
