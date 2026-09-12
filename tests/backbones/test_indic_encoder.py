"""Fast runtime tests using a tiny random *upstream* model, not pretrained evidence."""
import hashlib
import tempfile
import unittest
from pathlib import Path

import yaml
import numpy as np
import torch
from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Config, Wav2Vec2Model
from omegaconf import OmegaConf

from src.backbones.indic_wav2vec import IndicWav2VecConfig, IndicWav2VecExtractor
from src.backbones.errors import BackboneInputError, BackboneLoadError
from src.backbones.schemas import ChunkMetadata


class IndicEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(4)
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name) / 'tiny.pt'
        cfg = Wav2Vec2Config(encoder_layers=2, encoder_embed_dim=16,
            encoder_ffn_embed_dim=32, encoder_attention_heads=2,
            conv_feature_layers='[(8, 10, 5), (8, 3, 2)]',
            conv_pos=8, conv_pos_groups=2, latent_vars=4, latent_groups=2,
            final_dim=8, dropout=0.3, encoder_layerdrop=0.3)
        cfg._name = 'wav2vec2'
        torch.save({'cfg': {'model': yaml.safe_load(OmegaConf.to_yaml(OmegaConf.structured(cfg))),
            'task': {'normalize': True}}, 'model': Wav2Vec2Model(cfg).state_dict()}, cls.path)
        cls.digest = hashlib.sha256(cls.path.read_bytes()).hexdigest()
        cls.encoder = cls.make_encoder().load()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @classmethod
    def make_encoder(cls, **kwargs):
        return IndicWav2VecExtractor(IndicWav2VecConfig(str(cls.path), **kwargs))

    def test_checkpoint_geometry_and_dynamic_dimension(self):
        self.assertEqual(self.encoder.embedding_dim, 16)
        self.assertEqual(self.encoder.min_samples, 20)
        self.assertEqual(self.encoder.frame_hop_ms, 0.625)
        self.assertEqual(self.encoder.checkpoint_sha256, self.digest)
        self.assertTrue(self.encoder.normalize)

    def test_unequal_batch_padding_and_single_parity(self):
        x = torch.rand(3, 100) * 0.1
        lengths = torch.tensor([100, 60, 100])
        x[1, 60:] = 99  # Finite padding is ignored, even outside PCM range.
        batch = self.encoder.extract_batch(x, lengths)
        self.assertEqual(batch.features.shape, (3, 9, 16))
        self.assertEqual(batch.valid_lengths.tolist(), [9, 5, 9])
        self.assertEqual(batch.padding_mask.dtype, torch.bool)
        self.assertTrue(torch.equal((~batch.padding_mask).sum(1), batch.valid_lengths))
        self.assertEqual(torch.count_nonzero(batch.features[batch.padding_mask]).item(), 0)
        for i, n in enumerate(lengths.tolist()):
            single = self.encoder.extract_batch(x[i:i+1, :n], torch.tensor([n]))
            torch.testing.assert_close(batch.features[i, :batch.valid_lengths[i]], single.features[0], atol=2e-5, rtol=2e-5)
        changed = x.clone()
        changed[1, 60:] = -99
        torch.testing.assert_close(batch.features, self.encoder.extract_batch(changed, lengths).features, atol=0, rtol=0)

    def test_frozen_eval_and_downstream_backward(self):
        model = self.encoder._model
        model.train()  # Even accidental mode changes must not enable dropout.
        x = torch.rand(1, 100, requires_grad=True)
        before = [p.detach().clone() for p in model.parameters()]
        batch = self.encoder.extract_batch(x, torch.tensor([100]))
        self.assertFalse(batch.features.requires_grad)
        self.assertFalse(torch.is_inference(batch.features))
        layer = torch.nn.Linear(batch.embedding_dim, 1)
        old_weight = layer.weight.detach().clone()
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
        layer(batch.features).square().mean().backward()
        self.assertTrue(torch.isfinite(layer.weight.grad).all())
        self.assertGreater(layer.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(layer.weight, old_weight))
        self.assertIsNone(x.grad)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
        self.assertTrue(all(not m.training for m in model.modules()))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, model.parameters())))
        torch.testing.assert_close(batch.features, self.encoder.extract_batch(x, torch.tensor([100])).features, atol=0, rtol=0)

    def test_layer_selection_matches_upstream(self):
        encoder = self.make_encoder(output_layer=0).load()
        x = torch.rand(1, 100)
        batch = encoder.extract_batch(x, torch.tensor([100]))
        with torch.no_grad():
            reference = encoder._model.extract_features(torch.nn.functional.layer_norm(x, (100,)), None, mask=False, layer=0)['x']
        torch.testing.assert_close(batch.features, reference, atol=0, rtol=0)
        with self.assertRaises(BackboneLoadError):
            self.make_encoder(output_layer=2).load()

    def test_invalid_inputs(self):
        x = torch.zeros(1, 100)
        cases = [(x[0], torch.tensor([100])), (x.to(torch.int16), torch.tensor([100])),
            (x, torch.tensor([0])), (x, torch.tensor([101])), (x, torch.tensor([19])),
            (x, torch.tensor([100.])), (x, torch.tensor([[100]])),
            (torch.empty(0, 100), torch.tensor([], dtype=torch.long)),
            (torch.full_like(x, float('nan')), torch.tensor([100])),
            (torch.full_like(x, float('inf')), torch.tensor([100])),
            (torch.full_like(x, 2), torch.tensor([100]))]
        for samples, lengths in cases:
            with self.subTest(shape=samples.shape, lengths=lengths):
                with self.assertRaises(BackboneInputError):
                    self.encoder.extract_batch(samples, lengths)
        with self.assertRaises(BackboneInputError):
            self.encoder.extract_batch(x, torch.tensor([100]), sample_rate=8000)

    def test_minimum_length_and_silence(self):
        result = self.encoder.extract_batch(torch.zeros(1, 20), torch.tensor([20]))
        self.assertEqual(result.valid_lengths.tolist(), [1])
        self.assertTrue(torch.isfinite(result.features).all())

    def test_outer_inference_and_autocast_do_not_leak(self):
        x = torch.rand(1, 100)
        expected = self.encoder.extract_batch(x, torch.tensor([100]))
        with torch.inference_mode(), torch.autocast('cpu', dtype=torch.bfloat16):
            actual = self.encoder.extract_batch(x, torch.tensor([100]))
        self.assertFalse(torch.is_inference(actual.features))
        self.assertEqual(actual.features.dtype, torch.float32)
        torch.testing.assert_close(actual.features, expected.features, atol=0, rtol=0)

    def test_identity_dimension_and_corrupt_checkpoint(self):
        with self.assertRaisesRegex(BackboneLoadError, 'SHA-256'):
            self.make_encoder(checkpoint_sha256='0' * 64).load()
        with self.assertRaisesRegex(BackboneLoadError, 'dimension'):
            self.make_encoder(expected_embedding_dim=1024).load()
        with self.assertRaises(BackboneLoadError):
            IndicWav2VecExtractor(IndicWav2VecConfig('missing.pt')).load()
        corrupt = Path(self.temp.name) / 'corrupt.pt'
        corrupt.write_bytes(b'not a checkpoint')
        with self.assertRaises(BackboneLoadError):
            IndicWav2VecExtractor(IndicWav2VecConfig(str(corrupt))).load()

    def test_single_chunk_bridge(self):
        x = np.zeros(100, dtype=np.float32)
        result = self.encoder.extract(x, ChunkMetadata('fixture', 0, 6.25))
        self.assertEqual(result.features.shape, (9, 16))
        self.assertEqual(result.route, 'indic')
        self.assertEqual(result.checkpoint_version, self.digest)
        with self.assertRaises(BackboneInputError):
            self.encoder.extract(x, ChunkMetadata('fixture', 0, 1000))
