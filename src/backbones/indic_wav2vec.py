"""Frozen, local-checkpoint IndicWav2Vec acoustic encoder (no ASR decoding)."""
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np
from .base import BackboneExtractor
from .errors import BackboneInputError, BackboneLoadError
from .schemas import ChunkMetadata, EmbeddingSequence

if TYPE_CHECKING:
    import torch
    from .tensor_interface import EncoderBatch


@dataclass(frozen=True)
class IndicWav2VecConfig:
    checkpoint_path: str
    device: str = "cpu"
    output_layer: int | None = None  # Zero-based transformer block; None = final output.
    backbone_id: str = "indicwav2vec"
    checkpoint_version: str | None = None
    checkpoint_sha256: str | None = None
    expected_embedding_dim: int | None = None

    def __post_init__(self):
        if self.output_layer is not None and (type(self.output_layer) is not int or self.output_layer < 0):
            raise ValueError("output_layer must be None or a nonnegative integer")
        if self.expected_embedding_dim is not None and (
            type(self.expected_embedding_dim) is not int or self.expected_embedding_dim <= 0
        ):
            raise ValueError("expected_embedding_dim must be a positive integer")
        if self.checkpoint_sha256 is not None and (
            len(self.checkpoint_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.checkpoint_sha256)
        ):
            raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 hex digest")


class IndicWav2VecExtractor(BackboneExtractor):
    """Caller owns decoding, mono conversion, resampling and chunking.

    Input is finite floating PCM in [-1,1], at 16 kHz. This wrapper owns only
    checkpoint-required per-utterance normalization. It never downloads weights.
    Unequal lengths are encoded in exact-length groups to avoid padding effects.
    """
    def __init__(self, config: IndicWav2VecConfig) -> None:
        self.config = config
        self.backbone_id = config.backbone_id
        self._model = None
        self.checkpoint_sha256 = None

    def load(self) -> "IndicWav2VecExtractor":
        self._load_model()
        return self

    def _load_model(self):
        if self._model is not None:
            return self._model
        checkpoint = Path(self.config.checkpoint_path)
        if not checkpoint.is_file():
            raise BackboneLoadError(f"IndicWav2Vec checkpoint not found: {checkpoint}")
        digest = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        actual_hash = digest.hexdigest()
        if self.config.checkpoint_sha256 and actual_hash != self.config.checkpoint_sha256:
            raise BackboneLoadError("IndicWav2Vec checkpoint SHA-256 mismatch")
        try:
            import torch
            from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Config, Wav2Vec2Model
            from omegaconf import OmegaConf
            # Official checkpoints contain Python config objects: use trusted files only.
            # mmap avoids materializing training optimizer tensors in this checkpoint.
            state = torch.load(str(checkpoint), map_location="cpu", mmap=True, weights_only=False)
            cfg = state["cfg"]
            model_cfg = cfg["model"]
            if model_cfg.get("_name") != "wav2vec2":
                raise ValueError("expected a pretrained wav2vec2 checkpoint, not a CTC fine-tune")
            normalize = cfg["task"].get("normalize")
            if type(normalize) is not bool:
                raise ValueError("checkpoint must explicitly declare task.normalize")
            merged = OmegaConf.merge(OmegaConf.structured(Wav2Vec2Config()), model_cfg)
            # Build the acoustic model directly; training task/dataset paths aren't needed.
            model = Wav2Vec2Model(merged)
            model.load_state_dict(state["model"], strict=True, model_cfg=merged)
            del state
            model = model.float().to(torch.device(self.config.device)).eval()
            model.requires_grad_(False)
            dim = int(merged.encoder_embed_dim)
            if self.config.expected_embedding_dim is not None and dim != self.config.expected_embedding_dim:
                raise ValueError(f"embedding dimension mismatch: checkpoint={dim}, expected={self.config.expected_embedding_dim}")
            if self.config.output_layer is not None and self.config.output_layer >= len(model.encoder.layers):
                raise ValueError(f"output_layer must be less than {len(model.encoder.layers)}")
            # Read actual Conv1d modules, not a duration/frames approximation.
            convs = [m for m in model.feature_extractor.modules() if isinstance(m, torch.nn.Conv1d)]
            if not convs or any(m.padding != (0,) or m.dilation != (1,) for m in convs):
                raise ValueError("unsupported convolution geometry")
            geometry = [(m.kernel_size[0], m.stride[0]) for m in convs]
            stride, receptive = 1, 1
            for kernel, step in geometry:
                receptive += (kernel - 1) * stride
                stride *= step
            if int(merged.crop_seq_to_multiple) != 1:
                raise ValueError("crop_seq_to_multiple != 1 is not supported")
            self.normalize = normalize
            self._embedding_dim = dim
            self.num_layers = len(model.encoder.layers)
            self.conv_geometry = geometry
            self.min_samples = receptive
            self.frame_hop_ms = stride / 16_000 * 1000
            self.checkpoint_sha256 = actual_hash
            self._model = model
        except Exception as exc:
            raise BackboneLoadError(f"Unable to load IndicWav2Vec checkpoint: {exc}") from exc
        return self._model

    @property
    def embedding_dim(self) -> int:
        self._load_model()
        return self._embedding_dim

    def extract_batch(
        self, samples: "torch.Tensor", valid_lengths: "torch.Tensor", *, sample_rate: int = 16_000
    ) -> "EncoderBatch":
        """Lengths count samples on input, frames on output.

        Right-padding after each valid length is ignored, but all values must be
        finite. Outputs are ordinary detached tensors, allowing head backward.
        """
        import torch
        from .tensor_interface import EncoderBatch
        if sample_rate != 16_000:
            raise BackboneInputError("backbones require 16 kHz audio")
        if (not isinstance(samples, torch.Tensor) or samples.ndim != 2
                or 0 in samples.shape or samples.layout != torch.strided):
            raise BackboneInputError("samples must have non-empty shape [B,S]")
        if not samples.is_floating_point() or not torch.isfinite(samples).all():
            raise BackboneInputError("samples must be finite floating PCM")
        if not isinstance(valid_lengths, torch.Tensor) or valid_lengths.shape != (samples.shape[0],):
            raise BackboneInputError("valid_lengths must have shape [B]")
        if valid_lengths.dtype not in (torch.int32, torch.int64):
            raise BackboneInputError("valid_lengths must be integer sample counts")
        lengths = valid_lengths.detach().cpu().tolist()
        if any(n <= 0 or n > samples.shape[1] for n in lengths):
            raise BackboneInputError("valid_lengths must be in [1,S]")
        for i, n in enumerate(lengths):
            if (samples[i, :n].abs() > 1).any():
                raise BackboneInputError("valid PCM samples must be in [-1,1]; decode integer PCM upstream")
        model = self._load_model()
        if min(lengths) < self.min_samples:
            raise BackboneInputError(f"audio needs at least {self.min_samples} samples for one encoder frame")
        model.eval()
        model.requires_grad_(False)
        device = next(model.parameters()).device
        rows = [None] * len(lengths)
        # no_grad rather than inference_mode permits downstream backward.
        with torch.inference_mode(False), torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            for n in sorted(set(lengths)):
                indices = [i for i, length in enumerate(lengths) if length == n]
                source = samples[indices, :n].detach().to(device=device, dtype=torch.float32).contiguous()
                if self.normalize:
                    source = torch.nn.functional.layer_norm(source, (n,))
                result = model.extract_features(source=source, padding_mask=None, mask=False, layer=self.config.output_layer)
                features = result["x"].float()
                expected = n
                for kernel, stride in self.conv_geometry:
                    expected = (expected - kernel) // stride + 1
                if features.shape != (len(indices), expected, self.embedding_dim) or not torch.isfinite(features).all():
                    raise BackboneLoadError("encoder returned invalid shape or nonfinite features")
                for j, i in enumerate(indices):
                    rows[i] = features[j]
            output_lengths = torch.tensor([row.shape[0] for row in rows], device=device, dtype=torch.int64)
            output = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True)
            padding = torch.arange(output.shape[1], device=device)[None, :] >= output_lengths[:, None]
        return EncoderBatch(output, output_lengths, padding, self.frame_hop_ms, self.backbone_id,
                            self.checkpoint_sha256, self.config.output_layer)

    def extract(self, samples: np.ndarray, metadata: ChunkMetadata) -> EmbeddingSequence:
        """Compatibility bridge for the existing single-chunk router contract."""
        import torch
        original = np.asarray(samples)
        if not np.issubdtype(original.dtype, np.floating):
            raise BackboneInputError("samples must be floating PCM")
        waveform = self.validate_samples(original, metadata)
        if not math.isfinite(metadata.duration_ms) or not math.isclose(
            metadata.duration_ms, waveform.size / 16, abs_tol=1 / 16, rel_tol=0
        ):
            raise BackboneInputError("duration_ms must match the sample count at 16 kHz")
        batch = self.extract_batch(torch.from_numpy(waveform.copy())[None], torch.tensor([waveform.size]),
                                   sample_rate=metadata.sample_rate)
        return EmbeddingSequence(batch.features[0].cpu().numpy(), batch.frame_hop_ms,
                                 self.backbone_id, "indic", metadata, batch.checkpoint_version)
