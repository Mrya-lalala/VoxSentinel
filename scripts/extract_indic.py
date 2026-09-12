"""Run from repo root: python -m scripts.extract_indic --audio clip.wav."""
import argparse
import json
from pathlib import Path
import time

import soundfile as sf
import torch
import yaml
from src.backbones.indic_wav2vec import IndicWav2VecConfig, IndicWav2VecExtractor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', required=True, nargs='+', help='Local mono 16 kHz files; no automatic resampling')
    parser.add_argument('--config', default='configs/backbones.yaml')
    parser.add_argument('--output-layer', type=int, help='Zero-based transformer block (0..23 for Large); default final output')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', help='Optional local .pt tensor output')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    config = yaml.safe_load(Path(args.config).read_text())['indic_wav2vec']
    if args.output_layer is not None:
        config['output_layer'] = args.output_layer
    rows = []
    for path in args.audio:
        audio, rate = sf.read(path, dtype='float32')
        if rate != 16000 or audio.ndim != 1:
            parser.error(f'{path}: expected mono 16 kHz audio; decoding/resampling belongs upstream')
        rows.append(torch.from_numpy(audio))
    start = time.perf_counter()
    encoder = IndicWav2VecExtractor(IndicWav2VecConfig(**config)).load()
    load_s = time.perf_counter() - start
    lengths = torch.tensor([row.numel() for row in rows])
    start = time.perf_counter()
    result = encoder.extract_batch(torch.nn.utils.rnn.pad_sequence(rows, batch_first=True), lengths)
    elapsed = time.perf_counter() - start
    print(json.dumps({'shape': list(result.features.shape), 'valid_lengths': result.valid_lengths.tolist(),
        'padding_mask': 'True = ignore; False = valid', 'frame_hop_ms': result.frame_hop_ms,
        'checkpoint_sha256': result.checkpoint_version, 'output_layer': result.output_layer,
        'normalize': encoder.normalize, 'load_seconds': load_s, 'extraction_seconds': elapsed,
        'device': str(result.features.device), 'threads': torch.get_num_threads()}, indent=2))
    if args.output:
        torch.save({'features': result.features.cpu(), 'valid_lengths': result.valid_lengths.cpu(),
                    'padding_mask': result.padding_mask.cpu(), 'checkpoint_sha256': result.checkpoint_version,
                    'frame_hop_ms': result.frame_hop_ms, 'output_layer': result.output_layer}, args.output)


if __name__ == '__main__':
    main()
