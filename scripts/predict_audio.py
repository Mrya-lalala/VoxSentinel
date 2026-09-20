"""Predict local audio files using an explicit, hash-checked model release."""
import argparse
import json
from pathlib import Path
import sys

import torch

from src.scoring.predict import FilePredictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True)
    parser.add_argument("--encoder-path", default=None)
    parser.add_argument("--audio", required=True, nargs="+")
    parser.add_argument("--language", default=None)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    try:
        predictor = FilePredictor(args.release, encoder_path=args.encoder_path)
        results = [predictor.predict(path, language=args.language) for path in args.audio]
    except (ValueError, OSError, RuntimeError) as error:
        print(json.dumps({"status": "error", "error": str(error)}), file=sys.stderr)
        return 2
    text = json.dumps(results, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
