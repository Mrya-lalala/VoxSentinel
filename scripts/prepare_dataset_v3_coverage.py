"""CLI for the v3 male-genuine coverage acquisition (see src/dataset_prep/coverage_v3.py).

    ./.venv-data/bin/python -m scripts.prepare_dataset_v3_coverage snapshot     # frozen hashes before mutation
    ./.venv-data/bin/python -m scripts.prepare_dataset_v3_coverage plan         # metadata-only feasibility + selection
    ./.venv-data/bin/python -m scripts.prepare_dataset_v3_coverage materialize  # bounded fetches + prep-2 windows
    ./.venv-data/bin/python -m scripts.prepare_dataset_v3_coverage finalize     # combined manifests + version
    ./.venv-data/bin/python -m scripts.prepare_dataset_v3_coverage replay       # offline recomputation check
    ./.venv-data/bin/python -m scripts.prepare_dataset_v3_coverage verify       # frozen-artifact preservation check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.dataset_prep import coverage_v3  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("snapshot", help="hash frozen artifacts before any mutation")
    sub.add_parser("plan", help="metadata-only feasibility + frozen selection")
    materialize = sub.add_parser("materialize", help="bounded fetches + prep-2 windows")
    materialize.add_argument("--languages", nargs="*", default=None)
    sub.add_parser("finalize", help="combined manifests + dataset-version.v3.json")
    sub.add_parser("replay", help="offline recomputation + manifest comparison")
    sub.add_parser("verify", help="frozen-artifact preservation check")
    args = parser.parse_args(argv)

    if args.command == "snapshot":
        coverage_v3.command_snapshot()
    elif args.command == "plan":
        coverage_v3.command_plan()
    elif args.command == "materialize":
        coverage_v3.command_materialize(languages=args.languages)
    elif args.command == "finalize":
        coverage_v3.command_finalize()
    elif args.command == "replay":
        coverage_v3.command_replay()
    elif args.command == "verify":
        coverage_v3.command_verify_snapshot()
    else:  # pragma: no cover
        parser.error(f"unknown command {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
