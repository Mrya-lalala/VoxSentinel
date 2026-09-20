"""CLI for v2 Indic split preparation (``docs/DATA_SPLIT_AGENT_MASTER_PROMPT.md``).

Commands are resumable and bounded; every network read is revision-pinned and
ledger-charged.  Nothing here trains, extracts features or evaluates models.

    .venv-data/bin/python -m scripts.prepare_dataset_v2 registry
    .venv-data/bin/python -m scripts.prepare_dataset_v2 discover-kathbath --languages Bengali ...
    .venv-data/bin/python -m scripts.prepare_dataset_v2 discover-heldout
    .venv-data/bin/python -m scripts.prepare_dataset_v2 discover-indicsynth
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.dataset_prep import split_v2 as sv2  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("registry", help="extend the prior-exposure registry from local artifacts")

    kb = sub.add_parser("discover-kathbath", help="scan unscanned Kathbath shards (revision-pinned, charged)")
    kb.add_argument("--languages", nargs="*", default=sv2.CORE_LANGUAGES)
    kb.add_argument("--max-shards", type=int, default=14, help="max new shards per language this run")
    kb.add_argument("--min-male-speakers", type=int, default=6)
    kb.add_argument("--min-fresh-speakers", type=int, default=12)
    kb.add_argument("--batch", type=int, default=4)
    kb.add_argument("--workers", type=int, default=6)

    ho = sub.add_parser("discover-heldout", help="scan upstream valid/test shards for held-out record ids")
    ho.add_argument("--languages", nargs="*", default=sv2.CORE_LANGUAGES)
    ho.add_argument("--workers", type=int, default=6)

    sc = sub.add_parser("scan-counts", help="footer-only per-shard row counts for unparsed-row accounting")
    sc.add_argument("--languages", nargs="*", default=sv2.CORE_LANGUAGES)
    sc.add_argument("--workers", type=int, default=6)

    isy = sub.add_parser("discover-indicsynth", help="scan IndicSynth rows for fresh synthetic candidates")
    isy.add_argument("--languages", nargs="*", default=sv2.CORE_LANGUAGES)
    isy.add_argument("--strata", type=int, default=24)
    isy.add_argument("--page", type=int, default=100)

    pl = sub.add_parser("plan", help="deterministic component-aware test selection from frozen inputs")
    pl.add_argument("--languages", nargs="*", default=sv2.CORE_LANGUAGES)
    pl.add_argument("--target", type=int, default=sv2.TEST_TARGET_PER_CLASS)
    pl.add_argument("--seed", type=int, default=sv2.SEED_DEFAULT)

    mt = sub.add_parser("materialize", help="fetch + prepare the selected test windows")
    mt.add_argument("--languages", nargs="*", default=None)
    mt.add_argument("--byte-cap-per-language", type=int, default=64_000_000)

    sub.add_parser("finalize", help="copy v1 train/dev manifests and write dataset-version.v2.json")

    rp = sub.add_parser("replay", help="recompute selection from frozen inputs and compare")
    rp.add_argument("--languages", nargs="*", default=sv2.CORE_LANGUAGES)
    rp.add_argument("--target", type=int, default=sv2.TEST_TARGET_PER_CLASS)
    rp.add_argument("--seed", type=int, default=sv2.SEED_DEFAULT)

    args = parser.parse_args(argv)

    if args.command == "registry":
        report = sv2.command_registry()
        print(json.dumps(report["counts"], indent=2))
        return 0
    if args.command == "discover-kathbath":
        sv2.command_discover_kathbath(
            languages=args.languages, max_shards_per_language=args.max_shards,
            min_male_speakers=args.min_male_speakers, min_fresh_speakers=args.min_fresh_speakers,
            batch=args.batch, workers=args.workers,
        )
        return 0
    if args.command == "discover-heldout":
        sv2.command_discover_heldout(languages=args.languages, workers=args.workers)
        return 0
    if args.command == "scan-counts":
        sv2.command_scan_counts(languages=args.languages, workers=args.workers)
        return 0
    if args.command == "discover-indicsynth":
        sv2.command_discover_indicsynth(languages=args.languages, strata=args.strata, page=args.page)
        return 0
    if args.command == "plan":
        sv2.command_plan(languages=args.languages, target_per_class=args.target, seed=args.seed)
        return 0
    if args.command == "materialize":
        sv2.command_materialize(languages=args.languages, byte_cap_per_language=args.byte_cap_per_language)
        return 0
    if args.command == "finalize":
        sv2.command_finalize()
        return 0
    if args.command == "replay":
        sv2.command_replay(languages=args.languages, target_per_class=args.target, seed=args.seed)
        return 0
    raise SystemExit(f"unknown command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
