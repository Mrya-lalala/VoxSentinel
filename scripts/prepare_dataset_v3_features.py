"""Build / check / parity-verify the v3 coverage feature caches.

Runs under the pinned encoder environment (``.venv``):

    ./.venv/bin/python -m scripts.prepare_dataset_v3_features check
    ./.venv/bin/python -m scripts.prepare_dataset_v3_features build
    ./.venv/bin/python -m scripts.prepare_dataset_v3_features parity

``check`` validates the emitted bundles and sidecars against
``dataset-version.v3.json`` without touching the network or the ledger.
``build`` reuses the historical v1 cache per example where identities match
and extracts everything else with the pinned frozen encoder.  ``parity``
re-extracts a small predeclared sample directly through the encoder and
compares it to the cache (encoder parity, not classifier evaluation).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VERSION_DEFAULT = "artifacts/datasets-v3-coverage/manifests/dataset-version.v3.json"
FEATURES_DEFAULT = "artifacts/datasets-v3-coverage/features"
REUSE_DEFAULT = "artifacts/datasets"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check", "parity"))
    parser.add_argument("--dataset-version", default=VERSION_DEFAULT)
    parser.add_argument("--features-dir", default=FEATURES_DEFAULT)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--tolerance", type=float, default=1e-5,
                        help="parity tolerance for the max absolute feature error (reused items)")
    parser.add_argument("--reuse-root", default=REUSE_DEFAULT)
    parser.add_argument("--no-reuse", action="store_true", help="extract everything with the pinned encoder")
    parser.add_argument("--force", action="store_true", help="rebuild even when the caches are already current")
    parser.add_argument("--no-audio-hashes", action="store_true", help="skip prepared-audio re-hashing in check")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    from src.dataset_prep.features_v3 import (
        FeatureCacheError,
        audit_v3_cache,
        build_cache,
        run_parity,
    )

    try:
        if args.command == "build":
            report = build_cache(
                args.dataset_version,
                args.features_dir,
                repo_root=args.repo_root,
                reuse_root=None if args.no_reuse else args.reuse_root,
                batch_size=args.batch_size,
                device=args.device,
                threads=args.threads,
                force=args.force,
            )
            print(json.dumps({
                "status": report["status"],
                "splits": {split: {"counts": entry["counts"], "bundle_sha256": entry["bundle_sha256"]}
                           for split, entry in report["splits"].items()} if "splits" in report else {},
                "reuse": report.get("reuse"),
                "seconds": report.get("seconds"),
                "audit_ready": (report.get("audit") or {}).get("ready"),
            }, indent=2))
            return 0

        if args.command == "check":
            report = audit_v3_cache(
                args.dataset_version,
                args.features_dir,
                repo_root=args.repo_root,
                check_audio_hashes=not args.no_audio_hashes,
            )
            out = Path(args.json_out) if args.json_out else Path(args.features_dir) / "reports" / "cache_check.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps({"ready": report["ready"], "n_failures": len(report["failures"]),
                              "failures": report["failures"][:8],
                              "splits": {split: {"counts": entry.get("counts")}
                                         for split, entry in report["splits"].items()},
                              "report": str(out)}, indent=2))
            return 0 if report["ready"] else 1

        report = run_parity(
            args.dataset_version,
            args.features_dir,
            repo_root=args.repo_root,
            tolerance=args.tolerance,
            device=args.device,
        )
        print(json.dumps({
            "passed": report["passed"],
            "max_abs_error": report["max_abs_error"],
            "tolerance": report["tolerance"],
            "languages_covered": report["languages_covered"],
            "samples": [{key: sample[key] for key in ("reason", "split", "window_id", "language", "num_frames", "max_abs_error", "autograd_ok")}
                        for sample in report["samples"]],
            "report": str(Path(args.features_dir) / "reports" / "parity.json"),
        }, indent=2))
        return 0 if report["passed"] else 1
    except FeatureCacheError as error:
        print(f"FAILED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
