"""Command-line stages for the bounded dataset-preparation pilot.

Stages are separate so that acquisition (data environment), splitting
(data environment), feature caching (pinned encoder environment) and reporting
run in the correct interpreter:

    python -m scripts.prepare_datasets catalog
    python -m scripts.prepare_datasets fetch --source indicsynth
    python -m scripts.prepare_datasets fetch --source all
    python -m scripts.prepare_datasets split
    .venv/bin/python -m scripts.prepare_datasets features
    python -m scripts.prepare_datasets report

Every invocation is recorded in ``manifests/run-log.jsonl`` for the handoff.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .budget import BudgetExceeded, DownloadLedger
from .config import DEFAULT_CONFIG_PATH, PrepConfig, load_prep_config
from .records import append_jsonl, read_json, read_jsonl, write_json_atomic

LEDGER_NAME = "downloads.json"
RUN_LOG_NAME = "run-log.jsonl"


def _ledger(cfg: PrepConfig) -> DownloadLedger:
    cfg.ensure_dirs()
    return DownloadLedger.load(cfg.manifests_dir / LEDGER_NAME, cfg.budget.max_download_bytes)


def _log(cfg: PrepConfig, stage: str, argv: Sequence[str], *, notes: Sequence[str] | None = None, summary: Any = None) -> None:
    append_jsonl(
        cfg.manifests_dir / RUN_LOG_NAME,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "argv": list(argv),
            "notes": list(notes or []),
            "summary": summary,
        },
    )


def command_catalog(cfg: PrepConfig, argv: Sequence[str]) -> int:
    from .adapters.catalog import source_catalog

    entries = [entry.to_dict() for entry in source_catalog()]
    write_json_atomic(
        cfg.manifests_dir / "source-catalog.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sources": entries,
        },
    )
    _log(cfg, "catalog", argv, summary={"sources": len(entries)})
    print(f"wrote source catalog with {len(entries)} sources")
    for entry in entries:
        print(f"  {entry['source_id']:20s} {entry['status']}")
    return 0


def command_fetch(cfg: PrepConfig, argv: Sequence[str], args: argparse.Namespace) -> int:
    from .adapters import FETCHERS

    ledger = _ledger(cfg)
    selected = list(FETCHERS) if args.source == "all" else [args.source]
    exit_code = 0
    for source_id in selected:
        if not cfg.source_enabled(source_id):
            print(f"== fetch {source_id}: disabled in the pilot configuration; skipped ==")
            _log(cfg, "fetch", argv, notes=[f"{source_id}: disabled"], summary={"source": source_id})
            continue
        fetcher = FETCHERS[source_id]
        kwargs: dict[str, Any] = {"force": args.force}
        if source_id == "indicsynth" and args.languages:
            kwargs["languages"] = args.languages
        if source_id == "nisp" and args.groups:
            kwargs["groups"] = args.groups
        if source_id == "synthetic_english":
            kwargs["import_dir"] = args.import_dir
            kwargs["metadata_file"] = args.metadata_file
        print(f"== fetch {source_id} ==", flush=True)
        try:
            run = fetcher(cfg, ledger, **kwargs)
        except BudgetExceeded as error:
            print(f"  stopped: {error}")
            _log(cfg, "fetch", argv, notes=[f"{source_id}: {error}"], summary={"source": source_id})
            exit_code = 1
            continue
        print(
            f"  windows={len(run.windows)} recordings={len(run.recordings)} "
            f"exclusions={len(run.exclusions)} new_bytes={run.download_bytes}"
        )
        for note in run.notes:
            print(f"  note: {note}")
        _log(
            cfg,
            "fetch",
            argv,
            notes=run.notes,
            summary={
                "source": source_id,
                "windows": len(run.windows),
                "recordings": len(run.recordings),
                "exclusions": len(run.exclusions),
                "download_bytes": run.download_bytes,
            },
        )
    return exit_code


def command_core(cfg: PrepConfig, argv: Sequence[str], args: argparse.Namespace) -> int:
    from .joint_core import build_core

    ledger = _ledger(cfg)
    summary = build_core(
        cfg,
        ledger,
        languages=args.languages,
        workers=args.workers,
        force_scan=args.force_scan,
        plan_only=args.plan_only,
    )
    _log(cfg, "core", argv, notes=summary.get("notes"), summary=summary["totals"])
    print(json.dumps({"totals": summary["totals"], "download_bytes": ledger.total_bytes}, indent=2, sort_keys=True))
    print("staged under artifacts/datasets/staging/core; run `promote` after review.")
    return 0


def command_promote(cfg: PrepConfig, argv: Sequence[str]) -> int:
    from .assemble import assemble_pools
    from .joint_core import promote_core

    result = promote_core(cfg)
    print(json.dumps(result, indent=2, sort_keys=True))
    counters = assemble_pools(cfg)
    _log(cfg, "promote", argv, summary={**result, "pools": counters["pools"]})
    print("pools after promote:", json.dumps(counters["pools"], indent=2, sort_keys=True))
    return 0


def command_split(cfg: PrepConfig, argv: Sequence[str]) -> int:
    from .assemble import assemble_pools

    counters = assemble_pools(cfg)
    _log(cfg, "split", argv, summary=counters)
    print(json.dumps(counters, indent=2, sort_keys=True))
    return 0


def command_features(cfg: PrepConfig, argv: Sequence[str], args: argparse.Namespace) -> int:
    from .features import build_cache

    manifest_map = {
        "train": cfg.manifests_dir / "windows.core_train.jsonl",
        "val": cfg.manifests_dir / "windows.core_val.jsonl",
    }
    missing = [name for name, path in manifest_map.items() if not path.exists()]
    if missing:
        raise SystemExit(f"run the split stage first; missing manifests: {missing}")
    summary = build_cache(
        dataset_root=cfg.dataset_root,
        split_manifests=manifest_map,
        backbone_config=args.backbone_config,
        features_config={
            **cfg.features,
            "preprocessing_version": cfg.preprocessing.get("config_version"),
        },
        batch_size=int(cfg.features.get("batch_size", 8)),
        device=str(args.device or cfg.features.get("device", "cpu")),
        force=args.force,
        threads=args.threads,
    )
    _log(cfg, "features", argv, summary=summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def command_report(cfg: PrepConfig, argv: Sequence[str]) -> int:
    from .report import write_reports

    report = write_reports(cfg)
    _log(cfg, "report", argv, summary={"pools": report["pools"]})
    print(f"wrote {cfg.reports_dir / 'preparation-report.json'}")
    print(f"wrote {cfg.reports_dir / 'preparation-report.md'}")
    return 0


def command_status(cfg: PrepConfig, argv: Sequence[str]) -> int:
    ledger = read_json(cfg.manifests_dir / LEDGER_NAME, default={}) or {}
    print(f"dataset root: {cfg.dataset_root}")
    print(f"download bytes: {ledger.get('total_bytes', 0)} / cap {ledger.get('cap_bytes', 0)}")
    for source_file in sorted(cfg.manifests_dir.glob("source_windows.*.jsonl")):
        rows = read_jsonl(source_file)
        print(f"  {source_file.name}: {len(rows)} windows")
    for name in (
        "windows.core_train.jsonl",
        "windows.core_val.jsonl",
        "windows.supplementary.jsonl",
        "windows.external_eval.jsonl",
        "windows.fallback_baseline.jsonl",
        "windows.unpaired_candidate.jsonl",
        "windows.excluded.jsonl",
        "windows.missing_coverage.jsonl",
    ):
        path = cfg.manifests_dir / name
        if path.exists():
            print(f"  {name}: {len(read_jsonl(path))} rows")
    for meta in sorted(cfg.features_dir.glob("*.meta.json")):
        print(f"  features: {meta.name}")
    _log(cfg, "status", argv)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.prepare_datasets",
        description="Bounded dataset-preparation pilot (no training, no evaluation).",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="pilot configuration YAML")
    parser.add_argument("--dataset-root", default=None, help="override dataset root directory")
    sub = parser.add_subparsers(dest="stage", required=True)

    sub.add_parser("catalog", help="write the source-access catalog")
    sub.add_parser("status", help="print a quick inventory of manifests")

    core = sub.add_parser("core", help="build the joint Kathbath+IndicSynth core pool into staging")
    core.add_argument("--languages", nargs="*", default=None, help="subset of the 12 core languages")
    core.add_argument("--workers", type=int, default=6, help="parallel Hub metadata/file reads")
    core.add_argument("--force-scan", action="store_true", help="re-scan IndicSynth candidates")
    core.add_argument("--plan-only", action="store_true", help="plan only; do not fetch audio")

    sub.add_parser("promote", help="validate staged core artifacts, swap them in, reassemble pools")

    fetch = sub.add_parser("fetch", help="inventory/acquisition/materialization per source")
    fetch.add_argument(
        "--source",
        default="all",
        choices=["all", "indicsynth", "nisp", "asvspoof2019", "nptel", "svarah", "synthetic_english"],
    )
    fetch.add_argument("--languages", nargs="*", default=None, help="IndicSynth language configs to process")
    fetch.add_argument("--groups", nargs="*", default=None, help="NISP native-language groups to process")
    fetch.add_argument("--force", action="store_true", help="ignore previous source manifests")
    fetch.add_argument("--import-dir", default=None, help="synthetic_english: local asset directory")
    fetch.add_argument("--metadata-file", default=None, help="synthetic_english: metadata JSONL path")

    sub.add_parser("split", help="assemble pool manifests + coverage/gap inventory")

    features = sub.add_parser("features", help="extract frozen encoder features (encoder environment)")
    features.add_argument("--backbone-config", default="configs/backbones.yaml")
    features.add_argument("--device", default=None)
    features.add_argument("--threads", type=int, default=None)
    features.add_argument("--force", action="store_true", help="rebuild even if the cache identity matches")

    sub.add_parser("report", help="write reports/preparation-report.{json,md}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_prep_config(args.config, dataset_root=args.dataset_root)
    cfg.ensure_dirs()
    effective_argv = list(sys.argv[1:] if argv is None else argv)

    if args.stage == "catalog":
        return command_catalog(cfg, effective_argv)
    if args.stage == "status":
        return command_status(cfg, effective_argv)
    if args.stage == "fetch":
        return command_fetch(cfg, effective_argv, args)
    if args.stage == "core":
        return command_core(cfg, effective_argv, args)
    if args.stage == "promote":
        return command_promote(cfg, effective_argv)
    if args.stage == "split":
        return command_split(cfg, effective_argv)
    if args.stage == "features":
        return command_features(cfg, effective_argv, args)
    if args.stage == "report":
        return command_report(cfg, effective_argv)
    raise SystemExit(f"unknown stage {args.stage}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
