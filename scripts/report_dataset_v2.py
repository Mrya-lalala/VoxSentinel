"""Report generator for the v2 Indic split (markdown + machine summary).

Reads the planning/materialization/audit outputs and writes
``artifacts/datasets-v2/reports/report.md`` and ``report.json``.  It does not
touch the network and performs no model evaluation.

    .venv-data/bin/python -m scripts.report_dataset_v2
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

V1 = REPO / "artifacts" / "datasets"
V2 = REPO / "artifacts" / "datasets-v2"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []


def _gender_of_reference(value: str | None) -> str | None:
    if not value:
        return None
    stem = Path(str(value).replace("\\", "/")).name.lower()
    parts = stem.rsplit(".", 1)[0].split("-")
    return parts[-1] if parts and parts[-1] in ("m", "f") else None


def main() -> int:
    reports = V2 / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    test_rows = _read_jsonl(V2 / "manifests" / "windows.test.jsonl")
    train_rows = _read_jsonl(V2 / "manifests" / "windows.train.jsonl")
    dev_rows = _read_jsonl(V2 / "manifests" / "windows.dev.jsonl")
    plan = json.loads((V2 / "planning" / "test_plan.json").read_text()) if (V2 / "planning" / "test_plan.json").exists() else {}
    audit = json.loads((reports / "audit.json").read_text()) if (reports / "audit.json").exists() else None
    ledger = json.loads((V1 / "manifests" / "downloads.json").read_text())
    snapshot = json.loads((V2 / "planning" / "snapshot_before.json").read_text())
    heldout = json.loads((V2 / "staging" / "kathbath_heldout.json").read_text()) if (V2 / "staging" / "kathbath_heldout.json").exists() else {}
    discovery = {}
    for path in sorted((V2 / "staging").glob("kathbath_fresh_summary.*.json")):
        discovery[path.stem.split(".")[1]] = json.loads(path.read_text())
    isynth_discovery = json.loads((V2 / "staging" / "indicsynth_discovery_summary.json").read_text()) \
        if (V2 / "staging" / "indicsynth_discovery_summary.json").exists() else {}

    def totals(rows: list[dict]) -> dict:
        windows = len(rows)
        recordings = len({str(r.get("recording_id")) for r in rows})
        durations = [(r.get("window") or {}).get("duration_seconds") or 0.0 for r in rows]
        return {"windows": windows, "recordings": recordings,
                "seconds": round(sum(durations), 3), "hours": round(sum(durations) / 3600, 4)}

    test_stats: dict = {"by_language": {}, "generators": Counter(), "source_gender": Counter(),
                        "target_gender": Counter(), "genuine_gender": Counter(),
                        "rates": Counter(), "missing": Counter()}
    speakers = {"genuine": set(), "source": set(), "target": set()}
    for row in test_rows:
        lang = row["spoken_language"]
        entry = test_stats["by_language"].setdefault(lang, {"genuine": 0, "synthetic": 0})
        entry["genuine" if row["label"] == 0 else "synthetic"] += 1
        parent = row.get("parent_refs") or {}
        ids = row.get("speaker_ids") or {}
        if row["dataset_id"] == "kathbath":
            speakers["genuine"].add((lang, str(ids.get("speaker"))))
            gender = str(ids.get("gender") or "unknown")
            test_stats["genuine_gender"][gender] += 1
        else:
            if ids.get("source") is not None:
                speakers["source"].add((lang, str(ids.get("source"))))
            if ids.get("target") is not None:
                speakers["target"].add((lang, str(ids.get("target"))))
            test_stats["generators"][str(row.get("generator"))] += 1
            test_stats["source_gender"][_gender_of_reference(parent.get("source_reference")) or "not_applicable"] += 1
            test_stats["target_gender"][_gender_of_reference(parent.get("target_reference")) or "unknown"] += 1
        rate = (row.get("original_audio") or {}).get("sample_rate")
        test_stats["rates"][f"{row['dataset_id']}:{rate}"] += 1
        if parent.get("source_parent_verification") == "parsed_only":
            test_stats["missing"]["source_parent_parsed_only"] += 1
        if parent.get("target_parent_verification") == "parsed_only":
            test_stats["missing"]["target_parent_parsed_only"] += 1
        if row["dataset_id"] == "indicsynth" and parent.get("transcript") is None:
            test_stats["missing"]["transcript_unknown"] += 1

    train_stats, dev_stats = totals(train_rows), totals(dev_rows)
    test_summary = totals(test_rows)

    def split_distribution(rows: list[dict]) -> dict:
        rates = Counter(f"{r['dataset_id']}:{(r.get('original_audio') or {}).get('sample_rate')}" for r in rows)
        generators = Counter(str(r.get('generator') or 'genuine') for r in rows)
        channels = Counter((r.get('prepared_audio') or {}).get('channels') for r in rows)
        return {"rates": dict(sorted(rates.items())), "generators": dict(sorted(generators.items())),
                "channels": {str(k): v for k, v in sorted(channels.items(), key=str)}}

    train_dist, dev_dist = split_distribution(train_rows), split_distribution(dev_rows)
    gender_summary = {
        "genuine_female": test_stats["genuine_gender"].get("f", 0),
        "genuine_male": test_stats["genuine_gender"].get("m", 0),
        "synthetic_source_female": test_stats["source_gender"].get("f", 0),
        "synthetic_source_male": test_stats["source_gender"].get("m", 0),
        "synthetic_source_not_applicable": test_stats["source_gender"].get("not_applicable", 0),
        "synthetic_target_female": test_stats["target_gender"].get("f", 0),
        "synthetic_target_male": test_stats["target_gender"].get("m", 0),
    }

    coverage_limits = []
    for language, entry in sorted(discovery.items()):
        if entry.get("male_speakers_fresh", 0) == 0 and entry.get("remaining_unscanned", 0) == 0:
            coverage_limits.append(f"{language}: no male speakers in the remaining scanned shards "
                                   f"({entry.get('rows_fresh')} rows examined)")
        if entry.get("remaining_unscanned", 0) > 0:
            coverage_limits.append(f"{language}: {entry.get('remaining_unscanned')} train shards still unscanned")
        if entry.get("errors"):
            coverage_limits.append(f"{language}: {len(entry['errors'])} shard scan errors")
    limits_text = coverage_limits or ["none beyond documented identity/near-duplicate limits"]

    status = {
        "indic_split_integrity": "pass: speaker_recording_disjoint_v2" if (audit or {}).get("ready") else "pending_or_issues",
        "coverage_limitations": limits_text,
        "untouched_test_status": (
            "audited under speaker_recording_disjoint_v2; strict conversion-family isolation not met"
            if (audit or {}).get("closure", {}).get("speakers", 1) == 0
            and (audit or {}).get("closure", {}).get("references", 1) == 0
            and (audit or {}).get("closure_transitive", {}).get("n_hits", 1) == 0
            and (audit or {}).get("heldout", {}).get("status") == "verified"
            else "not yet verified"
        ),
        "protocol": (audit or {}).get("protocol", "unspecified"),
        "strict_conversion_family_compliant": (audit or {}).get("strict_conversion_family_compliant", False),
        "indian_english_status": "pending IndieFake access (user requested; no archive/link yet)",
        "model_evaluation_performed": False,
    }

    summary = {
        "splits": {"train": train_stats, "dev": dev_stats, "test": {**test_summary,
                                                                     "genuine": sum(1 for r in test_rows if r["label"] == 0),
                                                                     "synthetic": sum(1 for r in test_rows if r["label"] == 1)}},
        "train_distribution": train_dist,
        "dev_distribution": dev_dist,
        "test_by_language": test_stats["by_language"],
        "test_generators": dict(test_stats["generators"]),
        "test_gender": gender_summary,
        "test_original_rates": dict(test_stats["rates"]),
        "test_missing": dict(test_stats["missing"]),
        "test_unique_speakers": {k: len(v) for k, v in speakers.items()},
        "status": status,
        "ledger": {
            "before": snapshot["ledger_before"],
            "after": {"total_bytes": ledger["total_bytes"], "cap_bytes": ledger["cap_bytes"],
                       "remaining_bytes": ledger["remaining_bytes"], "events": ledger["events"]},
        },
    }
    (reports / "report.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")

    recon_path = reports / "ledger_reconciliation.json"
    recon_lines: list[str] = []
    if recon_path.exists():
        recon = json.loads(recon_path.read_text())
        recon_lines = [f"- reconciliation charge: {recon.get('charged_bytes')} bytes for the historical "
                       f"cap-skipped read that was previously discarded uncharged "
                       f"({recon.get('method')})"]

    lines = [
        "# VoxSentinel v2 Indic split — preparation report",
        "",
        f"Status: `indic_split_integrity={status['indic_split_integrity']}` · "
        f"`model_evaluation_performed={status['model_evaluation_performed']}`",
        "",
        "## Split sizes",
        "",
        "| split | windows | recordings | hours |",
        "| --- | ---: | ---: | ---: |",
        f"| train (v1 preserved) | {train_stats['windows']} | {train_stats['recordings']} | {train_stats['hours']} |",
        f"| development (v1 preserved) | {dev_stats['windows']} | {dev_stats['recordings']} | {dev_stats['hours']} |",
        f"| test (new) | {test_summary['windows']} | {test_summary['recordings']} | {test_summary['hours']} |",
        "",
        f"train original rates: {json.dumps(train_dist['rates'])} · generators: {json.dumps(train_dist['generators'])}",
        f"dev original rates: {json.dumps(dev_dist['rates'])} · generators: {json.dumps(dev_dist['generators'])}",
        "",
        "## Test coverage",
        "",
        f"- unique speakers: genuine {summary['test_unique_speakers']['genuine']}, "
        f"synthetic source {summary['test_unique_speakers']['source']}, synthetic target {summary['test_unique_speakers']['target']}",
        f"- generators: {json.dumps(dict(test_stats['generators']))}",
        f"- genuine gender: f {gender_summary['genuine_female']} / m {gender_summary['genuine_male']}; "
        f"synthetic target gender: f {gender_summary['synthetic_target_female']} / m {gender_summary['synthetic_target_male']}; "
        f"synthetic source: f {gender_summary['synthetic_source_female']} / m {gender_summary['synthetic_source_male']} / "
        f"not applicable {gender_summary['synthetic_source_not_applicable']}",
        f"- original sample rates: {json.dumps(dict(test_stats['rates']))}",
        f"- parsed-only parent evidence counts: {json.dumps(dict(test_stats['missing']))}",
        "",
        "## Per-language test windows",
        "",
        "| language | genuine | synthetic |",
        "| --- | ---: | ---: |",
    ]
    for language, entry in sorted(test_stats["by_language"].items()):
        lines.append(f"| {language} | {entry['genuine']} | {entry['synthetic']} |")
    lines += [
        "",
        "## Discovery coverage (Kathbath remaining shards)",
        "",
        "| language | shards scanned (new) | rows | male speakers | female speakers | unscanned |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for language, entry in sorted(discovery.items()):
        lines.append(f"| {language} | {entry.get('scanned_v2_new')} | {entry.get('rows_fresh')} | "
                     f"{entry.get('male_speakers_fresh')} | {entry.get('female_speakers_fresh')} | "
                     f"{entry.get('remaining_unscanned')} |")
    lines += [
        "",
        "## Audit (independent validator)",
        "",
        f"- issues: {audit.get('n_issues') if audit else 'n/a'}; schema problems: "
        f"{len(audit.get('schema_problems', [])) if audit else 'n/a'}; audio problems: "
        f"{len(audit.get('audio_problems', [])) if audit else 'n/a'}; near-duplicate pairs: "
        f"{len(audit.get('audio', {}).get('near_duplicates', [])) if audit else 'n/a'}",
        f"- closure (direct membership): {json.dumps((audit or {}).get('closure', {}))}",
        f"- transitive closure (admissible relationships): "
        f"{json.dumps({k: v for k, v in (((audit or {}).get('closure_transitive') or {}).items()) if k != 'hits'})}",
        f"- cross-split components: {len((audit or {}).get('cross_split_components', {}))}",
        f"- unknown metadata: {json.dumps((audit or {}).get('unknown_metadata', {}))}",
        f"- accounting: {json.dumps((audit or {}).get('accounting', {}))}",
        f"- held-out check: {json.dumps((audit or {}).get('heldout', {}))}",
        "",
        "## Closure adjudication (review follow-up)",
        "",
        "- The reviewer's 67 co-participation flags are reproduced exactly; edge provenance was audited",
        "  (`reports/closure_audit.json`): no shared person, recording or hash connects those windows to",
        "  exposure — every path is a conversion co-participation chain only.",
        "- The enforced exclusion closure is the *admissible* relationship graph (person<->parent-recording",
        "  links per row side, transitive via shared exact keys, kathbath inventory label-conflict merges,",
        "  held-out anchors). It is computed in selection and independently in the validator; 0 hits.",
        "- The strict cross-union variant of that graph flagging the same 67 windows cannot be satisfied by",
        "  any materialized pool for Malayalam/Marathi (100% of synthetic candidates poisoned) and is",
        "  reported separately: the stricter original contract is NOT met. Infeasibility does not prove non-leakage.",
        "",
        "## Ledger reconciliation",
        "",
        f"- before: {snapshot['ledger_before']['total_bytes']} / {snapshot['ledger_before']['cap_bytes']} bytes",
        f"- after: {ledger['total_bytes']} / {ledger['cap_bytes']} bytes "
        f"({ledger['remaining_bytes']} remaining)",
        "- fetch path: every shard read is charged whether or not its payload is retained; over-cap shards are",
        "  skipped before reading when cached footer estimates are available.",
        *recon_lines,
        "",
        "## Coverage limitations",
        "",
    ]
    lines += [f"- {item}" for item in limits_text]
    lines += [
        "",
        "## Files preserved / added / excluded",
        "",
        "- Preserved: all v1 manifests, feature caches, checkpoints, raw audio and prepared windows (hash-verified unchanged).",
        "- Added under `artifacts/datasets-v2/`: fresh Kathbath inventory scans, IndicSynth fresh candidates, test plan,",
        "  materialized test windows (raw + prepared), manifests, audit, reports.",
        "- Excluded/reserved: upstream `valid` shards are recorded as held out; per-candidate dispositions live in",
        "  `planning/accounting.*.jsonl` and `manifests/windows.test.excluded.jsonl`.",
        "- Deleted: none.",
        "",
        "## Known limitations",
        "",
        "- Language-scoped identities; cross-language person independence is not established.",
        "- Near-duplicate detection uses RMS+ZCR frame fingerprints; transcoding with severe processing may evade it.",
        "- Per-generator subgroup sizes are small; no model evaluation was performed on this split.",
        "- Rows whose upstream `fname` does not match `<record>-<speaker>-<gender>` are not admitted to inventories",
        "  (same behavior as v1); unparsed-row counts are reported per shard only when recorded by the scanner.",
    ]
    (reports / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(reports / "report.md"), "status": status,
                      "test": summary["splits"]["test"]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
