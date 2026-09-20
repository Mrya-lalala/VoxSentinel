"""Independent validator for the v3 male-genuine coverage manifests.

Consumes the emitted manifests and planning evidence; does NOT import the
selector (``src/dataset_prep/coverage_v3.py``).  Reuses the generic audited
primitives of the v2 validator (reference parsing, provenance reconstruction,
closure audit, fingerprints) while adding the coverage-specific checks:

1. retention: every retained v2 train/dev row is present and unchanged;
2. all-pair cross-role identity/reference checks across retained splits, the
   new additions and the evaluated benchmark — benchmark exclusion ENFORCED;
3. original/prepared hash equalities and near-duplicate fingerprints;
4. file existence, decode contract, prep-2 window bounds, stored hashes;
5. documented male gender from source metadata (filename convention);
6. deterministic selection replay: zero-charge replan reproduces the frozen
   plan projection (point-in-time accounting fields excluded, identity
   verified) -> identical assignments + manifest hashes;
7. budget retention and frozen-artifact preservation.

    ./.venv-data/bin/python -m scripts.validate_dataset_v3_coverage --fixtures
    ./.venv-data/bin/python -m scripts.validate_dataset_v3_coverage
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.validate_dataset_v2 import (  # noqa: E402
    NEAR_DUP_DURATION_TOL,
    NEAR_DUP_SIM,
    PREP_VERSION,
    _fingerprint,
    _numeric,
    _parse_reference,
    _sha256,
    audit_closure,
    load_closure_data,
    row_identities,
)
from src.dataset_prep.budget import BudgetExceeded  # noqa: E402
from src.dataset_prep.bounded_reader import BoundedReader  # noqa: E402

V1 = REPO / "artifacts" / "datasets"
V2 = REPO / "artifacts" / "datasets-v2"
V3 = REPO / "artifacts" / "datasets-v3-coverage"
BENCHMARK = REPO / "artifacts" / "evaluations" / "gru-v2-test-epoch5"
SCHEMA = "voxsentinel.v3_coverage.audit.v1"
REV_KB = "5b9e92849222026d9141acba4e8434fe816396bf"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []


# --------------------------------------------------------------------------- #
# key sets
# --------------------------------------------------------------------------- #

def benchmark_keys() -> dict[str, set]:
    exposure = json.loads((BENCHMARK / "exposure.json").read_text())
    test_rows = _rows(V2 / "manifests" / "windows.test.jsonl")
    speakers, references, records = set(), set(), set()
    for _dataset, lang, speaker in exposure.get("speaker_keys", []):
        speakers.add((str(lang).lower(), str(int(float(str(speaker))))))
    for _dataset, lang, canonical in exposure.get("reference_keys", []):
        lang = str(lang).lower()
        references.add((lang, str(canonical)))
        records.add((lang, str(canonical).split("-")[0]))
    for row in test_rows:
        ids = row_identities(row)
        lang = ids["language"]
        for key in ids["speakers"].values():
            speakers.add((lang, key[2]))
        for key in ids["references"].values():
            references.add((lang, key[2]))
        for key in ids["records"]:
            records.add((lang, key[2]))
    hashes = {str(h) for h in exposure.get("prepared_audio_sha256", [])}
    hashes |= {str((r.get("prepared_audio") or {}).get("sha256")) for r in test_rows}
    return {"speakers": speakers, "references": references, "records": records, "hashes": hashes}


def v1_keys() -> dict[str, set]:
    registry = json.loads((V2 / "planning" / "exposure_registry.v2.json").read_text())
    speakers, references, records = set(), set(), set()
    for _dataset, lang, speaker in registry.get("speaker_keys", []):
        speakers.add((str(lang).lower(), str(int(float(str(speaker))))))
    for _dataset, lang, canonical in registry.get("reference_keys", []):
        lang = str(lang).lower()
        references.add((lang, str(canonical)))
        records.add((lang, str(canonical).split("-")[0]))
    return {"speakers": speakers, "references": references, "records": records}


def addition_keys(row: dict) -> dict[str, set]:
    """Identity keys of one added row reconstructed from its provenance."""
    lang = str(row.get("spoken_language", "")).lower()
    ids = row.get("speaker_ids") or {}
    speaker = _numeric(ids.get("speaker"))
    parsed = _parse_reference(Path(str(row.get("source_file", ""))).name)
    out = {"speakers": set(), "references": set(), "records": set()}
    if speaker is not None:
        out["speakers"].add((lang, speaker))
    if parsed is not None:
        out["references"].add((lang, parsed["canonical"]))
        out["records"].add((lang, parsed["record"]))
    return out


# --------------------------------------------------------------------------- #
# core audit (pure; fixture-testable)
# --------------------------------------------------------------------------- #

def audit_coverage(additions: list[dict], retained: list[dict], benchmark: dict, v1: dict,
                   closure_edges: Sequence = (), closure_anchors: Sequence = (),
                   v2_test_rows: list[dict] | None = None) -> dict:
    issues: list[dict] = []

    # -- per-addition structural + exclusion checks --------------------------
    seen_ids: Counter = Counter()
    speaker_split: dict[tuple[str, str], str] = {}
    for row in additions:
        wid = str(row.get("window_id"))
        seen_ids[wid] += 1
        lang = str(row.get("spoken_language", "")).lower()
        keys = addition_keys(row)
        if str(row.get("path_base")) != "artifacts/datasets-v3-coverage":
            issues.append({"check": "addition_wrong_path_base", "window_id": wid,
                           "path_base": row.get("path_base")})
        for (lang_key, speaker) in keys["speakers"]:
            if (lang_key, speaker) in benchmark["speakers"]:
                issues.append({"check": "addition_speaker_in_benchmark_exposure",
                               "window_id": wid, "speaker": speaker, "language": lang_key})
            if (lang_key, speaker) in v1["speakers"]:
                issues.append({"check": "addition_speaker_in_v1_prior_exposure",
                               "window_id": wid, "speaker": speaker, "language": lang_key})
            split = "train" if row.get("split") == "train" else "val"
            prev = speaker_split.get((lang_key, speaker))
            if prev is not None and prev != split:
                issues.append({"check": "addition_speaker_shared_across_splits",
                               "speaker": speaker, "language": lang_key, "splits": sorted({prev, split})})
            speaker_split[(lang_key, speaker)] = split
        for kind in ("references", "records"):
            for key in keys[kind]:
                if key in benchmark[kind]:
                    issues.append({"check": f"addition_{kind[:-1]}_in_benchmark_exposure",
                                   "window_id": wid, "key": list(key)})
                if key in v1[kind]:
                    issues.append({"check": f"addition_{kind[:-1]}_in_v1_prior_exposure",
                                   "window_id": wid, "key": list(key)})
        prepared_sha = str((row.get("prepared_audio") or {}).get("sha256"))
        if prepared_sha in benchmark["hashes"]:
            issues.append({"check": "addition_prepared_hash_in_benchmark", "window_id": wid})
        if not (row.get("parent_refs") or {}).get("dataset_revision") == REV_KB:
            issues.append({"check": "addition_revision_mismatch", "window_id": wid})
        if row.get("label") != 0 or row.get("dataset_id") != "kathbath":
            issues.append({"check": "addition_label_or_dataset_invalid", "window_id": wid})
        gender = str((row.get("speaker_ids") or {}).get("gender"))
        parsed = _parse_reference(Path(str(row.get("source_file", ""))).name)
        if gender != "m" or parsed is None or parsed["gender"] != "m" or _numeric((row.get("speaker_ids") or {}).get("speaker")) != parsed["speaker"]:
            issues.append({"check": "addition_gender_or_identity_not_documented_male", "window_id": wid})
        if row.get("preprocessing_version") != PREP_VERSION:
            issues.append({"check": "addition_preprocessing_version", "window_id": wid})

    for wid, count in seen_ids.items():
        if count > 1:
            issues.append({"check": "duplicate_window_id", "window_id": wid, "count": count})

    # -- transitive closure: additions vs v1 + held-out + benchmark anchors --
    closure_report = None
    if closure_edges and closure_anchors:
        closure_report = audit_closure(additions, closure_edges, closure_anchors)
        if closure_report["n_hits"]:
            issues.append({"check": "addition_closure_component_intersects_exposure",
                           "count": closure_report["n_hits"], "examples": closure_report["hits"][:5]})

    # -- pairwise checks vs retained rows and benchmark ----------------------
    def pair_keys(rows: list[dict]) -> dict[str, set]:
        speakers, references, records = set(), set(), set()
        for row in rows:
            ids = row_identities(row)
            lang = ids["language"]
            for key in ids["speakers"].values():
                speakers.add((lang, key[2]))
            for key in ids["references"].values():
                references.add((lang, key[2]))
            for key in ids["records"]:
                records.add((lang, key[2]))
        return {"speakers": speakers, "references": references, "records": records}

    add_keys = {"speakers": set(), "references": set(), "records": set()}
    for row in additions:
        for kind in add_keys:
            add_keys[kind] |= addition_keys(row)[kind]
    retained_keys = pair_keys(retained)
    benchmark_rows = v2_test_rows or []
    benchmark_row_keys = pair_keys(benchmark_rows) if benchmark_rows else {
        "speakers": benchmark["speakers"], "references": benchmark["references"], "records": benchmark["records"]}
    for kind in ("speakers", "references", "records"):
        retained_overlap = add_keys[kind] & retained_keys[kind]
        if retained_overlap:
            issues.append({"check": f"addition_{kind[:-1]}_shared_with_retained_split",
                           "count": len(retained_overlap), "examples": [list(x) for x in sorted(retained_overlap)[:5]]})
        bench_overlap = add_keys[kind] & benchmark_row_keys[kind]
        if bench_overlap:
            issues.append({"check": f"addition_{kind[:-1]}_shared_with_benchmark",
                           "count": len(bench_overlap), "examples": [list(x) for x in sorted(bench_overlap)[:5]]})

    return {"issues": issues, "n_issues": len(issues), "closure": closure_report,
            "n_additions": len(additions)}


# --------------------------------------------------------------------------- #
# file / audio checks
# --------------------------------------------------------------------------- #

def _resolve_base(row: dict, default_root: Path) -> Path:
    base = row.get("path_base")
    if base:
        return REPO / str(base)
    return default_root


def audit_files(additions: list[dict], retained: list[dict], benchmark_rows: list[dict]) -> dict:
    import soundfile as sf

    problems: list[dict] = []
    profiles: dict[str, dict] = {}
    checked = 0

    def check_row(row: dict, default_root: Path, need_raw: bool) -> None:
        nonlocal checked
        wid = str(row.get("window_id"))
        rel = (row.get("prepared_audio") or {}).get("path")
        if not rel:
            problems.append({"check": "missing_prepared_path", "window_id": wid})
            return
        path = (_resolve_base(row, default_root) / rel).resolve()
        if not path.exists():
            problems.append({"check": "prepared_path_missing", "window_id": wid, "path": str(path)})
            return
        if _sha256(path) != (row.get("prepared_audio") or {}).get("sha256"):
            problems.append({"check": "prepared_hash_mismatch", "window_id": wid})
        try:
            samples, rate = sf.read(path, dtype="float32")
        except Exception as error:  # noqa: BLE001
            problems.append({"check": "decode_failed", "window_id": wid, "detail": str(error)[:200]})
            return
        checked += 1
        if rate != 16000 or samples.ndim != 1:
            problems.append({"check": "audio_contract", "window_id": wid, "rate": rate})
        if not np.isfinite(samples).all():
            problems.append({"check": "nonfinite_samples", "window_id": wid})
        duration = len(samples) / rate
        if not (1.0 <= duration <= 4.0):
            problems.append({"check": "audio_duration", "window_id": wid, "duration": duration})
        profiles[wid] = {"base": "addition" if need_raw else "retained", "duration": duration,
                         "fingerprint": _fingerprint(samples, rate)}
        if need_raw:
            raw_rel = f"raw/kathbath/{str(row.get('spoken_language', '')).lower()}/{Path(str(row.get('source_file'))).name}"
            raw_path = (REPO / "artifacts" / "datasets-v3-coverage" / raw_rel)
            if not raw_path.exists():
                problems.append({"check": "addition_raw_missing", "window_id": wid})
            elif _sha256(raw_path) != (row.get("original_audio") or {}).get("sha256"):
                problems.append({"check": "addition_original_hash_mismatch", "window_id": wid})

    for row in retained:
        check_row(row, V1, need_raw=False)
    for row in benchmark_rows:
        check_row(row, V2, need_raw=False)
    for row in additions:
        check_row(row, V3, need_raw=True)

    # near-duplicate scan across all profiles (additions included)
    near = []
    ids = sorted(profiles)
    if ids:
        matrix = np.stack([profiles[w]["fingerprint"] for w in ids])
        durations = np.array([profiles[w]["duration"] for w in ids])
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if abs(durations[i] - durations[j]) > NEAR_DUP_DURATION_TOL:
                    continue
                similarity = float(matrix[i] @ matrix[j])
                if similarity >= NEAR_DUP_SIM:
                    near.append({"window_a": ids[i], "window_b": ids[j],
                                 "similarity": round(similarity, 6),
                                 "touches_addition": "addition" in (profiles[ids[i]]["base"], profiles[ids[j]]["base"])})
    return {"problems": problems, "checked": checked, "near_duplicates": near,
            "near_duplicate_method": (
                f"cosine similarity >= {NEAR_DUP_SIM} over joint RMS+ZCR frame profiles (40 ms frames, "
                f"64+64 fixed bins), |duration delta| <= {NEAR_DUP_DURATION_TOL}s; audio-only, "
                "severe transcodes can evade (documented limitation)")}


# --------------------------------------------------------------------------- #
# retention / replay / budget / preservation
# --------------------------------------------------------------------------- #

def check_retention(retained_v2: list[dict], manifest_rows: list[dict]) -> dict:
    by_id = {str(r.get("window_id")): r for r in manifest_rows}
    missing, changed = [], []
    for row in retained_v2:
        wid = str(row.get("window_id"))
        current = by_id.get(wid)
        if current is None:
            missing.append(wid)
        elif json.dumps(current, sort_keys=True) != json.dumps(row, sort_keys=True):
            changed.append(wid)
    return {"expected": len(retained_v2), "missing": missing[:10], "changed": changed[:10],
            "missing_count": len(missing), "changed_count": len(changed)}


def check_plan_replay(plan: dict, train_rows: list[dict], dev_rows: list[dict], version: dict) -> dict:
    expected = []
    for language, entry in plan["languages"].items():
        for row in entry.get("train", []):
            expected.append((str(row["fname"]), "train"))
        for row in entry.get("dev", []):
            expected.append((str(row["fname"]), "val"))
    manifest = []
    for split, rows in (("train", train_rows), ("dev", dev_rows)):
        for row in rows:
            if row.get("coverage_addition") != "v3-male-genuine":
                continue
            fname = Path(str(row.get("source_file"))).name
            manifest.append((fname, split if split == "train" else "val"))
    expected_set, manifest_set = set(expected), set(manifest)
    hashes = {}
    for split in ("train", "dev"):
        path = Path(version.get("manifests", {}).get(split, {}).get("path", ""))
        hashes[split] = bool(path.exists() and _sha256(path) == version["manifests"][split]["sha256"])
    return {"assignment_match": expected_set == manifest_set,
            "missing_from_manifest": sorted(x[0] for x in (expected_set - manifest_set))[:5],
            "unexpected_in_manifest": sorted(x[0] for x in (manifest_set - expected_set))[:5],
            "manifest_hashes_match": hashes}


def check_budget_preservation() -> dict:
    snapshot_after = json.loads((V3 / "planning" / "snapshot_after_check.json").read_text()) \
        if (V3 / "planning" / "snapshot_after_check.json").exists() else None
    footer_charges = json.loads((V3 / "planning" / "footer_charges.json").read_text()) \
        if (V3 / "planning" / "footer_charges.json").exists() else {}
    summary = json.loads((V3 / "staging" / "materialization_summary.json").read_text())
    failed = json.loads((V3 / "staging_failed_attempt1" / "materialization_summary.json").read_text()) \
        if (V3 / "staging_failed_attempt1" / "materialization_summary.json").exists() else None
    ledger = json.loads((V1 / "manifests" / "downloads.json").read_text())

    snapshot_ok = bool(snapshot_after) and not snapshot_after.get("changed") \
        and not snapshot_after.get("missing")
    failed_charged = 0
    if failed:
        for entry in failed["languages"].values():
            for event in entry.get("fetch", {}).get("charged_skips", []):
                failed_charged += int(event.get("charged_bytes") or 0)
    return {
        "snapshot": {"ok": snapshot_ok,
                     "changed": (snapshot_after or {}).get("changed", []),
                     "missing": (snapshot_after or {}).get("missing", [])},
        "footer_charges_bytes": sum(sum(v.values()) for v in footer_charges.values()) if footer_charges else 0,
        "failed_attempt_charged_bytes": failed_charged,
        "ledger": {"total_bytes": ledger["total_bytes"], "cap_bytes": ledger["cap_bytes"],
                   "remaining_bytes": ledger["remaining_bytes"], "events": ledger["events"]},
    }


def _strip_state_fields(plan: dict) -> dict:
    """Copy of the plan with internal and point-in-time accounting fields removed.

    Duplicated deliberately from the selector module (the validator uses its
    own implementation so the emitted digests are checked against an
    independently derived projection).  Underscore-prefixed internal keys are
    dropped exactly as the serialized plan drops them; ledger totals, the
    estimated total and the one-time footer-read charges are the documented
    point-in-time fields.
    """
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if not str(k).startswith("_")}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    view = clean(plan)
    budget = view.get("budget")
    if isinstance(budget, dict):
        budget.pop("ledger_before", None)
        budget.pop("ledger_after_planning", None)
        budget.pop("estimated_total", None)
    for entry in (view.get("languages") or {}).values():
        if isinstance(entry, dict):
            entry.pop("footer_charged_bytes", None)
    return view


def _canonical_sha(payload) -> str:
    import hashlib
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def check_selection_determinism(stored_plan: dict) -> dict:
    """Zero-charge replan reproduces the frozen selection exactly.

    Recomputes the plan from the planning caches (no network reads allowed),
    strips the documented point-in-time fields, and requires byte-identical
    canonical digests, the accounting identity, and agreement with the
    emitted replay report.
    """
    report_path = V3 / "reports" / "replay_check.json"
    emitted = json.loads(report_path.read_text()) if report_path.exists() else {}
    from src.dataset_prep.coverage_v3 import plan_with_targets

    ledger_path = V1 / "manifests" / "downloads.json"
    ledger_before = json.loads(ledger_path.read_text())["total_bytes"]
    recomputed = plan_with_targets()
    ledger_after = json.loads(ledger_path.read_text())["total_bytes"]

    validator_digest_stored = _canonical_sha(_strip_state_fields(stored_plan))
    validator_digest_recomputed = _canonical_sha(_strip_state_fields(recomputed))

    def footer_total(plan) -> int:
        return sum(int(e.get("footer_charged_bytes") or 0)
                   for e in (plan.get("languages") or {}).values())

    stored_est = int(stored_plan["budget"]["estimated_total"])
    recomp_est = int(recomputed["budget"]["estimated_total"])
    stored_footer, recomp_footer = footer_total(stored_plan), footer_total(recomputed)
    return {
        "zero_charge_replay": ledger_before == ledger_after,
        "selection_match": validator_digest_stored == validator_digest_recomputed,
        "selection_digest": validator_digest_stored,
        "footer_charged_bytes_stored": stored_footer,
        "footer_charged_bytes_recomputed": recomp_footer,
        "estimate_identity_ok": (stored_est - stored_footer) == (recomp_est - recomp_footer),
        "emitted_report_agrees": bool(
            emitted.get("selection_match") is True
            and emitted.get("selection_digest_stored") == validator_digest_stored
            and emitted.get("selection_digest_recomputed") == validator_digest_recomputed
            and (emitted.get("state_fields") or {}).get("identity_ok") is True
            and int(emitted.get("replay_charges_bytes") or 0) == 0
        ),
    }


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

def _addition(wid: str, language: str, speaker: str, split: str = "train",
              record: str = "700001", fname: str | None = None,
              path_base: str = "artifacts/datasets-v3-coverage") -> dict:
    fname = fname or f"{record}-{speaker}-m.m4a"
    return {
        "window_id": wid, "recording_id": wid, "dataset_id": "kathbath",
        "source_url": "https://example.invalid/kathbath", "source_file": f"{language.lower()}/{fname}",
        "original_split": "train", "pool": "core", "split": split, "label": 0,
        "label_name": "genuine", "label_source": "official Kathbath corpus (genuine speech)",
        "spoken_language": language, "native_language": None,
        "speaker_ids": {"speaker": speaker, "gender": "m"}, "generator": None, "generator_version": None,
        "parent_refs": {"dataset_revision": REV_KB, "shard": f"{language.lower()}/train-00000-of-00001.parquet",
                        "row_group": 0, "row_index": 1, "upstream_duration_seconds": 5.0, "upstream_split": "train"},
        "original_audio": {"sha256": f"orig-{wid}"}, "prepared_audio": {"sha256": f"prep-{wid}", "path": f"prepared/kathbath/{language.lower()}/{wid}.wav"},
        "window": {"duration_seconds": 3.0, "sample_rate": 16000},
        "preprocessing_version": PREP_VERSION, "path_base": path_base,
        "coverage_addition": "v3-male-genuine",
    }


def run_fixtures() -> dict:
    results = []
    benchmark = {"speakers": {("bengali", "99")}, "references": {("bengali", "900-99-m")},
                 "records": {("bengali", "900")}, "hashes": {"prep-bench"}}
    v1 = {"speakers": {("hindi", "55")}, "references": {("hindi", "800-55-m")}, "records": {("hindi", "800")}}

    def case(name: str, additions: list[dict], must_include: Sequence[str] = (), must_exclude: Sequence[str] = (),
             closure_edges: Sequence = (), closure_anchors: Sequence = ()):
        report = audit_coverage(additions, [], benchmark, v1,
                                closure_edges=closure_edges, closure_anchors=closure_anchors)
        checks = {issue["check"] for issue in report["issues"]}
        ok = all(c in checks for c in must_include) and all(c not in checks for c in must_exclude)
        results.append({"case": name, "must_include": list(must_include), "must_exclude": list(must_exclude),
                        "issues": sorted(checks), "passed": ok, "detail": report["issues"][:3]})

    # 1) benchmark male genuine speaker attempted in training
    case("benchmark_male_genuine_attempted_in_train",
         [_addition("add-1", "Bengali", "99", record="900")],
         must_include=["addition_speaker_in_benchmark_exposure"])

    # 2) benchmark synthetic SOURCE/TARGET speaker attempted as new genuine
    case("benchmark_synthetic_participant_attempted",
         [_addition("add-2", "Bengali", "99", record="901")],
         must_include=["addition_speaker_in_benchmark_exposure"])

    # 3) old-development identity attempted in training
    case("old_exposure_identity_attempted",
         [_addition("add-3", "Hindi", "55", record="801")],
         must_include=["addition_speaker_in_v1_prior_exposure"])

    # 4) shared speaker across added train/dev
    case("speaker_shared_across_added_splits",
         [_addition("add-4a", "Tamil", "70", split="train", record="700010"),
          _addition("add-4b", "Tamil", "70", split="val", record="700011")],
         must_include=["addition_speaker_shared_across_splits"])

    # 5) clean additions pass
    case("clean_additions_accepted",
         [_addition("add-5a", "Punjabi", "71", split="train", record="700020"),
          _addition("add-5b", "Punjabi", "72", split="val", record="700021")],
         must_exclude=["addition_speaker_in_benchmark_exposure", "addition_speaker_in_v1_prior_exposure",
                       "addition_speaker_shared_across_splits"])

    # 6) wrong path base
    case("wrong_path_base_rejected",
         [_addition("add-6", "Urdu", "73", record="700030", path_base="artifacts/datasets-v2")],
         must_include=["addition_wrong_path_base"])

    # 7) transitive bridge into benchmark anchor (closure)
    case("transitive_bridge_to_benchmark_rejected",
         [_addition("add-7", "Marathi", "74", record="700040")],
         must_include=["addition_closure_component_intersects_exposure"],
         closure_edges=[(("j", "marathi", "700040"), ("s", "marathi", "777"))],
         closure_anchors=[("s", "marathi", "777")])

    # 8) renamed / re-encoded duplicate (identical prepared hash to benchmark)
    dup = _addition("add-8", "Gujarati", "75", record="700050")
    dup["prepared_audio"]["sha256"] = "prep-bench"
    case("renamed_duplicate_hash_rejected",
         [dup], must_include=["addition_prepared_hash_in_benchmark"])

    # 9) insufficient speaker shortfall: one speaker may not serve both splits —
    #    reserving it for dev (train empty) is accepted and reported by the plan
    case("single_speaker_reserved_for_dev_accepted",
         [_addition("add-9", "Odia", "76", split="val", record="700060")],
         must_exclude=["addition_speaker_shared_across_splits"])

    # 10) failed-read budget retention (reader behavior, no network)
    class _FailingInner:
        def __init__(self): self.calls = 0
        def read(self, size): self.calls += 1; raise RuntimeError("simulated network failure")
        def close(self): pass

    class _FakeLedger:
        def __init__(self): self.total = 0; self.events = 0
        def charge(self, source, size): self.total += size; self.events += 1

    ledger = _FakeLedger()
    reader = BoundedReader(_FailingInner(), ledger, "kathbath", {"remaining": 1000})
    retained = False
    try:
        reader.read(256)
    except RuntimeError:
        retained = ledger.total == 256
    results.append({"case": "failed_read_charge_retained", "passed": retained,
                    "charged": ledger.total, "must_include": ["charge_before_failed_read"]})

    # 11) allowance exhaustion blocks read before transfer
    ledger2 = _FakeLedger()
    reader2 = BoundedReader(_FailingInner(), ledger2, "kathbath", {"remaining": 10})
    blocked = False
    try:
        reader2.read(256)
    except BudgetExceeded:
        blocked = ledger2.total == 0
    results.append({"case": "allowance_exhaustion_blocks_before_io", "passed": blocked,
                    "charged": ledger2.total, "must_include": ["no_charge_when_blocked"]})

    passed = all(r["passed"] for r in results)
    return {"passed": passed, "cases": results}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", action="store_true")
    parser.add_argument("--json-out", default=str(V3 / "reports" / "audit.json"))
    args = parser.parse_args(argv)

    if args.fixtures:
        report = run_fixtures()
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 1

    train_rows = _rows(V3 / "manifests" / "windows.train.jsonl")
    dev_rows = _rows(V3 / "manifests" / "windows.dev.jsonl")
    retained_train = _rows(V2 / "manifests" / "windows.train.jsonl")
    retained_dev = _rows(V2 / "manifests" / "windows.dev.jsonl")
    test_rows = _rows(V2 / "manifests" / "windows.test.jsonl")
    additions = [r for r in train_rows + dev_rows if r.get("coverage_addition") == "v3-male-genuine"]
    retained = retained_train + retained_dev

    benchmark = benchmark_keys()
    v1 = v1_keys()
    closure = load_closure_data()
    anchors = list(closure["anchors"])
    for lang, speaker in benchmark["speakers"]:
        anchors.append(("s", lang, speaker))
    for lang, canonical in benchmark["references"]:
        anchors.append(("r", lang, canonical))
        anchors.append(("j", lang, canonical.split("-")[0]))
    for lang, record in benchmark["records"]:
        anchors.append(("j", lang, record))

    core = audit_coverage(additions, retained, benchmark, v1,
                          closure_edges=closure["edges"], closure_anchors=anchors,
                          v2_test_rows=test_rows)
    files = audit_files(additions, retained, test_rows)
    retention = {"train": check_retention(retained_train, train_rows),
                 "dev": check_retention(retained_dev, dev_rows)}
    version = json.loads((V3 / "manifests" / "dataset-version.v3.json").read_text())
    plan = json.loads((V3 / "planning" / "coverage_plan.json").read_text())
    replay = check_plan_replay(plan, train_rows, dev_rows, version)
    determinism = check_selection_determinism(plan)
    budget = check_budget_preservation()

    issues = list(core["issues"])
    for split, check in retention.items():
        if check["missing_count"] or check["changed_count"]:
            issues.append({"check": f"retention_{split}_rows_missing_or_changed",
                           "missing": check["missing_count"], "changed": check["changed_count"]})
    if not replay["assignment_match"]:
        issues.append({"check": "replay_assignment_mismatch",
                       "missing": replay["missing_from_manifest"], "unexpected": replay["unexpected_in_manifest"]})
    if not all(replay["manifest_hashes_match"].values()):
        issues.append({"check": "manifest_hash_mismatch", "detail": replay["manifest_hashes_match"]})
    if not (determinism["zero_charge_replay"] and determinism["selection_match"]
            and determinism["estimate_identity_ok"] and determinism["emitted_report_agrees"]):
        issues.append({"check": "selection_determinism_failed", "detail": determinism})
    if files["problems"]:
        grouped = defaultdict(list)
        for problem in files["problems"]:
            grouped[problem["check"]].append(problem)
        for check, items in sorted(grouped.items()):
            issues.append({"check": check, "count": len(items), "examples": items[:5]})
    addition_near = [pair for pair in files["near_duplicates"] if pair["touches_addition"]]
    if addition_near:
        issues.append({"check": "addition_near_duplicate", "count": len(addition_near), "examples": addition_near[:5]})
    if not budget["snapshot"]["ok"]:
        issues.append({"check": "frozen_artifact_preservation_failed",
                       "changed": budget["snapshot"]["changed"], "missing": budget["snapshot"]["missing"]})

    report = {
        "schema": SCHEMA, "ready": len(issues) == 0, "n_issues": len(issues), "issues": issues,
        "counts": {"train_windows": len(train_rows), "dev_windows": len(dev_rows),
                   "additions": len(additions),
                   "retained_train": len(retained_train), "retained_dev": len(retained_dev)},
        "retention": retention, "replay": replay, "determinism": determinism, "budget": budget,
        "closure": core["closure"], "files": {k: v for k, v in files.items() if k != "problems"},
        "audio_problems": len(files["problems"]), "near_duplicate_pairs": len(files["near_duplicates"]),
    }
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({k: report[k] for k in ("ready", "n_issues", "counts", "retention", "replay",
                                             "audio_problems", "near_duplicate_pairs")},
                     indent=2, default=str))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
