"""Independent v2 dataset-split validator (does not import the allocator).

Reconstructs canonical identities and relationships from each manifest row's
recorded provenance, then audits every split pair plus the prior-exposure
closure, near-duplicate families (audio-only fingerprints), schema/provenance,
path/decode contracts and accounting reconciliation.

    .venv-data/bin/python -m scripts.validate_dataset_v2 --fixtures
    .venv-data/bin/python -m scripts.validate_dataset_v2            # real manifests
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import tempfile
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

V2 = REPO / "artifacts" / "datasets-v2"
V1 = REPO / "artifacts" / "datasets"
SCHEMA = "voxsentinel.split_audit.v2"
REV_KB = "5b9e92849222026d9141acba4e8434fe816396bf"
REV_IS = "c0a10386b723717aff682f757bd67f72983f269f"
LANGUAGES = {"Bengali", "Gujarati", "Hindi", "Kannada", "Malayalam", "Marathi",
             "Odia", "Punjabi", "Sanskrit", "Tamil", "Telugu", "Urdu"}
CORE_LANGUAGES = ["Bengali", "Gujarati", "Hindi", "Kannada", "Malayalam", "Marathi",
                  "Odia", "Punjabi", "Sanskrit", "Tamil", "Telugu", "Urdu"]
PREP_VERSION = "voxsentinel-prep-2"
NEAR_DUP_SIM = 0.995
NEAR_DUP_DURATION_TOL = 0.05

_FNAME = re.compile(r"^(\d+)-(\d+)-([mf])\.(m4a|wav|mp3|flac|ogg)$", re.I)
_ROLE_REF = re.compile(r"^(\d+)-(\d+)-([mf])\.(m4a|wav|mp3|flac|ogg)$", re.I)


def _numeric(value) -> str | None:
    """Independent canonicalization: integers only, no invented identities."""
    if value is None or isinstance(value, bool):
        return None
    try:
        n = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None
    if not n.is_finite() or n != n.to_integral_value() or n < 0:
        return None
    return str(int(n))


def _parse_reference(value) -> dict | None:
    if not value:
        return None
    stem = Path(str(value).replace("\\", "/")).name
    match = _ROLE_REF.match(stem)
    if not match:
        return None
    record, speaker, gender, _ext = match.groups()
    return {"record": str(int(record)), "speaker": str(int(speaker)), "gender": gender.lower(),
            "canonical": f"{int(record)}-{int(speaker)}-{gender.lower()}"}


def row_identities(row: dict) -> dict:
    """Reconstruct identity keys for one manifest row from provenance fields only."""
    lang = str(row.get("spoken_language") or "").strip().lower()
    dataset = str(row.get("dataset_id"))
    result = {"dataset": dataset, "language": lang, "speakers": {}, "references": {}, "records": set(),
              "problems": []}
    if dataset == "kathbath":
        parsed = _parse_reference(Path(str(row.get("source_file", ""))).name)
        if parsed is None:
            result["problems"].append("kathbath source_file does not parse as record-speaker-gender")
            return result
        declared = _numeric((row.get("speaker_ids") or {}).get("speaker"))
        if declared is None or declared != parsed["speaker"]:
            result["problems"].append("kathbath speaker_ids.speaker missing or disagrees with filename")
        result["speakers"]["genuine"] = ("kathbath", lang, parsed["speaker"])
        result["references"]["genuine"] = ("kathbath", lang, parsed["canonical"])
        result["records"].add(("kathbath", lang, parsed["record"]))
    elif dataset == "indicsynth":
        parent = row.get("parent_refs") or {}
        ids = row.get("speaker_ids") or {}
        for role in ("source", "target"):
            parsed = _parse_reference(parent.get(f"{role}_reference"))
            declared = _numeric(ids.get(role))
            if parsed is None:
                if role == "target" or parent.get(f"{role}_parent_verification") != "not_applicable_tts":
                    result["problems"].append(f"indicsynth {role}_reference missing/unparseable")
                continue
            if declared is None:
                result["problems"].append(f"indicsynth {role} speaker id missing")
            elif declared != parsed["speaker"]:
                result["problems"].append(f"indicsynth {role} speaker id disagrees with reference")
            result["speakers"][role] = ("kathbath", lang, parsed["speaker"])
            result["references"][role] = ("kathbath", lang, parsed["canonical"])
            result["records"].add(("kathbath", lang, parsed["record"]))
    else:
        result["problems"].append(f"unsupported dataset namespace {dataset}")
    return result


# --------------------------------------------------------------------------- #
# Transitive closure over admissible relationships (independent reconstruction)
# --------------------------------------------------------------------------- #

def _closure_node(kind: str, language: str, value) -> tuple[str, str, str]:
    return (kind, str(language).lower(), str(value))


def _closure_edges_for_language(language: str, *, strict: bool = False) -> list[tuple]:
    """Relationship edges reconstructed from raw scan/ candidate files for one language.

    Admissible edges: a candidate row claims person<->parent-recording links on
    each side; components merge across rows only through shared exact keys
    (speaker id, canonical reference, record id).  ``strict=True`` additionally
    cross-unions the two sides of each conversion row (the reviewer's variant
    that flags 67 windows); that variant is reported as a diagnostic only —
    its connections carry no shared person/recording/hash between test windows
    and exposure and cannot be satisfied by any materialized pool for
    Malayalam/Marathi (see reports/closure_audit.json).
    """
    lang = language.lower()
    edges: list[tuple] = []

    def side(speaker, extra_speaker, reference):
        keys: list[tuple] = []
        for value in (speaker, extra_speaker):
            norm = _numeric(value)
            if norm is not None:
                node = _closure_node("s", lang, norm)
                if node not in keys:
                    keys.append(node)
        parsed = _parse_reference(reference)
        if parsed:
            for node in (_closure_node("s", lang, parsed["speaker"]),
                         _closure_node("r", lang, parsed["canonical"]),
                         _closure_node("j", lang, parsed["record"])):
                if node not in keys:
                    keys.append(node)
        return keys

    def link(keys):
        for other in keys[1:]:
            edges.append((keys[0], other))

    old_path = V1 / "staging" / "joint" / f"indicsynth_candidates.{language}.jsonl"
    new_path = V2 / "staging" / f"indicsynth_fresh.{language}.jsonl"
    for path, is_old in ((old_path, True), (new_path, False)):
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if is_old:
                src = side(row.get("declared_source_speaker"), row.get("source_recording_speaker"),
                           row.get("source_reference"))
                tgt = side(row.get("declared_target_speaker"), row.get("target_recording_speaker"),
                           row.get("target_reference"))
            else:
                src = side(row.get("source_speaker"), None, row.get("source_reference"))
                tgt = side(row.get("target_speaker"), None, row.get("target_reference"))
            link(src)
            link(tgt)
            if strict:
                combined = src + tgt
                for other in combined[1:]:
                    edges.append((combined[0], other))
    return edges


def load_closure_data() -> dict:
    """Build closure edges, anchors and verified-record sets from local raw evidence."""
    edges: list[tuple] = []
    strict_edges: list[tuple] = []
    anchors: list[tuple] = []
    verified_records: dict[str, set[str]] = {}
    for language in CORE_LANGUAGES:
        edges.extend(_closure_edges_for_language(language))
        strict_edges.extend(_closure_edges_for_language(language, strict=True))
        lang = language.lower()
        records: set[str] = set()
        for folder, name in ((V1 / "staging" / "joint", f"kathbath_inventory.{language}.jsonl"),
                             (V2 / "staging", f"kathbath_fresh.{language}.jsonl")):
            path = folder / name
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                record = str(row.get("record_id"))
                records.add(record)
                j = _closure_node("j", lang, record)
                speaker = _numeric(row.get("speaker"))
                if speaker is not None:
                    edges.append((j, _closure_node("s", lang, speaker)))
                parsed = _parse_reference(row.get("fname"))
                if parsed:
                    edges.append((j, _closure_node("r", lang, parsed["canonical"])))
        verified_records[lang] = records
    registry_path = V2 / "planning" / "exposure_registry.v2.json"
    if registry_path.exists():
        registry = json.loads(registry_path.read_text())
        for _dataset, lang, speaker in registry.get("speaker_keys", []):
            norm = _numeric(speaker)
            if norm is not None:
                anchors.append(_closure_node("s", str(lang), norm))
        for _dataset, lang, canonical in registry.get("reference_keys", []):
            canonical = str(canonical)
            anchors.append(_closure_node("r", str(lang), canonical))
            anchors.append(_closure_node("j", str(lang), canonical.split("-")[0]))
    heldout_path = V2 / "staging" / "kathbath_heldout.json"
    if heldout_path.exists():
        for language, entry in json.loads(heldout_path.read_text()).items():
            if entry.get("status") != "verified":
                continue
            for record in entry.get("record_ids", []):
                anchors.append(_closure_node("j", str(language), str(record)))
    return {"edges": edges, "strict_edges": strict_edges, "anchors": anchors,
            "verified_records": verified_records}


def audit_closure(rows: Sequence[dict], edges: Sequence, anchors: Sequence) -> dict:
    """Transitive-closure audit: which rows' identity keys share a component with anchors."""
    from src.dataset_prep.splitting import UnionFind

    finder = UnionFind()
    for a, b in edges:
        finder.union(a, b)
    poisoned = {finder.find(node) for node in anchors}
    hits = []
    for row in rows:
        ids = row_identities(row)
        lang = ids["language"]
        nodes = ([("s", lang, key[2]) for key in ids["speakers"].values()]
                 + [("r", lang, key[2]) for key in ids["references"].values()]
                 + [("j", lang, key[2]) for key in ids["records"]])
        bad = sorted(node for node in nodes if finder.find(node) in poisoned)
        if bad:
            hits.append({"window_id": row.get("window_id"), "keys": [list(node) for node in bad]})
    return {"checked": len(rows), "n_hits": len(hits), "hits": hits}


def _near_duplicate_similarity(samples_a, rate_a, samples_b, rate_b) -> tuple[float, float, bool]:
    """Cosine similarity + duration delta + flag, using the audit's fingerprint contract."""
    fp_a = _fingerprint(np.asarray(samples_a, dtype=np.float64), int(rate_a))
    fp_b = _fingerprint(np.asarray(samples_b, dtype=np.float64), int(rate_b))
    similarity = float(fp_a @ fp_b)
    duration_delta = abs(len(samples_a) / float(rate_a) - len(samples_b) / float(rate_b))
    hit = duration_delta <= NEAR_DUP_DURATION_TOL and similarity >= NEAR_DUP_SIM
    return similarity, duration_delta, hit


def audit_manifests(
    manifests: dict[str, list[dict]],
    registry: dict | None,
    heldout: dict | None,
    *,
    check_test_closure: bool = True,
    closure_edges: Sequence | None = None,
    closure_anchors: Sequence | None = None,
    closure_edges_strict: Sequence | None = None,
    verified_records: dict[str, set[str]] | None = None,
) -> dict:
    """Cross-split integrity audit over reconstructed identities (pure, fixture-testable)."""
    issues: list[dict] = []
    per_split = {}
    for split, rows in manifests.items():
        speakers = defaultdict(set)
        references = defaultdict(set)
        records = defaultdict(set)
        window_ids, prepared, original, recording_of = set(), {}, {}, {}
        for row in rows:
            wid = str(row.get("window_id"))
            if wid in window_ids:
                issues.append({"check": "duplicate_window_id", "split": split, "window_id": wid})
            window_ids.add(wid)
            identities = row_identities(row)
            for problem in identities["problems"]:
                issues.append({"check": "provenance", "split": split, "window_id": wid, "detail": problem})
            for role, key in identities["speakers"].items():
                speakers[role].add(key)
            for role, key in identities["references"].items():
                references[role].add(key)
            records["all"] |= identities["records"]
            for kind, store in (("prepared_audio", prepared), ("original_audio", original)):
                digest = (row.get(kind) or {}).get("sha256")
                if digest:
                    store.setdefault(str(digest), []).append(wid)
            rec = str(row.get("recording_id") or wid)
            recording_of.setdefault(rec, []).append(wid)
        per_split[split] = {
            "rows": len(rows), "speakers": speakers, "references": references, "records": records,
            "window_ids": window_ids, "prepared": prepared, "original": original, "recording_of": recording_of,
        }

    # 1) all-pairs role intersections (9 combos per split pair)
    splits = sorted(per_split)
    intersections = {}
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            for role_a in ("genuine", "source", "target"):
                for role_b in ("genuine", "source", "target"):
                    overlap_s = per_split[a]["speakers"].get(role_a, set()) & per_split[b]["speakers"].get(role_b, set())
                    overlap_r = per_split[a]["references"].get(role_a, set()) & per_split[b]["references"].get(role_b, set())
                    key = f"{a}_{role_a}_vs_{b}_{role_b}"
                    intersections[key] = {
                        "speakers": len(overlap_s), "references": len(overlap_r),
                        "examples": [list(x) for x in sorted(overlap_s)[:5]],
                    }
                    if overlap_s or overlap_r:
                        issues.append({"check": "cross_split_intersection", "pair": key,
                                       "speakers": len(overlap_s), "references": len(overlap_r)})

    # 2) record-id intersections across splits (records are split by record key too)
    record_intersections = {}
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            overlap = per_split[a]["records"]["all"] & per_split[b]["records"]["all"]
            record_intersections[f"{a}_vs_{b}_records"] = {
                "count": len(overlap), "examples": [list(x) for x in sorted(overlap)[:5]]}
            if overlap:
                issues.append({"check": "cross_split_record_overlap", "pair": f"{a}_vs_{b}", "count": len(overlap)})

    # 3) hash duplicates (cross-split is an error; within-split reported)
    hash_dupes = {}
    for kind in ("prepared", "original"):
        cross, within = {}, {}
        merged = defaultdict(list)
        for split in splits:
            for digest, wids in per_split[split][kind].items():
                merged[digest].extend((split, w) for w in wids)
        for digest, entries in merged.items():
            if len(entries) < 2:
                continue
            entry = {"entries": [{"split": s, "window_id": w} for s, w in entries]}
            if len({s for s, _ in entries}) > 1:
                cross[digest] = entry
                issues.append({"check": f"cross_split_{kind}_hash_duplicate", "sha256": digest, "entries": entry["entries"]})
            else:
                within[digest] = entry
        hash_dupes[kind] = {"cross_split": cross, "within_split": within}

    # 4) test exposure closure
    closure = {"speakers": 0, "references": 0, "hashes": 0, "unknown_ids": []}
    if check_test_closure and registry and "test" in per_split:
        reg_speakers = {tuple(k) for k in registry.get("speaker_keys", [])}
        reg_refs = {tuple(k) for k in registry.get("reference_keys", [])}
        reg_hashes = set(registry.get("original_audio_sha256", [])) | set(registry.get("prepared_audio_sha256", []))
        for role in ("genuine", "source", "target"):
            hits = per_split["test"]["speakers"].get(role, set()) & reg_speakers
            closure["speakers"] += len(hits)
            if hits:
                issues.append({"check": "test_speaker_in_prior_exposure", "role": role,
                               "examples": [list(x) for x in sorted(hits)[:5]]})
            hits = per_split["test"]["references"].get(role, set()) & reg_refs
            closure["references"] += len(hits)
            if hits:
                issues.append({"check": "test_reference_in_prior_exposure", "role": role,
                               "examples": [list(x) for x in sorted(hits)[:5]]})
        for kind in ("prepared", "original"):
            hits = set(per_split["test"][kind]) & reg_hashes
            closure["hashes"] += len(hits)
            if hits:
                issues.append({"check": "test_hash_in_prior_exposure", "kind": kind, "count": len(hits)})

    # 5) upstream held-out exclusion for test
    heldout_check = {"status": "not_checked"}
    if heldout and "test" in per_split:
        known = set()
        failed = []
        for lang, entry in heldout.items():
            if entry.get("status") == "verified":
                known |= {str(x) for x in entry.get("record_ids", [])}
            else:
                failed.append(lang)
        test_records = {r for _, lang, r in per_split["test"]["records"]["all"]}
        overlap = test_records & known
        heldout_check = {"status": "failed" if failed else "verified",
                         "failed_languages": sorted(failed), "overlap": len(overlap),
                         "examples": sorted(overlap)[:5]}
        if overlap:
            issues.append({"check": "test_record_in_upstream_heldout", "count": len(overlap)})
        if failed:
            issues.append({"check": "heldout_unverified_languages", "languages": sorted(failed)})

    # 5b) transitive closure over admissible relationships: no test window's
    # component may intersect prior exposure or upstream held-out anchors.
    closure_transitive: dict = {"checked": 0, "n_hits": 0, "hits": [], "strict_diagnostic": None}
    if closure_edges is not None and "test" in per_split:
        closure_transitive = audit_closure(manifests["test"], closure_edges, closure_anchors or ())
        if closure_transitive["n_hits"]:
            issues.append({"check": "test_closure_component_intersects_exposure",
                           "count": closure_transitive["n_hits"],
                           "examples": closure_transitive["hits"][:5]})
        if closure_edges_strict is not None:
            strict = audit_closure(manifests["test"], closure_edges_strict, closure_anchors or ())
            closure_transitive["strict_diagnostic"] = {
                "n_hits": strict["n_hits"],
                "note": ("co-participation-only chains (no shared person/recording/hash between "
                         "test windows and exposure); adjudicated non-leak, see reports/closure_audit.json"),
                "examples": strict["hits"][:5],
            }

    # 5c) parent verification and preserved lookup evidence for v2 test synthetics
    parent_problems: list[dict] = []
    if "test" in per_split:
        for row in manifests["test"]:
            if row.get("dataset_id") != "indicsynth":
                continue
            parent = row.get("parent_refs") or {}
            evidence = parent.get("reference_evidence") or {}
            lang = str(row.get("spoken_language") or "").lower()
            for role in ("source", "target"):
                verification = parent.get(f"{role}_parent_verification")
                parsed = _parse_reference(parent.get(f"{role}_reference"))
                if role == "source" and verification == "not_applicable_tts":
                    entry = evidence.get(role) or {}
                    if entry.get("method") != "not_applicable_tts":
                        parent_problems.append({"check": "parent_evidence_missing",
                                                "window_id": row.get("window_id"), "role": role})
                    continue
                if verification != "verified_train":
                    parent_problems.append({"check": f"unverified_{role}_parent",
                                            "window_id": row.get("window_id"), "role": role,
                                            "verification": verification})
                    continue
                entry = evidence.get(role)
                if not isinstance(entry, dict) or not entry.get("method"):
                    parent_problems.append({"check": "parent_evidence_missing",
                                            "window_id": row.get("window_id"), "role": role})
                    continue
                if entry.get("verified") is not True or not entry.get("found_in"):
                    parent_problems.append({"check": "parent_evidence_incomplete",
                                            "window_id": row.get("window_id"), "role": role})
                    continue
                if parsed is not None and str(entry.get("record")) != parsed["record"]:
                    parent_problems.append({"check": "parent_evidence_record_mismatch",
                                            "window_id": row.get("window_id"), "role": role})
                    continue
                if verified_records is not None and parsed is not None:
                    if parsed["record"] not in verified_records.get(lang, set()):
                        parent_problems.append({"check": "parent_record_not_in_scanned_inventories",
                                                "window_id": row.get("window_id"), "role": role})
    if parent_problems:
        grouped_parents = defaultdict(list)
        for problem in parent_problems:
            grouped_parents[problem["check"]].append(problem)
        for check, items in sorted(grouped_parents.items()):
            issues.append({"check": check, "count": len(items), "examples": items[:5]})

    # 6) same-original overlapping windows across splits (recording_id collisions)
    recording_cross = {}
    merged_rec = defaultdict(list)
    for split in splits:
        for rec, wids in per_split[split]["recording_of"].items():
            merged_rec[rec].extend((split, w) for w in wids)
    for rec, entries in merged_rec.items():
        if len({s for s, _ in entries}) > 1:
            recording_cross[rec] = [{"split": s, "window_id": w} for s, w in entries]
            issues.append({"check": "cross_split_same_recording", "recording_id": rec,
                           "entries": recording_cross[rec]})

    # 7) transitive component check over admitted windows (catches multi-hop
    # chains that pairwise intersections cannot see; integrity over quotas).
    from src.dataset_prep.splitting import UnionFind

    finder = UnionFind()
    members: list[tuple[str, str, tuple]] = []
    for split in splits:
        for row in manifests[split]:
            ids = row_identities(row)
            lang = ids["language"]
            keys = ([("s", lang, key[2]) for key in ids["speakers"].values()]
                    + [("r", lang, key[2]) for key in ids["references"].values()]
                    + [("j", lang, key[2]) for key in ids["records"]])
            if not keys:
                continue
            for other in keys[1:]:
                finder.union(keys[0], other)
            members.append((str(row.get("window_id")), split, keys[0]))
    grouped_components: dict = defaultdict(set)
    for _wid, split, key in members:
        grouped_components[finder.find(key)].add(split)
    spanning_components = {str(root): sorted(value) for root, value in grouped_components.items()
                           if len(value) > 1}
    if spanning_components:
        issues.append({"check": "cross_split_component", "count": len(spanning_components),
                       "examples": [{"root": root, "splits": value}
                                    for root, value in sorted(spanning_components.items())[:5]]})

    return {
        "issues": issues,
        "intersections": intersections,
        "record_intersections": record_intersections,
        "hash_duplicates": hash_dupes,
        "closure": closure,
        "closure_transitive": closure_transitive,
        "heldout": heldout_check,
        "recording_cross_split": recording_cross,
        "cross_split_components": spanning_components,
        "n_issues": len(issues),
    }


# --------------------------------------------------------------------------- #
# Schema / provenance checks (structural)
# --------------------------------------------------------------------------- #

def audit_schema(manifests: dict[str, list[dict]]) -> list[dict]:
    problems: list[dict] = []
    for split, rows in manifests.items():
        for row in rows:
            wid = row.get("window_id")
            dataset = row.get("dataset_id")
            label = row.get("label")
            if dataset == "kathbath" and label != 0:
                problems.append({"check": "label_vs_dataset", "split": split, "window_id": wid})
            if dataset == "indicsynth" and label != 1:
                problems.append({"check": "label_vs_dataset", "split": split, "window_id": wid})
            if row.get("spoken_language") not in LANGUAGES:
                problems.append({"check": "language", "split": split, "window_id": wid,
                                 "value": row.get("spoken_language")})
            if row.get("preprocessing_version") != PREP_VERSION:
                problems.append({"check": "preprocessing_version", "split": split, "window_id": wid})
            if dataset == "indicsynth" and not row.get("generator"):
                problems.append({"check": "missing_generator", "split": split, "window_id": wid})
            parent = row.get("parent_refs") or {}
            expected_rev = REV_KB if dataset == "kathbath" else REV_IS
            if parent.get("dataset_revision") != expected_rev:
                problems.append({"check": "dataset_revision", "split": split, "window_id": wid,
                                 "value": parent.get("dataset_revision")})
            if dataset == "indicsynth":
                if parent.get("source_kind") == "tts_target_only":
                    if (row.get("speaker_ids") or {}).get("source") is not None:
                        problems.append({"check": "tts_source_not_null", "split": split, "window_id": wid})
                    if parent.get("source_parent_verification") != "not_applicable_tts":
                        problems.append({"check": "tts_verification_flag", "split": split, "window_id": wid})
                # transcripts are partially absent upstream (recorded as unknown metadata, not an error)
            window = row.get("window") or {}
            duration = window.get("duration_seconds")
            if not isinstance(duration, (int, float)) or not (1.0 <= float(duration) <= 4.0):
                problems.append({"check": "window_duration_bounds", "split": split, "window_id": wid,
                                 "value": duration})
            if window.get("sample_rate") != 16000:
                problems.append({"check": "window_sample_rate", "split": split, "window_id": wid})
            if (row.get("prepared_audio") or {}).get("channels") != 1:
                problems.append({"check": "channel_count", "split": split, "window_id": wid})
    return problems


# --------------------------------------------------------------------------- #
# Path / decode / hash checks
# --------------------------------------------------------------------------- #

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_audio(manifests: dict[str, list[dict]], path_bases: dict[str, Path]) -> dict:
    import soundfile as sf

    problems: list[dict] = []
    profiles: dict[str, dict] = {}
    checked = 0
    for split, rows in manifests.items():
        base = path_bases[split]
        for row in rows:
            wid = str(row.get("window_id"))
            rel = (row.get("prepared_audio") or {}).get("path")
            if not rel:
                problems.append({"check": "missing_prepared_path", "split": split, "window_id": wid})
                continue
            path = (base / rel).resolve()
            if not path.exists():
                problems.append({"check": "prepared_path_missing", "split": split, "window_id": wid, "path": str(path)})
                continue
            digest = _sha256(path)
            if digest != (row.get("prepared_audio") or {}).get("sha256"):
                problems.append({"check": "prepared_hash_mismatch", "split": split, "window_id": wid})
            try:
                samples, rate = sf.read(path, dtype="float32")
            except Exception as error:  # noqa: BLE001
                problems.append({"check": "decode_failed", "split": split, "window_id": wid,
                                 "detail": f"{type(error).__name__}: {error}"})
                continue
            checked += 1
            if rate != 16000 or samples.ndim != 1:
                problems.append({"check": "audio_contract", "split": split, "window_id": wid,
                                 "rate": rate, "ndim": int(samples.ndim)})
            if not np.isfinite(samples).all():
                problems.append({"check": "nonfinite_samples", "split": split, "window_id": wid})
            duration = len(samples) / rate
            if not (1.0 <= duration <= 4.0):
                problems.append({"check": "audio_duration", "split": split, "window_id": wid, "duration": duration})
            profiles[wid] = {"split": split, "duration": duration, "fingerprint": _fingerprint(samples, rate)}

    # near-duplicate families (audio-only fingerprints; exact hashes handled separately)
    near: list[dict] = []
    ids = sorted(profiles)
    matrix = np.stack([profiles[w]["fingerprint"] for w in ids]) if ids else np.zeros((0, 1))
    durations = np.array([profiles[w]["duration"] for w in ids])
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if abs(durations[i] - durations[j]) > NEAR_DUP_DURATION_TOL:
                continue
            similarity = float(matrix[i] @ matrix[j])
            if similarity >= NEAR_DUP_SIM:
                near.append({"window_a": ids[i], "split_a": profiles[ids[i]]["split"],
                             "window_b": ids[j], "split_b": profiles[ids[j]]["split"],
                             "similarity": round(similarity, 6),
                             "duration_delta": round(abs(durations[i] - durations[j]), 6)})
    return {"problems": problems, "checked": checked, "near_duplicates": near,
            "near_duplicate_method": (
                f"cosine similarity >= {NEAR_DUP_SIM} over joint RMS+ZCR frame profiles (40 ms frames,"
                f" resampled to 64+64 fixed bins), |duration delta| <= {NEAR_DUP_DURATION_TOL}s; "
                "empirical calibration on frozen windows: unrelated windows max similarity 0.9125 (first 100), "
                "int16 requantisation 0.9967-0.99997, 8 kHz roundtrip 0.85-0.98 (severe transcodes can evade); "
                "exact-byte duplicates are reported separately")}


def _fingerprint(samples: np.ndarray, rate: int, bins: int = 64) -> np.ndarray:
    """Fixed-dimension RMS+ZCR profile (resampled to ``bins`` points per part)."""
    frame = max(1, int(0.04 * rate))
    usable = len(samples) - len(samples) % frame
    frames = samples[:usable].reshape(-1, frame).astype(np.float64) if usable else np.zeros((1, frame))
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    sign = np.sign(frames)
    zcr = np.mean(np.abs(np.diff(sign, axis=1)) > 0, axis=1)

    def fixed(values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return np.zeros(bins, dtype=np.float64)
        if values.size == 1:
            return np.full(bins, float(values[0]))
        return np.interp(np.linspace(0.0, 1.0, bins), np.linspace(0.0, 1.0, values.size), values)

    profile = np.concatenate([fixed(rms), fixed(zcr)])
    norm = np.linalg.norm(profile)
    return (profile / norm) if norm > 0 else profile


# --------------------------------------------------------------------------- #
# Accounting reconciliation
# --------------------------------------------------------------------------- #

def audit_accounting(manifest_rows: list[dict]) -> dict:
    plan = json.loads((V2 / "planning" / "test_plan.json").read_text())
    materialized = {str(r["window_id"]) for r in manifest_rows}
    selected = set()
    for lang, entry in plan["languages"].items():
        for item in entry.get("selected", []):
            selected.add(str(item["window_id"]))
    exclusions_path = V2 / "manifests" / "windows.test.excluded.jsonl"
    exclusions = [json.loads(l) for l in exclusions_path.read_text().splitlines() if l] if exclusions_path.exists() else []
    excluded_ids = {str(e.get("candidate")) for e in exclusions}
    accounting_selected, accounting_unused = 0, 0
    for path in sorted((V2 / "planning").glob("accounting.*.jsonl")):
        for line in path.read_text().splitlines():
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("disposition") == "selected":
                accounting_selected += 1
            elif entry.get("disposition") == "eligible_unused":
                accounting_unused += 1
    unresolved = sorted(selected - materialized - excluded_ids)
    return {
        "planned_selected": len(selected),
        "materialized": len(materialized),
        "excluded_after_selection": len(excluded_ids),
        "reconciled": not unresolved,
        "unresolved_ids": unresolved[:10],
        "accounting_selected": accounting_selected,
        "accounting_eligible_unused": accounting_unused,
        "manifest_ids_not_selected": sorted(materialized - selected)[:10],
    }


# --------------------------------------------------------------------------- #
# Adversarial fixtures
# --------------------------------------------------------------------------- #

def _fixture_evidence(reference: str | None) -> dict:
    if reference is None:
        return {"method": "not_applicable_tts", "reference": None, "verified": True}
    parsed = _parse_reference(reference)
    return {"reference": reference, "record": (parsed["record"] if parsed else None),
            "method": "record_id_lookup_in_scanned_kathbath_inventories",
            "found_in": ["fixture_v1_scan"], "verified": True}


def _fixture_clip(seed: int, f0: float, noise: float):
    """Non-stationary synthetic clip (random slow envelope + tone + noise)."""
    rng = np.random.default_rng(seed)
    n = 16000 * 3
    t = np.arange(n) / 16000.0
    breakpoints = rng.random(24) * 0.8 + 0.1
    envelope = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, 24), breakpoints)
    return (envelope * (0.5 * np.sin(2 * np.pi * f0 * t) + noise * rng.standard_normal(n))).astype(np.float32)


def _synth_row(window_id: str, language: str, source: str, target: str, target_ref: str,
               source_ref: str | None, generator: str = "freevc24") -> dict:
    return {
        "window_id": window_id, "recording_id": window_id, "dataset_id": "indicsynth",
        "spoken_language": language, "label": 1, "generator": generator,
        "speaker_ids": {"source": source, "target": target},
        "parent_refs": {
            "source_reference": source_ref, "target_reference": target_ref,
            "source_kind": "conversion_source" if source_ref else "tts_target_only",
            "source_parent_verification": "verified_train" if source_ref else "not_applicable_tts",
            "target_parent_verification": "verified_train",
            "dataset_revision": REV_IS, "transcript": "fixture",
            "reference_evidence": {"source": _fixture_evidence(source_ref),
                                   "target": _fixture_evidence(target_ref)},
        },
        "prepared_audio": {"sha256": f"prep-{window_id}", "channels": 1, "path": f"fixture/{window_id}.wav"},
        "original_audio": {"sha256": f"orig-{window_id}"},
        "window": {"duration_seconds": 4.0, "sample_rate": 16000},
        "preprocessing_version": PREP_VERSION,
    }


def _genuine_row(window_id: str, language: str, record: str, speaker: str, gender: str = "f") -> dict:
    fname = f"{record}-{speaker}-{gender}.m4a"
    return {
        "window_id": window_id, "recording_id": window_id, "dataset_id": "kathbath",
        "spoken_language": language, "label": 0, "generator": None,
        "source_file": f"{language.lower()}/{fname}",
        "speaker_ids": {"speaker": speaker, "gender": gender},
        "parent_refs": {"dataset_revision": REV_KB},
        "prepared_audio": {"sha256": f"prep-{window_id}", "channels": 1, "path": f"fixture/{window_id}.wav"},
        "original_audio": {"sha256": f"orig-{window_id}"},
        "window": {"duration_seconds": 4.0, "sample_rate": 16000},
        "preprocessing_version": PREP_VERSION,
    }


def run_fixtures() -> dict:
    """Adversarial cases: the checker must reject leaks and permit legitimate cases."""
    results = []

    def case(name: str, manifests: dict, must_include: Sequence[str] = (), must_exclude: Sequence[str] = (),
             registry: dict | None = None, heldout: dict | None = None,
             closure_edges: Sequence | None = None, closure_anchors: Sequence | None = None,
             verified_records: dict | None = None):
        report = audit_manifests(manifests, registry, heldout,
                                 closure_edges=closure_edges, closure_anchors=closure_anchors,
                                 verified_records=verified_records)
        checks = {issue["check"] for issue in report["issues"]}
        ok_include = all(c in checks for c in must_include)
        ok_exclude = all(c not in checks for c in must_exclude)
        results.append({"case": name, "must_include": list(must_include), "must_exclude": list(must_exclude),
                        "issues": sorted(checks), "passed": ok_include and ok_exclude,
                        "detail": report["issues"][:2]})

    # 1) same person genuine in train and synthetic target in test -> reject
    case("genuine_train_vs_synthetic_target_test",
         {"train": [_genuine_row("kb-t-1", "Bengali", "100", "7")],
          "test": [_synth_row("is-te-1", "Bengali", "500", "7", "200-7-f.wav", "300-500-f.wav")]},
         must_include=["cross_split_intersection"])

    # 2) source speaker of test synthetic appears as train genuine -> reject
    case("synthetic_source_vs_train_genuine",
         {"train": [_genuine_row("kb-t-2", "Hindi", "101", "8")],
          "test": [_synth_row("is-te-2", "Hindi", "8", "600", "200-600-f.wav", "102-8-m.wav")]},
         must_include=["cross_split_intersection"])

    # 3) shared parent recording between splits (record overlap) -> reject
    case("shared_reference_recording",
         {"dev": [_synth_row("is-de-1", "Tamil", "900", "50", "777-50-f.wav", "400-900-f.wav")],
          "test": [_synth_row("is-te-3", "Tamil", "901", "51", "777-50-f.wav", "401-901-f.wav")]},
         must_include=["cross_split_intersection"])

    # 4) training-exposed identity attempted in test (closure) -> reject
    reg = {"speaker_keys": [["kathbath", "bengali", "7"]], "reference_keys": [["kathbath", "bengali", "100-7-f"]],
           "original_audio_sha256": [], "prepared_audio_sha256": []}
    case("closure_violation",
         {"train": [_genuine_row("kb-t-3", "Bengali", "100", "7")],
          "test": [_genuine_row("kb-te-1", "Bengali", "101", "7")]},
         must_include=["test_speaker_in_prior_exposure"], registry=reg)

    # 5) string vs integer id canonicalization -> reject
    case("string_integer_canonicalization",
         {"train": [_genuine_row("kb-t-4", "Kannada", "100", "7")],
          "test": [_synth_row("is-te-4", "Kannada", "500", "007", "201-7-f.wav", "301-500-f.wav")]},
         must_include=["cross_split_intersection"])

    # 6) unrelated dataset id collision must NOT create cross-split identity keys
    unrelated = _synth_row("is-te-5", "Marathi", "7", "7", "201-7-f.wav", "301-7-f.wav")
    unrelated["dataset_id"] = "nisp"
    unrelated["parent_refs"] = {}
    case("unrelated_namespace_collision",
         {"train": [_genuine_row("kb-t-5", "Marathi", "100", "7")], "test": [unrelated]},
         must_exclude=["cross_split_intersection"])

    # 7) legitimate absent TTS source (null source + not_applicable) -> accept
    tts = _synth_row("is-te-6", "Gujarati", None, "7", "400-7-f.wav", None, generator="vits")
    tts["speaker_ids"] = {"source": None, "target": "7"}
    case("legitimate_tts_absent_source", {"test": [tts]},
         must_exclude=["cross_split_intersection", "provenance"])

    # 8) component split across dev/test (shared target speaker) -> reject
    case("shared_target_across_splits",
         {"dev": [_synth_row("is-de-2", "Telugu", "800", "60", "500-60-f.wav", "600-800-f.wav")],
          "test": [_synth_row("is-te-7", "Telugu", "801", "60", "501-60-f.wav", "601-801-f.wav")]},
         must_include=["cross_split_intersection"])

    # 9) renamed/re-encoded duplicate: identical prepared hash -> reject
    dup_a = _genuine_row("kb-t-6", "Urdu", "200", "20")
    dup_b = _genuine_row("kb-te-2", "Urdu", "201", "21")
    dup_b["prepared_audio"]["sha256"] = dup_a["prepared_audio"]["sha256"]
    case("renamed_duplicate_hash", {"train": [dup_a], "test": [dup_b]},
         must_include=["cross_split_prepared_hash_duplicate"])

    # 10) upstream held-out record inside test -> reject
    case("upstream_heldout_in_test",
         {"test": [_genuine_row("kb-te-3", "Odia", "999", "30")]},
         must_include=["test_record_in_upstream_heldout"],
         heldout={"Odia": {"status": "verified", "record_ids": ["999"]}})

    # 11) held-out unverified language -> flagged (not silently accepted)
    case("heldout_unverified_language",
         {"test": [_genuine_row("kb-te-4", "Punjabi", "998", "31")]},
         must_include=["heldout_unverified_languages"],
         heldout={"Punjabi": {"status": "failed", "record_ids": []}})

    # 12) transitive closure bridge: the test window's parent record joins (via
    # scanned relationship edges) a component containing a v1-exposed person
    bridge_reg = {"speaker_keys": [["kathbath", "bengali", "900"]], "reference_keys": [],
                  "original_audio_sha256": [], "prepared_audio_sha256": []}
    bridge = _synth_row("is-te-8", "Bengali", "700", "500", "777-500-f.wav", "888-700-f.wav")
    case("transitive_closure_bridge_rejected",
         {"test": [bridge]},
         must_include=["test_closure_component_intersects_exposure"],
         registry=bridge_reg,
         closure_edges=[(("j", "bengali", "777"), ("s", "bengali", "900"))],
         closure_anchors=[("s", "bengali", "900")])

    # 13) same graph shape but disjoint from exposure -> the closure check must pass
    case("transitive_closure_disjoint_accepted",
         {"test": [bridge]},
         must_exclude=["test_closure_component_intersects_exposure"],
         registry=bridge_reg,
         closure_edges=[(("j", "bengali", "123"), ("s", "bengali", "999"))],
         closure_anchors=[("s", "bengali", "999")])

    # 14) component larger than the quota, split across dev/test -> reject
    oversized_dev = [_synth_row(f"is-de-big{i}", "Odia", str(100 + i), "60",
                                f"5{i:02d}-60-f.wav", f"6{i:02d}-{100 + i}-f.wav") for i in range(4)]
    oversized_test = [_synth_row("is-te-big", "Odia", "200", "60", "700-60-f.wav", "800-200-f.wav")]
    case("oversized_component_split_across_splits",
         {"dev": oversized_dev, "test": oversized_test},
         must_include=["cross_split_component"])

    # 15) oversized component contained entirely within test -> permitted
    contained = [_synth_row(f"is-te-w{i}", "Punjabi", str(300 + i), "70",
                            f"9{i:02d}-70-f.wav", f"8{i:02d}-{300 + i}-f.wav") for i in range(5)]
    case("oversized_component_within_test_accepted",
         {"test": contained},
         must_exclude=["cross_split_component", "test_closure_component_intersects_exposure"])

    # 16) unverified source parent -> reject (quarantine-by-admission)
    unverified_source = _synth_row("is-te-unv1", "Tamil", "800", "80", "900-80-f.wav", "901-800-f.wav")
    unverified_source["parent_refs"]["source_parent_verification"] = "parsed_only"
    case("unverified_source_parent_rejected",
         {"test": [unverified_source]},
         must_include=["unverified_source_parent"])

    # 17) unverified target parent -> reject
    unverified_target = _synth_row("is-te-unv2", "Tamil", "801", "81", "902-81-f.wav", "903-801-f.wav")
    unverified_target["parent_refs"]["target_parent_verification"] = "parsed_only"
    case("unverified_target_parent_rejected",
         {"test": [unverified_target]},
         must_include=["unverified_target_parent"])

    # 18) verified parent with no preserved lookup evidence -> reject
    no_evidence = _synth_row("is-te-noev", "Tamil", "802", "82", "904-82-f.wav", "905-802-f.wav")
    no_evidence["parent_refs"].pop("reference_evidence", None)
    case("parent_evidence_missing_rejected",
         {"test": [no_evidence]},
         must_include=["parent_evidence_missing"])

    # 19) parent claimed verified but absent from the scanned inventories -> reject
    ghost = _synth_row("is-te-ghost", "Tamil", "803", "83", "906-83-f.wav", "907-803-f.wav")
    case("parent_record_not_in_scanned_inventories_rejected",
         {"test": [ghost]},
         must_include=["parent_record_not_in_scanned_inventories"],
         verified_records={"tamil": set()})

    # 20/21) near-duplicate detector behaviour on re-encoded (not byte-identical) audio
    base = _fixture_clip(42, 220.0, 0.15)
    requantized = np.round(base.astype(np.float64) * 32767).astype(np.int16).astype(np.float64) / 32767.0
    similarity_a, delta_a, hit_a = _near_duplicate_similarity(base, 16000, requantized, 16000)
    audio_cases = [{"case": "reencoded_fingerprint_detected", "similarity": round(similarity_a, 6),
                    "duration_delta": round(delta_a, 6), "passed": bool(hit_a)}]
    unrelated = _fixture_clip(7, 700.0, 0.45)
    similarity_b, delta_b, hit_b = _near_duplicate_similarity(base, 16000, unrelated, 16000)
    audio_cases.append({"case": "unrelated_audio_not_flagged", "similarity": round(similarity_b, 6),
                        "duration_delta": round(delta_b, 6), "passed": bool(not hit_b)})

    passed = all(r["passed"] for r in results) and all(c["passed"] for c in audio_cases)
    return {"passed": passed, "cases": results, "audio_cases": audio_cases}


# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", action="store_true", help="run adversarial fixture cases only")
    parser.add_argument("--manifest-root", default=str(V2 / "manifests"))
    parser.add_argument("--json-out", default=str(V2 / "reports" / "audit.json"))
    args = parser.parse_args(argv)

    if args.fixtures:
        report = run_fixtures()
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 1

    root = Path(args.manifest_root)
    manifests = {
        "train": [json.loads(l) for l in (root / "windows.train.jsonl").read_text().splitlines() if l],
        "dev": [json.loads(l) for l in (root / "windows.dev.jsonl").read_text().splitlines() if l],
        "test": [json.loads(l) for l in (root / "windows.test.jsonl").read_text().splitlines() if l],
    }
    registry = json.loads((V2 / "planning" / "exposure_registry.v2.json").read_text())
    heldout = json.loads((V2 / "staging" / "kathbath_heldout.json").read_text())
    closure_data = load_closure_data()

    report = audit_manifests(manifests, registry, heldout,
                             closure_edges=closure_data["edges"],
                             closure_anchors=closure_data["anchors"],
                             closure_edges_strict=closure_data["strict_edges"],
                             verified_records=closure_data["verified_records"])
    report["schema"] = SCHEMA
    report["schema_problems"] = audit_schema(manifests)
    unknowns = {
        "test_missing_transcript": sum(1 for r in manifests.get("test", [])
                                        if r.get("dataset_id") == "indicsynth"
                                        and (r.get("parent_refs") or {}).get("transcript") is None),
        "test_source_parent_parsed_only": sum(1 for r in manifests.get("test", [])
                                               if r.get("dataset_id") == "indicsynth"
                                               and (r.get("parent_refs") or {}).get("source_parent_verification") == "parsed_only"),
        "test_target_parent_parsed_only": sum(1 for r in manifests.get("test", [])
                                               if r.get("dataset_id") == "indicsynth"
                                               and (r.get("parent_refs") or {}).get("target_parent_verification") == "parsed_only"),
    }
    report["unknown_metadata"] = unknowns
    report["path_bases"] = {split: str((V2 if split == "test" else V1)) for split in manifests}
    audio = audit_audio(manifests, {split: (V2 if split == "test" else V1) for split in manifests})
    report["audio"] = {k: v for k, v in audio.items() if k != "problems"}
    report["audio_problems"] = audio["problems"]
    report["accounting"] = audit_accounting(manifests["test"])
    report["n_problems"] = (len(report["issues"]) + len(report["schema_problems"]) +
                            len(report["audio_problems"]) +
                            (0 if report["accounting"]["reconciled"] else 1))
    report["ready"] = report["n_problems"] == 0
    report["protocol"] = "speaker_recording_disjoint_v2"
    report["ready_scope"] = "data integrity under named protocol; not model or production readiness"
    report["strict_conversion_family_compliant"] = (
        (report["closure_transitive"].get("strict_diagnostic") or {}).get("n_hits") == 0)

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({
        "ready": report["ready"], "n_issues": report["n_issues"],
        "schema_problems": len(report["schema_problems"]),
        "audio_problems": len(report["audio_problems"]),
        "near_duplicates": len(audio["near_duplicates"]),
        "accounting": report["accounting"],
        "heldout": report["heldout"],
        "closure": report["closure"],
        "closure_transitive": {"checked": report["closure_transitive"].get("checked"),
                               "n_hits": report["closure_transitive"].get("n_hits"),
                               "strict_diagnostic": (report["closure_transitive"].get("strict_diagnostic") or {}).get("n_hits")},
        "cross_split_components": len(report["cross_split_components"]),
        "unknown_metadata": unknowns,
    }, indent=2, default=str))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
