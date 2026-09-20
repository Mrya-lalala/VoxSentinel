"""V3 male-genuine coverage acquisition (pinned Kathbath genuine source only).

Scope (per docs/MALE_GENUINE_ACQUISITION_AGENT_PROMPT.md):
- acquire additional documented male genuine speech for train/development,
  preferably 8 train + 4 dev windows per language (soft targets),
  from >= 4 train speakers and >= 2 dev speakers, <= 2 additions per speaker per split;
- retain all existing female genuine and synthetic examples unchanged;
- exclude every evaluated-test identity/recording (benchmark exposure), all v1
  prior exposure and upstream held-out participants;
- protocol: ``speaker_recording_disjoint_v2`` (identity/recording disjoint;
  conversion co-participation chains through unused candidates are NOT edges);
- same shared ledger; hardened serial pre-charged reads for every Kathbath byte;
- write everything under ``artifacts/datasets-v3-coverage/``.

Commands (see scripts/prepare_dataset_v3_coverage.py):
  snapshot   -> frozen-artifact hashes before mutation
  plan       -> metadata-only feasibility + deterministic frozen selection
  materialize-> bounded fetches + prep-2 windows + dispositions
  finalize   -> combined train/dev manifests + dataset-version.v3.json
  replay     -> offline recomputation + manifest hash comparison
  verify     -> frozen-artifact preservation check
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scripts.gru_identity_audit import reference as parse_reference
from scripts.validate_dataset_v2 import row_identities

from .budget import BudgetExceeded
from .materialize import HashRegistry, materialize_window, sha256_file, store_original_bytes
from .records import build_recording_record, build_window_record, read_json, read_jsonl, write_json_atomic, write_jsonl_atomic
from .splitting import UnionFind
from .split_v2 import (
    CORE_LANGUAGES,
    HF_DATASET,
    INDICSYNTH_REVISION,
    KATHBATH_REVISION,
    LANGUAGE_SLUGS,
    MIN_RECORDING_DURATION,
    SEED_DEFAULT,
    _candidate_id,
    _closure_edges,
    _closure_node,
    _load_fresh_state,
    fetch_kathbath_audio_pinned,
    v2_cfg,
    v2_ledger,
)

V1 = Path("artifacts/datasets")
V2 = Path("artifacts/datasets-v2")
V3 = Path("artifacts/datasets-v3-coverage")
BENCHMARK = Path("artifacts/evaluations/gru-v2-test-epoch5")

SCHEMA_PLAN = "voxsentinel.v3_coverage.plan.v1"
SCHEMA_VERSION = "voxsentinel.dataset_version.v3-coverage.v1"
SCHEMA_SNAPSHOT = "voxsentinel.v3_coverage.snapshot.v1"
PROTOCOL = "speaker_recording_disjoint_v2"

TRAIN_TARGET = 8
DEV_TARGET = 4
MAX_PER_SPEAKER_PER_SPLIT = 2
TRAIN_SPEAKER_TARGET = 4
DEV_SPEAKER_TARGET = 2

# Planning cap for this task: stay well below the remaining shared-ledger budget
# (990,915,958 bytes at task start) and account metadata reads inside it.
V3_BUDGET_CAP = 750_000_000
FOOTER_ALLOWANCE_PER_SHARD = 16 * 1024 * 1024


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def int_str(value: Any) -> str | None:
    try:
        return str(int(float(str(value))))
    except (TypeError, ValueError):
        return None


def v3_ledger():
    return v2_ledger(v2_cfg())


def ledger_facts(ledger) -> dict[str, Any]:
    return {"cap_bytes": ledger.cap_bytes, "total_bytes": ledger.total_bytes,
            "remaining_bytes": ledger.remaining_bytes, "events": ledger.events,
            "per_source_bytes": dict(sorted(ledger.per_source.items()))}


def _sha(path: Path) -> str | None:
    return sha256_file(path) if path.exists() else None


def planning_dir() -> Path:
    path = V3 / "planning"
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# exclusion closure (v1 exposure + upstream held-out + evaluated benchmark)
# --------------------------------------------------------------------------- #

def load_inputs() -> dict[str, Any]:
    registry = read_json(V2 / "planning" / "exposure_registry.v2.json", default={}) or {}
    heldout = read_json(V2 / "staging" / "kathbath_heldout.json", default={}) or {}
    benchmark = read_json(BENCHMARK / "exposure.json", default={}) or {}
    test_rows = read_jsonl(V2 / "manifests" / "windows.test.jsonl")
    return {"registry": registry, "heldout": heldout, "benchmark": benchmark, "test_rows": test_rows}


def build_exclusion(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Poisoned components over admissible relationships vs all exclusion anchors."""
    finder = UnionFind()
    edge_count = 0
    for language in CORE_LANGUAGES:
        for a, b in _closure_edges(language):
            finder.union(a, b)
            edge_count += 1

    v1_speakers: set[tuple[str, str, str]] = set()
    v1_references: set[tuple[str, str, str]] = set()
    for _dataset, lang, speaker in inputs["registry"].get("speaker_keys", []):
        norm = int_str(speaker)
        if norm is not None:
            v1_speakers.add(_closure_node("s", str(lang), norm))
    for _dataset, lang, canonical in inputs["registry"].get("reference_keys", []):
        canonical = str(canonical)
        v1_references.add(_closure_node("r", str(lang), canonical))
        v1_references.add(_closure_node("j", str(lang), canonical.split("-")[0]))

    benchmark_speakers: set[tuple[str, str, str]] = set()
    benchmark_references: set[tuple[str, str, str]] = set()
    for _dataset, lang, speaker in inputs["benchmark"].get("speaker_keys", []):
        norm = int_str(speaker)
        if norm is not None:
            benchmark_speakers.add(_closure_node("s", str(lang), norm))
    for _dataset, lang, canonical in inputs["benchmark"].get("reference_keys", []):
        canonical = str(canonical)
        benchmark_references.add(_closure_node("r", str(lang), canonical))
        benchmark_references.add(_closure_node("j", str(lang), canonical.split("-")[0]))

    # the complete evaluated manifest: all roles (genuine speaker; synthetic
    # source/target speakers; parent references and records), not just files
    for row in inputs["test_rows"]:
        ids = row_identities(row)
        lang = ids["language"]
        for key in ids["speakers"].values():
            benchmark_speakers.add(("s", lang, key[2]))
        for key in ids["references"].values():
            benchmark_references.add(("r", lang, key[2]))
        for key in ids["records"]:
            benchmark_references.add(("j", lang, key[2]))

    heldout_records: set[tuple[str, str, str]] = set()
    for language, entry in inputs["heldout"].items():
        if not isinstance(entry, Mapping) or entry.get("status") != "verified":
            continue
        for record in entry.get("record_ids", []):
            heldout_records.add(_closure_node("j", str(language), str(record)))

    anchors = v1_speakers | v1_references | benchmark_speakers | benchmark_references | heldout_records
    poisoned_roots = {finder.find(node) for node in anchors}
    memo: dict[tuple, bool] = {}

    def is_poisoned(node: tuple) -> bool:
        hit = memo.get(node)
        if hit is None:
            hit = finder.find(node) in poisoned_roots
            memo[node] = hit
        return hit

    return {"is_poisoned": is_poisoned, "edges": edge_count,
            "v1_speakers": v1_speakers, "v1_references": v1_references,
            "benchmark_speakers": benchmark_speakers, "benchmark_references": benchmark_references,
            "heldout_records": heldout_records,
            "anchor_count": len(anchors)}


def candidate_rows(language: str, exclusion: Mapping[str, Any]) -> tuple[list[dict[str, Any]], Counter]:
    """Male genuine candidates with eligibility verdicts (quarantines counted)."""
    counts: Counter = Counter()
    out: list[dict[str, Any]] = []
    lang = language.lower()
    for row in _load_fresh_state(language):
        raw_gender = str(row.get("gender") or "")
        if raw_gender != "m":
            counts["not_male"] += 1
            continue
        parsed = parse_reference(row.get("fname"))
        if parsed is None:
            counts["fname_unparseable"] += 1
            continue
        record = int_str(row.get("record_id"))
        speaker = int_str(row.get("speaker"))
        if record is None or speaker is None:
            counts["id_unparseable"] += 1
            continue
        if parsed["record"] != record or parsed["speaker"] != speaker or parsed["gender"] != "m":
            counts["identity_conflict_with_fname"] += 1
            continue
        upstream = row.get("upstream_speaker_id")
        if upstream is not None and str(upstream).strip() not in ("", "None"):
            if int_str(upstream) != speaker:
                counts["upstream_speaker_conflict"] += 1
                continue
        if float(row.get("duration") or 0.0) < MIN_RECORDING_DURATION:
            counts["recording_too_short"] += 1
            continue
        entry = {
            "language": language, "record_id": record, "speaker": speaker, "gender": "m",
            "fname": str(row["fname"]), "shard": str(row["shard"]),
            "row_group": int(row["row_group"]), "row_index": int(row["row_index"]),
            "duration": float(row["duration"]), "canonical": parsed["canonical"],
        }
        nodes = [("s", lang, speaker), ("r", lang, parsed["canonical"]), ("j", lang, record)]
        reason = None
        if any(node in exclusion["benchmark_speakers"] or node in exclusion["benchmark_references"] for node in nodes):
            reason = "benchmark_evaluated_test"
        elif any(node in exclusion["v1_speakers"] or node in exclusion["v1_references"] for node in nodes):
            reason = "v1_prior_exposure"
        elif any(node in exclusion["heldout_records"] for node in nodes):
            reason = "upstream_heldout_record"
        elif any(exclusion["is_poisoned"](node) for node in nodes):
            reason = "closure_component_in_prior_exposure"
        if reason:
            counts[f"excluded_{reason}"] += 1
            continue
        out.append(entry)
    return out, counts


# --------------------------------------------------------------------------- #
# row-group cost data (serial, pre-charged footer reads)
# --------------------------------------------------------------------------- #

def read_group_sizes(language: str, shards: Sequence[str], ledger) -> tuple[dict[str, dict[int, int]], int]:
    """Cached compressed audio-column sizes per row group; footer reads are serial and charged."""
    cache_path = planning_dir() / f"group_sizes.{language}.json"
    raw_cache = read_json(cache_path, default={}) or {}
    cached: dict[str, dict[int, int]] = {
        str(shard): {int(g): int(s) for g, s in (sizes or {}).items()}
        for shard, sizes in raw_cache.items()
    }
    charged_total = 0
    missing = [shard for shard in shards if shard not in cached]
    if missing:
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem

        from .bounded_reader import SeekableRangeReader

        charges_path = planning_dir() / "footer_charges.json"
        charges = read_json(charges_path, default={}) or {}
        language_charges = charges.setdefault(language, {})
        for shard in missing:
            before = ledger.total_bytes
            fs = HfFileSystem()
            counted = SeekableRangeReader(
                fs, f"datasets/{HF_DATASET}@{KATHBATH_REVISION}/{shard}",
                ledger, "kathbath", {"remaining": FOOTER_ALLOWANCE_PER_SHARD})
            try:
                parquet = pq.ParquetFile(counted)
                names = list(parquet.schema_arrow.names)
                column_index = names.index("audio_filepath")
                sizes = {g: int(parquet.metadata.row_group(g).column(column_index).total_compressed_size)
                         for g in range(parquet.metadata.num_row_groups)}
            finally:
                counted.close()
            charged = ledger.total_bytes - before
            charged_total += charged
            language_charges[shard] = charged
            cached[shard] = sizes
        cache_path.write_text(json.dumps({s: {str(g): v for g, v in sizes.items()}
                                          for s, sizes in sorted(cached.items())}, indent=2) + "\n")
        charges_path.write_text(json.dumps(charges, indent=2, sort_keys=True) + "\n")
    return {shard: cached[shard] for shard in shards if shard in cached}, charged_total


# --------------------------------------------------------------------------- #
# deterministic allocation + cost-aware recording selection
# --------------------------------------------------------------------------- #

def allocate_speakers(speaker_rows: Mapping[str, list[dict[str, Any]]]) -> dict[str, list[str]]:
    """Split eligible speakers into dev/train identities (never shared)."""
    order = sorted(speaker_rows, key=lambda s: (-len(speaker_rows[s]), int(s)))
    total = len(order)
    if total == 0:
        return {"dev": [], "train": []}
    if total <= 1:
        return {"dev": order[:1], "train": []}
    if total <= 3:
        dev, train = order[:1], order[1:]
    else:
        dev, train = order[:2], order[2:]
    return {"dev": dev[:DEV_SPEAKER_TARGET], "train": train[:TRAIN_SPEAKER_TARGET]}


def pick_recordings(rows: Sequence[Mapping[str, Any]], needed_groups: set,
                    sizes: Mapping[str, Mapping[int, int]], k: int) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    remaining = sorted(rows, key=lambda r: (str(r["shard"]), int(r["row_group"]), int(r["row_index"])))
    for _ in range(k):
        if not remaining:
            break

        def marginal(row: Mapping[str, Any]) -> tuple:
            group = (str(row["shard"]), int(row["row_group"]))
            if group in needed_groups:
                return (0, 0, -float(row["duration"]), int(row["row_index"]))
            size = int(sizes.get(group[0], {}).get(group[1], 10 ** 12))
            return (1, size, -float(row["duration"]), int(row["row_index"]))

        best = min(remaining, key=marginal)
        picked.append(best)
        needed_groups.add((str(best["shard"]), int(best["row_group"])))
        remaining = [r for r in remaining if r is not best]
    return picked


def plan_language(language: str, candidates: Sequence[Mapping[str, Any]] | None = None,
                  exclusion: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Full-target plan for one language (before global budget trimming)."""
    if candidates is None or exclusion is None:
        inputs = load_inputs()
        exclusion = build_exclusion(inputs)
        candidates, _ = candidate_rows(language, exclusion)
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_speaker[str(row["speaker"])].append(dict(row))
    allocation = allocate_speakers(by_speaker)

    allocated = set(allocation["train"]) | set(allocation["dev"])
    shards = sorted({str(r["shard"]) for speaker in allocated for r in by_speaker[speaker]})
    ledger = v3_ledger()
    sizes, footer_charged = read_group_sizes(language, shards, ledger)

    needed_groups: set = set()

    def take(speaker: str, k: int) -> list[dict[str, Any]]:
        return pick_recordings(by_speaker[speaker], needed_groups, sizes, k)

    train_picks = []
    for speaker in allocation["train"]:
        train_picks.extend(take(speaker, MAX_PER_SPEAKER_PER_SPLIT))
    dev_picks = []
    for speaker in allocation["dev"]:
        dev_picks.extend(take(speaker, MAX_PER_SPEAKER_PER_SPLIT))

    audio_bytes = sum(int(sizes.get(shard, {}).get(rg, 0)) for shard, rg in sorted(needed_groups))
    return {
        "language": language,
        "eligible_speakers": len(by_speaker),
        "eligible_recordings": sum(len(v) for v in by_speaker.values()),
        "train_speakers": allocation["train"],
        "dev_speakers": allocation["dev"],
        "train": [dict(r) for r in train_picks],
        "dev": [dict(r) for r in dev_picks],
        "needed_groups": sorted([list(g) for g in needed_groups]),
        "estimated_audio_bytes": audio_bytes,
        "footer_charged_bytes": footer_charged,
        "_group_sizes": sizes,
    }


def shortfalls(plan: Mapping[str, Any]) -> dict[str, int]:
    return {
        "train_windows": max(0, TRAIN_TARGET - len(plan["train"])),
        "dev_windows": max(0, DEV_TARGET - len(plan["dev"])),
        "train_speakers": max(0, TRAIN_SPEAKER_TARGET - len(plan["train_speakers"])),
        "dev_speakers": max(0, DEV_SPEAKER_TARGET - len(plan["dev_speakers"])),
    }


def plan_with_targets(*, languages: Sequence[str] = CORE_LANGUAGES) -> dict[str, Any]:
    """Metadata-only plan for all languages with a global budget trim."""
    inputs = load_inputs()
    exclusion = build_exclusion(inputs)
    ledger = v3_ledger()
    before = ledger_facts(ledger)

    plans: dict[str, Any] = {}
    exclusion_counts: dict[str, dict[str, int]] = {}
    for language in languages:
        candidates, counts = candidate_rows(language, exclusion)
        exclusion_counts[language] = dict(counts)
        plans[language] = plan_language(language, candidates, exclusion)

    def total_est() -> int:
        return sum(p["estimated_audio_bytes"] + p["footer_charged_bytes"] for p in plans.values())

    trim_log: list[dict[str, Any]] = []
    while total_est() > V3_BUDGET_CAP:
        scored = [(p["estimated_audio_bytes"], lang) for lang, p in plans.items() if len(p["train"]) > 0]
        if not scored:
            break
        scored.sort(key=lambda item: (-item[0], item[1]))
        _cost, language = scored[0]
        plan = plans[language]
        dropped_speaker = plan["train_speakers"][-1]
        plan["train_speakers"] = plan["train_speakers"][:-1]
        dropped = [r for r in plan["train"] if str(r["speaker"]) == dropped_speaker]
        plan["train"] = [r for r in plan["train"] if str(r["speaker"]) != dropped_speaker]
        if not plan["train"] and dropped_speaker:
            # keep at least dev; train may become empty and is reported as shortfall
            pass
        # recompute groups/estimates for the reduced plan
        needed = {(str(r["shard"]), int(r["row_group"])) for r in plan["train"] + plan["dev"]}
        plan["needed_groups"] = sorted([list(g) for g in needed])
        plan["estimated_audio_bytes"] = sum(int(plan["_group_sizes"].get(shard, {}).get(rg, 0))
                                            for shard, rg in needed)
        trim_log.append({"language": language, "dropped_train_speaker": dropped_speaker,
                         "dropped_windows": len(dropped), "reason": "global_budget_cap"})

    after = ledger_facts(ledger)
    return {"schema": SCHEMA_PLAN, "seed": SEED_DEFAULT, "protocol": PROTOCOL,
            "languages": plans, "exclusion_counts": exclusion_counts,
            "trim_log": trim_log,
            "budget": {"cap_for_this_task": V3_BUDGET_CAP, "estimated_total": total_est(),
                       "ledger_before": before, "ledger_after_planning": after},
            "targets": {"train_windows": TRAIN_TARGET, "dev_windows": DEV_TARGET,
                        "train_speakers": TRAIN_SPEAKER_TARGET, "dev_speakers": DEV_SPEAKER_TARGET,
                        "max_per_speaker_per_split": MAX_PER_SPEAKER_PER_SPLIT}}


def frozen_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Public (underscore-free) plan data for the frozen coverage_plan.json."""
    def clean(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {k: clean(v) for k, v in value.items() if not str(k).startswith("_")}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value
    return clean(plan)


def plan_digest(plan: Mapping[str, Any]) -> str:
    import hashlib
    canonical = json.dumps(frozen_plan(plan), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def selection_view(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Plan projection excluding point-in-time accounting fields.

    The frozen planning record legitimately embeds facts a zero-charge replay
    can never reproduce: ledger totals at plan time and the one-time footer
    read charges (23 row groups, 5,898,240 bytes). Deterministic selection is
    therefore defined over this projection.
    """
    import copy
    view = copy.deepcopy(frozen_plan(plan))
    budget = view.get("budget")
    if isinstance(budget, dict):
        budget.pop("ledger_before", None)
        budget.pop("ledger_after_planning", None)
        budget.pop("estimated_total", None)
    for entry in (view.get("languages") or {}).values():
        if isinstance(entry, dict):
            entry.pop("footer_charged_bytes", None)
    return view


def selection_digest(plan: Mapping[str, Any]) -> str:
    import hashlib
    canonical = json.dumps(selection_view(plan), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


# --------------------------------------------------------------------------- #
# command: snapshot (frozen-artifact hashes before any mutation)
# --------------------------------------------------------------------------- #

def _frozen_artifacts() -> list[Path]:
    paths: list[Path] = [
        V1 / "manifests" / "windows.core_train.jsonl",
        V1 / "manifests" / "windows.core_val.jsonl",
        V2 / "manifests" / "windows.train.jsonl",
        V2 / "manifests" / "windows.dev.jsonl",
        V2 / "manifests" / "windows.test.jsonl",
        V2 / "manifests" / "dataset-version.v2.json",
        V1 / "features" / "train.pt",
        V1 / "features" / "train.meta.json",
        V1 / "features" / "val.pt",
        V1 / "features" / "val.meta.json",
        Path("artifacts/runs/gru-core-v1/gru_best.pt"),
        Path("artifacts/runs/gru-core-v1/gru_final.pt"),
        Path("artifacts/runs/gru-core-v2-standardized/gru_best.pt"),
        Path("artifacts/runs/gru-core-v2-standardized/gru_final.pt"),
        Path("artifacts/runs/gru-core-v2-standardized/standardizer.pt"),
        Path("artifacts/runs/gru-v2-split-reproduction/gru_best.pt"),
        Path("artifacts/runs/gru-v2-split-reproduction/gru_final.pt"),
        Path("artifacts/runs/gru-v2-split-reproduction/split_contract.json"),
        Path("configs/base.yaml"),
        Path("configs/gru.yaml"),
        Path("configs/gru_standardized.yaml"),
        Path("configs/datasets.yaml"),
    ]
    paths += sorted(BENCHMARK.glob("*.json"))
    return paths


def command_snapshot() -> dict[str, Any]:
    import datetime
    ledger = v3_ledger()
    artifacts = {str(path): _sha(path) for path in _frozen_artifacts()}
    record = {"schema": SCHEMA_SNAPSHOT, "artifacts": artifacts,
              "ledger_before": ledger_facts(ledger),
              "generated": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    out = planning_dir() / "snapshot_before.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"snapshot": str(out), "artifacts": len(artifacts),
                      "ledger": record["ledger_before"]}, indent=2, default=str))
    return record


def command_verify_snapshot() -> dict[str, Any]:
    before = read_json(planning_dir() / "snapshot_before.json", default={}) or {}
    artifacts = before.get("artifacts", {})
    unchanged, changed, missing = [], [], []
    for path, digest in artifacts.items():
        actual = _sha(Path(path))
        if actual is None:
            missing.append(path)
        elif actual == digest:
            unchanged.append(path)
        else:
            changed.append(path)
    record = {"schema": SCHEMA_SNAPSHOT + ".check", "checked": len(artifacts),
              "unchanged": unchanged, "changed": changed, "missing": missing,
              "ledger_note": "shared ledger is the documented mutable exception"}
    out = planning_dir() / "snapshot_after_check.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"ok": not changed and not missing, "checked": len(artifacts),
                      "unchanged": len(unchanged), "changed": changed, "missing": missing}, indent=2))
    return record


# --------------------------------------------------------------------------- #
# command: plan (metadata-only feasibility + frozen selection)
# --------------------------------------------------------------------------- #

def command_plan() -> dict[str, Any]:
    import datetime
    plan = plan_with_targets()
    frozen = frozen_plan(plan)
    digest = plan_digest(plan)

    (planning_dir() / "coverage_plan.json").write_text(json.dumps(frozen, indent=2) + "\n")
    (planning_dir() / "feasibility.json").write_text(json.dumps(
        {"schema": SCHEMA_PLAN, "plan_digest": digest,
         "languages": {lang: {k: v for k, v in entry.items() if not k.startswith("_")}
                       for lang, entry in plan["languages"].items()},
         "exclusion_counts": plan["exclusion_counts"], "trim_log": plan["trim_log"],
         "budget": plan["budget"], "targets": plan["targets"]}, indent=2, default=str) + "\n")

    inputs = load_inputs()
    exclusion = build_exclusion(inputs)
    snapshot = {
        "schema": SCHEMA_PLAN + ".snapshot", "plan_digest": digest,
        "seed": SEED_DEFAULT, "protocol": PROTOCOL,
        "input_hashes": {
            "exposure_registry.v2.json": _sha(V2 / "planning" / "exposure_registry.v2.json"),
            "kathbath_heldout.json": _sha(V2 / "staging" / "kathbath_heldout.json"),
            "benchmark_exposure.json": _sha(BENCHMARK / "exposure.json"),
            "v2_test_manifest": _sha(V2 / "manifests" / "windows.test.jsonl"),
            **{f"kathbath_fresh.{lang}": _sha(V2 / "staging" / f"kathbath_fresh.{lang}.jsonl")
               for lang in CORE_LANGUAGES},
        },
        "exclusion_anchor_counts": {
            "v1_speakers": len(exclusion["v1_speakers"]),
            "v1_references": len(exclusion["v1_references"]),
            "benchmark_speakers": len(exclusion["benchmark_speakers"]),
            "benchmark_references": len(exclusion["benchmark_references"]),
            "heldout_records": len(exclusion["heldout_records"]),
        },
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    (planning_dir() / "candidate_snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n")

    lines = [f"# v3 male-genuine coverage feasibility (metadata only)", "",
             f"Plan digest: `{digest}` · protocol `{PROTOCOL}` · seed {SEED_DEFAULT}", "",
             "| language | eligible spk | recs | train spk | train win | dev spk | dev win | est MB | footer MB | shortfalls |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for lang, entry in plan["languages"].items():
        short = shortfalls(entry)
        short_txt = ", ".join(f"{k}-{v}" for k, v in short.items() if v) or "none"
        lines.append(f"| {lang} | {entry['eligible_speakers']} | {entry['eligible_recordings']} | "
                     f"{len(entry['train_speakers'])} | {len(entry['train'])} | "
                     f"{len(entry['dev_speakers'])} | {len(entry['dev'])} | "
                     f"{entry['estimated_audio_bytes'] / 1e6:.1f} | {entry['footer_charged_bytes'] / 1e6:.2f} | {short_txt} |")
    lines += ["", "## Exclusions (male rows examined)", "",
              "| language | " + " | ".join(sorted({k for c in plan["exclusion_counts"].values() for k in c})) + " |"]
    keys = sorted({k for c in plan["exclusion_counts"].values() for k in c})
    lines.append("| --- | " + " | ".join("---:" for _ in keys) + " |")
    for lang, counts in plan["exclusion_counts"].items():
        lines.append("| " + lang + " | " + " | ".join(str(counts.get(k, 0)) for k in keys) + " |")
    lines += ["", f"Budget cap for this task: {V3_BUDGET_CAP/1e6:.0f} MB estimated; "
              f"estimated total {plan['budget']['estimated_total']/1e6:.1f} MB; "
              f"trims: {len(plan['trim_log'])}"]
    (planning_dir() / "feasibility.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"plan_digest": digest, "estimated_total_bytes": plan["budget"]["estimated_total"],
                      "trims": plan["trim_log"], "ledger_after_planning": plan["budget"]["ledger_after_planning"]},
                     indent=2, default=str))
    return plan


# --------------------------------------------------------------------------- #
# command: materialize (bounded fetches + prep-2 windows)
# --------------------------------------------------------------------------- #

def command_materialize(*, languages: Sequence[str] | None = None) -> dict[str, Any]:
    plan = read_json(planning_dir() / "coverage_plan.json", default=None)
    if not plan:
        raise FileNotFoundError("run plan first")
    cfg = v2_cfg()
    ledger = v3_ledger()
    release_kb = cfg.release("kathbath")
    pre_version = str(cfg.preprocessing.get("config_version"))

    v1_rows = read_jsonl(V1 / "manifests" / "windows.core_train.jsonl") + \
        read_jsonl(V1 / "manifests" / "windows.core_val.jsonl")
    v2_test = read_jsonl(V2 / "manifests" / "windows.test.jsonl")
    registry = HashRegistry.from_records(v1_rows + v2_test)

    additions: list[dict[str, Any]] = []
    dispositions: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    staging = V3 / "staging"
    staging.mkdir(parents=True, exist_ok=True)

    for language in (languages or CORE_LANGUAGES):
        entry = plan["languages"][language]
        slug = LANGUAGE_SLUGS[language]
        v2_raw = V2 / "raw" / "kathbath" / slug
        v3_raw = V3 / "raw" / "kathbath" / slug
        v3_prepared = V3 / "prepared" / "kathbath" / slug
        planned = [dict(r, split="train") for r in entry["train"]] + [dict(r, split="val") for r in entry["dev"]]

        to_fetch = [r for r in planned
                    if not (v2_raw / str(r["fname"])).exists() and not (v3_raw / str(r["fname"])).exists()]
        fetch_info = {"requested": len(to_fetch), "charged_skips": [], "skipped_shards": []}
        if to_fetch:
            allowance = int(entry["estimated_audio_bytes"] * 1.8) + 2_000_000
            allowance = max(4_000_000, min(allowance, max(1_000_000, ledger.remaining_bytes)))
            try:
                payloads, _bytes_by_shard, _groups, skipped, skip_events = fetch_kathbath_audio_pinned(
                    ledger, to_fetch, byte_cap=allowance,
                    cache_root=Path("artifacts/datasets-v3-coverage") / "planning")
                for fname, payload in payloads.items():
                    store_original_bytes(v3_raw / fname, payload)
                fetch_info["fetched"] = len(payloads)
                fetch_info["skipped_shards"] = skipped
                fetch_info["charged_skips"] = skip_events
            except BudgetExceeded as error:
                fetch_info["budget_error"] = str(error)

        # reserve pool for same-speaker replacement (only raw files already present)
        reserves: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for split in ("train", "val"):
            split_rows = [r for r in planned if r["split"] == split]
            for r in split_rows:
                reserves[(split, str(r["speaker"]))].append(r)

        def raw_for(row: Mapping[str, Any]) -> tuple[Path | None, str]:
            if (v2_raw / str(row["fname"])).exists():
                return v2_raw / str(row["fname"]), "cache_v2_readonly"
            if (v3_raw / str(row["fname"])).exists():
                return v3_raw / str(row["fname"]), "fetched_v3"
            return None, "missing"

        language_added = 0
        for row in planned:
            window_id = _candidate_id(language, "genuine", row)
            disposition = {"language": language, "split": row["split"], "speaker": str(row["speaker"]),
                           "record_id": str(row["record_id"]), "fname": str(row["fname"]),
                           "window_id": window_id}
            done = False
            queue = [row] + [r for r in reserves[(row["split"], str(row["speaker"]))] if r is not row]
            for attempt, candidate_row in enumerate(queue):
                raw_path, raw_source = raw_for(candidate_row)
                if raw_path is None:
                    disposition.update({"status": "failed_raw_missing", "attempt": attempt})
                    continue
                prepared_path = v3_prepared / f"{_candidate_id(language, 'genuine', candidate_row)}.wav"
                try:
                    materialized = materialize_window(raw_path, prepared_path, policy=cfg.window)
                except Exception as error:  # noqa: BLE001
                    disposition.update({"status": f"decode_or_window_failed", "attempt": attempt,
                                        "error": f"{type(error).__name__}: {error}"})
                    continue
                duplicate = registry.check(materialized)
                if duplicate:
                    prepared_path.unlink(missing_ok=True)
                    disposition.update({"status": duplicate, "attempt": attempt})
                    continue
                registry.register(materialized)
                staged = {
                    **candidate_row, "window_id": _candidate_id(language, "genuine", candidate_row),
                    "raw_source": raw_source, "split_field": row["split"],
                    "original_audio": {**materialized.decoded.original_facts(), "sha256": materialized.original_sha256},
                    "decoded": materialized.decoded_facts(),
                    "prepared": materialized.prepared_facts(V3),
                    "window": materialized.window_facts(original_rate=materialized.decoded.original_rate),
                }
                additions.append(staged)
                disposition.update({"status": "materialized", "attempt": attempt,
                                    "replacement_for": None if attempt == 0 else row["fname"],
                                    "raw_source": raw_source})
                language_added += 1
                done = True
                break
            if not done and disposition.get("status") == "materialized":
                pass
            dispositions.append(disposition)

        summary[language] = {
            "planned": len(planned), "materialized": language_added,
            "train_windows": sum(1 for d in dispositions if d["language"] == language
                                 and d["split"] == "train" and d.get("status") == "materialized"),
            "dev_windows": sum(1 for d in dispositions if d["language"] == language
                               and d["split"] == "val" and d.get("status") == "materialized"),
            "fetch": fetch_info,
        }
        print(json.dumps({"language": language, **{k: v for k, v in summary[language].items() if k != "fetch"}}))

    write_jsonl_atomic(staging / "additions.jsonl", additions)
    write_jsonl_atomic(staging / "dispositions.jsonl", dispositions)
    (staging / "materialization_summary.json").write_text(json.dumps(
        {"languages": summary, "ledger": ledger_facts(ledger)}, indent=2, default=str) + "\n")
    print(json.dumps({"total_additions": len(additions), "ledger": ledger_facts(ledger)}, indent=2, default=str))
    return {"additions": len(additions), "summary": summary}


# --------------------------------------------------------------------------- #
# command: finalize (combined manifests + dataset-version metadata)
# --------------------------------------------------------------------------- #

def command_finalize() -> dict[str, Any]:
    import datetime
    cfg = v2_cfg()
    release_kb = cfg.release("kathbath")
    ledger = v3_ledger()
    additions = read_jsonl(V3 / "staging" / "additions.jsonl")
    plan = read_json(V3 / "planning" / "coverage_plan.json", default=None)
    if plan is None:
        raise FileNotFoundError("run plan first")
    digest = plan_digest(plan)

    def build_pair(staged: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        language = str(staged["language"])
        slug = LANGUAGE_SLUGS[language]
        window_id = str(staged["window_id"])
        parent_refs = {
            "dataset_revision": KATHBATH_REVISION,
            "shard": staged["shard"], "row_group": int(staged["row_group"]),
            "row_index": int(staged["row_index"]),
            "upstream_duration_seconds": float(staged["duration"]),
            "upstream_split": "train",
        }
        speaker_ids = {"speaker": str(staged["speaker"]), "gender": "m"}
        notes = ("v3 male genuine coverage addition; previously unscanned shard; "
                 f"raw_source={staged['raw_source']}")
        recording = build_recording_record(
            recording_id=window_id, dataset_id="kathbath",
            source_url=str(release_kb.get("source_url")),
            source_file=f"{slug}/{staged['fname']}", original_split="train", label=0,
            spoken_language=language, native_language=None, speaker_ids=speaker_ids,
            generator=None, generator_version=None, parent_refs=parent_refs,
            original_audio=staged["original_audio"], decoded=staged["decoded"],
            license_note=str(release_kb.get("license")), notes=notes)
        window = build_window_record(
            window_id=window_id, recording_id=window_id, dataset_id="kathbath",
            source_url=str(release_kb.get("source_url")),
            source_file=f"{slug}/{staged['fname']}", original_split="train",
            pool="core", split=staged["split_field"], label=0,
            label_source="official Kathbath corpus (genuine speech)",
            spoken_language=language, native_language=None, speaker_ids=speaker_ids,
            generator=None, generator_version=None, parent_refs=parent_refs,
            original_audio=staged["original_audio"], prepared_audio=staged["prepared"],
            window=staged["window"], preprocessing_version="voxsentinel-prep-2",
            license_note=str(release_kb.get("license")), access_status="public_with_terms",
            status="audio_ready", notes=notes)
        for record in (recording, window):
            record["path_base"] = "artifacts/datasets-v3-coverage"
            record["coverage_addition"] = "v3-male-genuine"
            record["raw_source"] = staged["raw_source"]
        return recording, window

    added_windows, added_recordings = [], []
    for staged in sorted(additions, key=lambda s: (str(s["language"]), int(s["speaker"]), int(s["row_index"]))):
        recording, window = build_pair(staged)
        added_windows.append(window)
        added_recordings.append(recording)

    retained_train = read_jsonl(V2 / "manifests" / "windows.train.jsonl")
    retained_dev = read_jsonl(V2 / "manifests" / "windows.dev.jsonl")
    combined_train = retained_train + [w for w in added_windows if w["split"] == "train"]
    combined_dev = retained_dev + [w for w in added_windows if w["split"] == "val"]

    manifests = V3 / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(manifests / "windows.train.jsonl", combined_train)
    write_jsonl_atomic(manifests / "windows.dev.jsonl", combined_dev)
    write_jsonl_atomic(manifests / "additions.train.jsonl", [w for w in added_windows if w["split"] == "train"])
    write_jsonl_atomic(manifests / "additions.dev.jsonl", [w for w in added_windows if w["split"] == "val"])
    write_jsonl_atomic(V3 / "staging" / "additions.recordings.jsonl", added_recordings)

    version = {
        "schema": SCHEMA_VERSION, "seed": SEED_DEFAULT, "protocol": PROTOCOL,
        "strict_conversion_family_compliant": False,
        "split_policy": ("v1 train/development retained byte-identical; v3 adds male genuine windows only, "
                         "from identities disjoint from v1 exposure, upstream held-out and the evaluated benchmark"),
        "revisions": {"kathbath": KATHBATH_REVISION, "indicsynth": INDICSYNTH_REVISION},
        "manifests": {
            "train": {"path": str(manifests / "windows.train.jsonl"),
                      "sha256": _sha(manifests / "windows.train.jsonl"),
                      "windows": len(combined_train), "retained": len(retained_train),
                      "additions": len(combined_train) - len(retained_train)},
            "dev": {"path": str(manifests / "windows.dev.jsonl"),
                    "sha256": _sha(manifests / "windows.dev.jsonl"),
                    "windows": len(combined_dev), "retained": len(retained_dev),
                    "additions": len(combined_dev) - len(retained_dev)},
        },
        "retained_sources": {
            "train": {"path": str(V2 / "manifests" / "windows.train.jsonl"),
                      "sha256": _sha(V2 / "manifests" / "windows.train.jsonl")},
            "dev": {"path": str(V2 / "manifests" / "windows.dev.jsonl"),
                    "sha256": _sha(V2 / "manifests" / "windows.dev.jsonl")},
        },
        "path_resolution": {
            "default_base": "artifacts/datasets",
            "row_field": "path_base",
            "bases": {"artifacts/datasets-v3-coverage": "v3 additions (added rows only)"},
            "rule": "rows without a path_base field resolve prepared_audio.path under default_base; rows carrying "
                    "path_base resolve under that base",
        },
        "benchmark": {
            "manifest": str(V2 / "manifests" / "windows.test.jsonl"),
            "manifest_sha256": _sha(V2 / "manifests" / "windows.test.jsonl"),
            "exposure": str(BENCHMARK / "exposure.json"),
            "exposure_sha256": _sha(BENCHMARK / "exposure.json"),
            "status": "evaluated benchmark — frozen; every participant identity/recording excluded from additions",
        },
        "coverage_plan": {"path": str(planning_dir() / "coverage_plan.json"), "digest": digest},
        "ledger": ledger_facts(ledger),
        "cache_staleness": ("existing feature caches are stale for these expanded manifests; the current trainer "
                            "--split-version guard accepts only the byte-identical v2 copies and must be replaced "
                            "or extended (new cache generation + dataset-aware training) before any retraining"),
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    (manifests / "dataset-version.v3.json").write_text(json.dumps(version, indent=2) + "\n")
    print(json.dumps({"train": version["manifests"]["train"], "dev": version["manifests"]["dev"],
                      "ledger": version["ledger"]}, indent=2, default=str))
    return version


# --------------------------------------------------------------------------- #
# command: replay (offline recomputation + manifest comparison)
# --------------------------------------------------------------------------- #

def command_replay() -> dict[str, Any]:
    stored = read_json(planning_dir() / "coverage_plan.json", default=None)
    if not stored:
        raise FileNotFoundError("run plan first")
    ledger = v3_ledger()
    before = ledger.total_bytes
    recomputed = frozen_plan(plan_with_targets())
    after = ledger.total_bytes

    selection_match = selection_view(stored) == selection_view(recomputed)
    stored_digest = selection_digest(stored)
    recomputed_digest = selection_digest(recomputed)

    def footer_total(plan: Mapping[str, Any]) -> int:
        return sum(int(e.get("footer_charged_bytes") or 0)
                   for e in (plan.get("languages") or {}).values())

    stored_est = int(stored["budget"]["estimated_total"])
    recomputed_est = int(recomputed["budget"]["estimated_total"])
    stored_footer, recomputed_footer = footer_total(stored), footer_total(recomputed)
    # At plan time the estimate included the one-time footer reads; the
    # cache-backed replay performs none.  The identity below must hold exactly.
    state_identity_ok = (stored_est - stored_footer) == (recomputed_est - recomputed_footer)

    version = read_json(V3 / "manifests" / "dataset-version.v3.json", default={}) or {}
    hashes_match = {}
    for split in ("train", "dev"):
        path = Path(version.get("manifests", {}).get(split, {}).get("path", ""))
        hashes_match[split] = bool(path.exists() and _sha(path) == version["manifests"][split]["sha256"])

    materialized = {str(s["window_id"]) for s in read_jsonl(V3 / "staging" / "additions.jsonl")}
    manifest_additions = set()
    for split in ("train", "dev"):
        path = V3 / "manifests" / f"windows.{split}.jsonl"
        if path.exists():
            manifest_additions |= {str(r["window_id"]) for r in read_jsonl(path)
                                   if r.get("coverage_addition") == "v3-male-genuine"}
    additions_match = materialized == manifest_additions

    report = {
        "schema": SCHEMA_PLAN + ".replay",
        # Full equality is expected to be False: the stored record embeds
        # point-in-time ledger facts and one-time planning read charges.
        "full_plan_match": stored == recomputed,
        "selection_match": selection_match,
        "selection_digest_stored": stored_digest,
        "selection_digest_recomputed": recomputed_digest,
        "state_fields": {
            "footer_charged_bytes_stored": stored_footer,
            "footer_charged_bytes_recomputed": recomputed_footer,
            "estimated_total_stored": stored_est,
            "estimated_total_recomputed": recomputed_est,
            "identity_ok": state_identity_ok,
        },
        "plan_digest_stored": plan_digest(stored),
        "replay_charges_bytes": after - before,
        "manifest_hashes_match": hashes_match,
        "additions_match": additions_match,
        "n_additions": len(materialized),
    }
    (V3 / "reports").mkdir(parents=True, exist_ok=True)
    (V3 / "reports" / "replay_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


__all__ = [
    "V1", "V2", "V3", "BENCHMARK", "PROTOCOL", "TRAIN_TARGET", "DEV_TARGET",
    "V3_BUDGET_CAP", "build_exclusion", "candidate_rows", "allocate_speakers",
    "pick_recordings", "plan_language", "plan_with_targets", "shortfalls",
    "read_group_sizes", "load_inputs", "ledger_facts", "v3_ledger", "int_str", "planning_dir",
    "frozen_plan", "plan_digest", "selection_view", "selection_digest",
    "command_snapshot", "command_verify_snapshot", "command_plan", "command_materialize",
    "command_finalize", "command_replay",
]
