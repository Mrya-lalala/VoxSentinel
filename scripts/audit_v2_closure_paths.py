"""Independent reproduction + edge-provenance audit of the reviewer's closure finding.

The review reports that a union-find over parsed source/target speaker references in
all 24 old+fresh per-language IndicSynth candidate files connects 67 of 96 v2 test
windows to a v1-exposed identity component. This script:

1. rebuilds that graph under several clearly-labeled semantic variants so the
   reviewer's number can be matched (or shown not to match),
2. walks the shortest connecting path for every flagged test window,
3. records the *provenance of every edge* on those paths (file, row, fields,
   parsed vs declared speakers, verification status of referenced recordings),

so the relationships can be verified before any window is treated as confirmed
leakage. Read-only with respect to all dataset artifacts; writes one audit JSON
under artifacts/datasets-v2/reports/.

Usage:
    ./.venv-data/bin/python -m scripts.audit_v2_closure_paths
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict, deque
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.gru_identity_audit import reference as parse_reference  # noqa: E402
from scripts.validate_dataset_v2 import row_identities  # noqa: E402
from src.dataset_prep.splitting import UnionFind  # noqa: E402

V1_JOINT = REPO / "artifacts" / "datasets" / "staging" / "joint"
V2_STAGING = REPO / "artifacts" / "datasets-v2" / "staging"
V2_MANIFESTS = REPO / "artifacts" / "datasets-v2" / "manifests"
V2_PLANNING = REPO / "artifacts" / "datasets-v2" / "planning"

LANGUAGES = ["Bengali", "Gujarati", "Hindi", "Kannada", "Malayalam", "Marathi",
             "Odia", "Punjabi", "Sanskrit", "Tamil", "Telugu", "Urdu"]


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def old_row_keys(row: dict) -> dict:
    """Identity keys an old v1 candidate row claims (declared + recorded sides)."""
    out = {"declared_source": None, "declared_target": None,
           "source_ref": None, "target_ref": None, "source_speaker": None, "target_speaker": None}
    ds = row.get("declared_source_speaker")
    dt = row.get("declared_target_speaker")
    out["declared_source"] = str(int(float(str(ds)))) if ds is not None and str(ds).strip() not in ("", "None") else None
    out["declared_target"] = str(int(float(str(dt)))) if dt is not None and str(dt).strip() not in ("", "None") else None
    src = parse_reference(row.get("source_reference"))
    tgt = parse_reference(row.get("target_reference"))
    if src:
        out["source_ref"] = src["canonical"]
        out["source_speaker"] = src["speaker"]
    if tgt:
        out["target_ref"] = tgt["canonical"]
        out["target_speaker"] = tgt["speaker"]
    # recording-speaker columns, if present and different from the parsed side
    rs = row.get("source_recording_speaker")
    rt = row.get("target_recording_speaker")
    if rs is not None and str(rs).strip() not in ("", "None"):
        out["source_recording_speaker"] = str(int(float(str(rs))))
    if rt is not None and str(rt).strip() not in ("", "None"):
        out["target_recording_speaker"] = str(int(float(str(rt))))
    return out


def fresh_row_keys(row: dict) -> dict:
    out = {"source_ref": None, "target_ref": None, "source_speaker": None, "target_speaker": None}
    src = parse_reference(row.get("source_reference"))
    tgt = parse_reference(row.get("target_reference"))
    if src:
        out["source_ref"] = src["canonical"]
        out["source_speaker"] = src["speaker"]
    if tgt:
        out["target_ref"] = tgt["canonical"]
        out["target_speaker"] = tgt["speaker"]
    if row.get("source_speaker") is not None:
        out["declared_source"] = str(int(float(str(row["source_speaker"]))))
    if row.get("target_speaker") is not None:
        out["declared_target"] = str(int(float(str(row["target_speaker"]))))
    return out


def s_node(lang: str, speaker: str) -> tuple:
    return ("s", lang, speaker)


def r_node(lang: str, canonical: str) -> tuple:
    return ("r", lang, canonical)


def j_node(lang: str, record: str) -> tuple:
    return ("j", lang, record)


def build_graph(language: str, files: list[tuple[str, Path]], variant: str):
    """Union-find + edge log for one language under a named semantic variant.

    variant:
      "row_union"  - all identity keys of one candidate row are one component (cross-side union)
      "side_only"  - source side and target side unioned independently; sides join only
                     through shared keys across rows
    """
    uf = UnionFind()
    edges: list[dict] = []
    lang = language.lower()

    def add_edge(a, b, meta):
        uf.union(a, a)
        uf.union(b, b)
        uf.union(a, b)
        edges.append({"a": a, "b": b, **meta})

    for fname, path in files:
        rows = load_rows(path)
        for row in rows:
            keys = old_row_keys(row) if fname.startswith("old:") else fresh_row_keys(row)
            row_ix = int(row.get("row_index", -1))
            base = {"file": fname, "row_index": row_ix, "generator": row.get("generator")}
            src_keys = []
            tgt_keys = []
            # speaker identity claims (declared + recording-derived + parsed-from-ref)
            for field in ("declared_source", "source_recording_speaker", "source_speaker"):
                v = keys.get(field)
                if v is not None:
                    src_keys.append(s_node(lang, v))
            for field in ("declared_target", "target_recording_speaker", "target_speaker"):
                v = keys.get(field)
                if v is not None:
                    tgt_keys.append(s_node(lang, v))
            # reference identities
            src_keys_b = list(src_keys)
            tgt_keys_b = list(tgt_keys)
            if keys.get("source_ref") is not None:
                canon = keys["source_ref"]
                src_keys_b.append(r_node(lang, canon))
                src_keys_b.append(j_node(lang, canon.split("-")[0]))
            if keys.get("target_ref") is not None:
                canon = keys["target_ref"]
                tgt_keys_b.append(r_node(lang, canon))
                tgt_keys_b.append(j_node(lang, canon.split("-")[0]))
            # record the union edges with provenance
            for group, label in ((src_keys_b, "source"), (tgt_keys_b, "target")):
                for i in range(1, len(group)):
                    add_edge(group[0], group[i], {**base, "side": label})
            if variant == "row_union":
                all_keys = src_keys_b + tgt_keys_b
                for i in range(1, len(all_keys)):
                    add_edge(all_keys[0], all_keys[i], {**base, "side": "cross"})
    return uf, edges


def exposure_nodes(registry: dict, language: str, use_refs: bool) -> set:
    lang = language.lower()
    out = set()
    for ds, rlang, speaker in registry.get("speaker_keys", []):
        if str(rlang).lower() == lang:
            out.add(s_node(lang, str(int(float(str(speaker))))))
    if use_refs:
        for ds, rlang, canon in registry.get("reference_keys", []):
            if str(rlang).lower() == lang:
                canon = str(canon)
                out.add(r_node(lang, canon))
                out.add(j_node(lang, canon.split("-")[0]))
    return out


def window_nodes(row: dict) -> list[tuple]:
    ids = row_identities(row)
    lang = ids["language"]
    out = []
    for role, key in ids["speakers"].items():
        out.append(s_node(lang, key[2]))
    for role, key in ids["references"].items():
        out.append(r_node(lang, key[2]))
    for key in ids["records"]:
        out.append(j_node(lang, key[2]))
    return out


def bfs_path(adjacency: dict, starts: list[tuple], targets: set) -> list[dict] | None:
    """Shortest path from any start node to any target node; returns list of edge dicts."""
    prev: dict[tuple, tuple | None] = {s: None for s in starts}
    queue = deque(starts)
    found = None
    while queue:
        node = queue.popleft()
        if node in targets:
            found = node
            break
        for other, meta in adjacency.get(node, ()):
            if other not in prev:
                prev[other] = (node, meta)
                queue.append(other)
    if found is None:
        return None
    steps = []
    cur = found
    while prev[cur] is not None:
        node, meta = prev[cur]
        steps.append({"from": list(node), "to": list(cur), **meta})
        cur = node
    steps.reverse()
    return steps


def main() -> int:
    registry = json.loads((V2_PLANNING / "exposure_registry.v2.json").read_text())
    test_rows = load_rows(V2_MANIFESTS / "windows.test.jsonl")
    print(f"test windows: {len(test_rows)}")

    results = {}
    for variant in ("row_union", "side_only"):
        for use_refs in (False, True):
            for file_set in ("all", "fresh_only"):
                flagged_total = 0
                per_language = {}
                for language in LANGUAGES:
                    files = []
                    if file_set == "all":
                        files.append((f"old:indicsynth_candidates.{language}.jsonl",
                                      V1_JOINT / f"indicsynth_candidates.{language}.jsonl"))
                    files.append((f"fresh:indicsynth_fresh.{language}.jsonl",
                                  V2_STAGING / f"indicsynth_fresh.{language}.jsonl"))
                    uf, edges = build_graph(language, files, variant)
                    exposure = exposure_nodes(registry, language, use_refs)
                    # components that contain any exposure node
                    poisoned_roots = {uf.find(n) for n in exposure}
                    flagged = []
                    for row in test_rows:
                        if str(row.get("spoken_language", "")).lower() != language.lower():
                            continue
                        nodes = window_nodes(row)
                        if any(uf.find(n) in poisoned_roots for n in nodes):
                            flagged.append(row["window_id"])
                    flagged_total += len(flagged)
                    per_language[language] = len(flagged)
                key = f"{variant}|refs={use_refs}|files={file_set}"
                results[key] = {"flagged": flagged_total, "per_language": per_language}
                print(f"{key}: {flagged_total} flagged | " +
                      ", ".join(f"{l[:3]}={n}" for l, n in per_language.items() if n))

    # Detailed path audit under the most conservative reproduced variant (row_union, refs, all files)
    detail = {"flagged": [], "edge_provenance": {}, "edge_side_counts": {}, "file_counts": {}}
    for language in LANGUAGES:
        files = [(f"old:indicsynth_candidates.{language}.jsonl",
                  V1_JOINT / f"indicsynth_candidates.{language}.jsonl"),
                 (f"fresh:indicsynth_fresh.{language}.jsonl",
                  V2_STAGING / f"indicsynth_fresh.{language}.jsonl")]
        uf, edges = build_graph(language, files, "row_union")
        adjacency = defaultdict(list)
        for e in edges:
            adjacency[e["a"]].append((e["b"], e))
            adjacency[e["b"]].append((e["a"], e))
        exposure = exposure_nodes(registry, language, True)
        poisoned_roots = {uf.find(n) for n in exposure}
        for row in test_rows:
            if str(row.get("spoken_language", "")).lower() != language.lower():
                continue
            nodes = window_nodes(row)
            if not any(uf.find(n) in poisoned_roots for n in nodes):
                continue
            path = bfs_path(adjacency, nodes, exposure)
            record = {"window_id": row["window_id"], "language": language,
                      "dataset": row.get("dataset_id"), "path_len": len(path or [])}
            if path:
                record["path"] = [
                    {"from": step["from"], "to": step["to"], "file": step["file"],
                     "row_index": step["row_index"], "side": step["side"],
                     "generator": step.get("generator")}
                    for step in path
                ]
                for step in path:
                    detail["file_counts"][step["file"]] = detail["file_counts"].get(step["file"], 0) + 1
                    detail["edge_side_counts"][step["side"]] = detail["edge_side_counts"].get(step["side"], 0) + 1
            detail["flagged"].append(record)

    out = {
        "schema": "voxsentinel.v2.closure_path_audit.v1",
        "variants": results,
        "detail_variant": "row_union|refs=True|files=all",
        "n_flagged_detail": len(detail["flagged"]),
        "path_lengths": sorted({r["path_len"] for r in detail["flagged"]}),
        "file_counts": detail["file_counts"],
        "edge_side_counts": detail["edge_side_counts"],
        "flagged": detail["flagged"],
    }
    out_path = V2_PLANNING.parent / "reports" / "closure_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nwrote {out_path}")
    print("path lengths seen:", out["path_lengths"])
    print("edge files:", json.dumps(out["file_counts"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
