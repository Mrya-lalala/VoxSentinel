"""Re-hash the v2 pre-preparation snapshot and report unchanged/changed/missing.

The shared acquisition ledger (``artifacts/datasets/manifests/downloads.json``)
is the single documented exception: it is expected to grow as exploration is
charged.  Everything else in the snapshot must be byte-identical.

    .venv-data/bin/python -m scripts.verify_v2_snapshot
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

V2 = REPO / "artifacts" / "datasets-v2"
LEDGER_EXCEPTION = "artifacts/datasets/manifests/downloads.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    snapshot = json.loads((V2 / "planning" / "snapshot_before.json").read_text())
    unchanged, changed, missing, ledger = [], [], [], None
    for rel, expected in sorted(snapshot["frozen_artifacts"].items()):
        path = REPO / rel
        if expected.get("sha256") is None:
            continue
        if not path.exists():
            missing.append(rel)
            continue
        actual = _sha256(path)
        entry = {"path": rel, "before": expected["sha256"], "after": actual}
        if actual == expected["sha256"]:
            unchanged.append(rel)
        elif rel == LEDGER_EXCEPTION:
            ledger = {**entry, "note": "documented exception: ledger grows monotonically with charged exploration"}
        else:
            changed.append(entry)
    report = {
        "schema": "voxsentinel.v2.snapshot_verification.v1",
        "checked": len(unchanged) + len(changed) + len(missing) + (1 if ledger else 0),
        "unchanged": unchanged,
        "changed": changed,
        "missing": missing,
        "ledger_exception": ledger,
        "ok": not changed and not missing,
        "generated": datetime.now(timezone.utc).isoformat(),
    }
    (V2 / "planning" / "snapshot_after_check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "unchanged": len(unchanged), "changed": changed,
                      "missing": missing, "ledger_note": bool(ledger)}, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
