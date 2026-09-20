"""Bounded download budget with a persisted ledger.

Every byte transferred for source acquisition passes through
:class:`DownloadLedger.charge`, which refuses charges that would exceed the
configured cap.  The ledger is stored as JSON under the dataset root and is
safe to keep across reruns (it only ever grows).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .records import read_json, write_json_atomic


class BudgetExceeded(RuntimeError):
    """Raised before starting a transfer that would exceed the budget."""


@dataclass
class DownloadLedger:
    path: Path
    cap_bytes: int
    total_bytes: int = 0
    per_source: dict[str, int] = field(default_factory=dict)
    events: int = 0

    @classmethod
    def load(cls, path: str | Path, cap_bytes: int) -> "DownloadLedger":
        file_path = Path(path)
        data: dict[str, Any] = read_json(file_path, default={}) or {}
        # Historically the file stored "per_source_bytes"; accept both keys so a
        # load can never silently drop per-source accounting.
        per_source_raw = data.get("per_source_bytes", data.get("per_source", {})) or {}
        return cls(
            path=file_path,
            cap_bytes=int(cap_bytes),
            total_bytes=int(data.get("total_bytes", 0)),
            per_source={str(k): int(v) for k, v in per_source_raw.items()},
            events=int(data.get("events", 0)),
        )

    @property
    def remaining_bytes(self) -> int:
        return max(0, self.cap_bytes - self.total_bytes)

    def affordable(self, size_bytes: int) -> bool:
        return size_bytes >= 0 and size_bytes <= self.remaining_bytes

    def charge(self, source_id: str, size_bytes: int) -> None:
        """Record transferred bytes; raise before the cap is exceeded.

        The on-disk ledger is re-read first so that two stages running
        concurrently cannot clobber each other's accounting: a long-lived
        process always merges fresh disk state before adding its own delta
        (the remaining reload→save window is milliseconds, not hours).
        """
        if size_bytes < 0:
            raise ValueError("size_bytes must be non-negative.")
        fresh = DownloadLedger.load(self.path, self.cap_bytes)
        if fresh.total_bytes > self.total_bytes:
            self.total_bytes = fresh.total_bytes
            self.per_source = dict(fresh.per_source)
            self.events = fresh.events
        if not self.affordable(size_bytes):
            raise BudgetExceeded(
                f"download budget exhausted: {size_bytes} more bytes would exceed "
                f"the {self.cap_bytes}-byte cap ({self.remaining_bytes} left)."
            )
        self.total_bytes += size_bytes
        self.per_source[source_id] = self.per_source.get(source_id, 0) + size_bytes
        self.events += 1
        self.save()

    def save(self) -> None:
        write_json_atomic(
            self.path,
            {
                "cap_bytes": self.cap_bytes,
                "total_bytes": self.total_bytes,
                "remaining_bytes": self.remaining_bytes,
                "per_source_bytes": self.per_source,
                "events": self.events,
            },
        )


__all__ = ["BudgetExceeded", "DownloadLedger"]
