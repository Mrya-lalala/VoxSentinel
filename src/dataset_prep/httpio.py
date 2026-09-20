"""HTTP helpers: preflight sizes, bounded reads, resumable atomic downloads.

Downloads write to ``<name>.part`` and are atomically renamed on completion.
A partially downloaded ``.part`` file is resumed with an HTTP Range request when
the server supports it, otherwise it is restarted.  Every transferred byte is
charged to the :class:`~src.dataset_prep.budget.DownloadLedger`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import requests

from .budget import BudgetExceeded, DownloadLedger

USER_AGENT = "VoxSentinel-dataset-prep/1.0 (bounded research pilot)"


class HttpError(RuntimeError):
    pass


@dataclass(frozen=True)
class DownloadResult:
    """Outcome of a resumable download: final path plus newly transferred bytes."""

    path: Path
    transferred_bytes: int


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def head_size(url: str, *, timeout: float = 60.0) -> int | None:
    """Return Content-Length without downloading, if the server reports it."""
    with _session() as session:
        response = session.head(url, allow_redirects=True, timeout=timeout)
        response.raise_for_status()
        length = response.headers.get("Content-Length")
        return int(length) if length is not None else None


def get_json(url: str, *, timeout: float = 60.0, retries: int = 5, sleep_seconds: float = 1.0) -> dict:
    """GET a small JSON document with bounded retries (models rate limiting)."""
    last_error: Exception | None = None
    with _session() as session:
        for attempt in range(1, retries + 1):
            try:
                response = session.get(url, timeout=timeout)
                if response.status_code == 429:
                    wait = float(response.headers.get("Retry-After", sleep_seconds * attempt * 2))
                    time.sleep(min(max(wait, 2.0), 60.0))
                    last_error = requests.HTTPError(f"HTTP 429 (rate limited) from {url}")
                    continue
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as error:
                last_error = error
                time.sleep(min(sleep_seconds * attempt, 15.0))
    raise HttpError(f"GET {url} failed after {retries} attempts: {last_error}")


def get_bytes(
    url: str,
    *,
    timeout: float = 60.0,
    retries: int = 3,
    headers: dict[str, str] | None = None,
    max_bytes: int | None = None,
) -> bytes:
    """GET a bounded byte payload (single audio file, protocol file, page)."""
    last_error: Exception | None = None
    with _session() as session:
        for attempt in range(1, retries + 1):
            try:
                response = session.get(url, timeout=timeout, headers=headers)
                if response.status_code == 429:
                    time.sleep(min(2.0 * attempt, 20.0))
                    continue
                response.raise_for_status()
                payload = response.content
                if max_bytes is not None and len(payload) > max_bytes:
                    raise HttpError(f"GET {url} returned {len(payload)} bytes, above the {max_bytes} limit.")
                return payload
            except requests.RequestException as error:
                last_error = error
                time.sleep(min(2.0 * attempt, 20.0))
    raise HttpError(f"GET {url} failed after {retries} attempts: {last_error}")


def download_resumable(
    url: str,
    destination: str | Path,
    *,
    ledger: DownloadLedger | None = None,
    source_id: str = "unknown",
    timeout: float = 120.0,
    progress: Callable[[int, int | None], None] | None = None,
    max_bytes: int | None = None,
) -> DownloadResult:
    """Download ``url`` to ``destination`` and return a :class:`DownloadResult`.

    Resumes ``.part`` files via Range when the server supports it and skips a
    destination that is already complete.  Raises :class:`BudgetExceeded`
    *before* transferring a byte if the declared or remaining size does not fit
    the budget.
    """
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")

    declared: int | None = None
    if target.exists():
        try:
            declared = head_size(url, timeout=timeout)
        except requests.RequestException:
            declared = None
        if declared is not None and target.stat().st_size == declared:
            # Already fully downloaded in an earlier run; nothing new to charge.
            if progress is not None:
                progress(0, declared)
            return DownloadResult(path=target, transferred_bytes=0)

    with _session() as session:
        existing = part.stat().st_size if part.exists() else 0
        headers: dict[str, str] = {}
        if existing:
            headers["Range"] = f"bytes={existing}-"
        response = session.get(url, stream=True, timeout=timeout, headers=headers)
        if existing and response.status_code == 200:
            # Server ignored the range; restart cleanly.
            response.close()
            part.unlink(missing_ok=True)
            existing = 0
            response = session.get(url, stream=True, timeout=timeout)
        if existing and response.status_code != 206:
            response.close()
            raise HttpError(f"resume of {url} returned HTTP {response.status_code}, expected 206.")
        if response.status_code not in (200, 206):
            response.close()
            raise HttpError(f"GET {url} returned HTTP {response.status_code}.")
        total_header = response.headers.get("Content-Length")
        declared = int(total_header) + existing if total_header is not None else None
        if max_bytes is not None and declared is not None and declared > max_bytes:
            response.close()
            raise HttpError(f"{url} declares {declared} bytes, above the {max_bytes}-byte limit.")
        if ledger is not None:
            expected = (declared - existing) if declared is not None else None
            if expected is not None and not ledger.affordable(expected):
                response.close()
                raise BudgetExceeded(
                    f"downloading {url} needs {expected} more bytes; only {ledger.remaining_bytes} remain."
                )

        mode = "ab" if existing else "wb"
        transferred = 0
        with part.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if not chunk:
                    continue
                if ledger is not None and not ledger.affordable(len(chunk)):
                    handle.flush()
                    response.close()
                    if transferred:
                        # Keep the partial file for a later resume; charge what moved.
                        ledger.charge(source_id, transferred)
                    raise BudgetExceeded(
                        f"download budget reached while streaming {url}; "
                        f"partial file kept at {part} for resume."
                    )
                handle.write(chunk)
                transferred += len(chunk)
        response.close()

    if ledger is not None and transferred:
        ledger.charge(source_id, transferred)
    if progress is not None:
        progress(transferred, declared - existing if declared is not None else None)

    part.replace(target)
    return DownloadResult(path=target, transferred_bytes=transferred)


__all__ = ["BudgetExceeded", "DownloadResult", "HttpError", "download_resumable", "get_bytes", "get_json", "head_size"]
