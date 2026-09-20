"""File-like reader over concatenated gzip segments downloaded on demand.

The NISP archives are distributed as ``RECS.tar.gz.a*`` parts of 95 MiB that
must be concatenated *in exact part order*.  The actual layering was verified
against the upstream bytes during preparation: the parts are a byte split of one
**outer gzip stream** whose payload is itself a gzip stream (the README's
``gzip -dc Complete.tar.gz | tar -xz`` implies the double decompression).
This reader therefore decompresses through a configurable chain of gzip layers
(default 2) and exposes the final tar bytes as a plain file-like object.

Parts are pulled only when the previous one is exhausted, every byte is charged
to the download ledger, and the whole stream stops cleanly (reads as EOF) when a
per-group byte ceiling or the global budget is reached.
"""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Callable, Sequence

from ..budget import BudgetExceeded
from ..httpio import DownloadResult, HttpError


class SegmentedGzipReader:
    """Sequential multi-layer decompression across ordered gzip part files."""

    def __init__(
        self,
        parts: Sequence[tuple[str, str, int]],
        *,
        download: Callable[[str, Path], DownloadResult],
        staging_dir: Path,
        group_budget_bytes: int,
        decompress_layers: int = 2,
    ) -> None:
        if not parts:
            raise ValueError("SegmentedGzipReader needs at least one part.")
        if decompress_layers < 1:
            raise ValueError("decompress_layers must be >= 1.")
        self._parts = list(parts)
        self._download = download  # (url, destination) -> DownloadResult
        self._staging = Path(staging_dir)
        self._group_budget = int(group_budget_bytes)
        self._layers = [zlib.decompressobj(16 + zlib.MAX_WBITS) for _ in range(decompress_layers)]
        self._buffer = bytearray()
        self._part_index = -1
        self._handle = None
        self._eof = False
        self.bytes_downloaded = 0
        self.stopped_early = False
        self.parts_used = 0

    # -- file-like interface -------------------------------------------

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            chunks: list[bytes] = []
            while True:
                chunk = self.read(65536)
                if not chunk:
                    break
                chunks.append(chunk)
            return b"".join(chunks)
        while len(self._buffer) < size and not self._eof:
            self._advance()
        take = min(size, len(self._buffer))
        result = bytes(self._buffer[:take])
        del self._buffer[:take]
        return result

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    # -- internals ------------------------------------------------------

    def _feed(self, block: bytes) -> None:
        for layer in self._layers:
            if not block:
                return
            block = layer.decompress(block)
        self._buffer.extend(block)

    def _flush_layers(self) -> None:
        try:
            block = self._layers[0].flush()
            for layer in self._layers[1:]:
                block = layer.decompress(block) + layer.flush() if block else layer.flush()
            self._buffer.extend(block)
        except zlib.error:
            pass

    def _advance(self) -> None:
        if self._handle is not None:
            block = self._handle.read(1024 * 256)
            if block:
                self._feed(block)
                return
            self._handle.close()
            self._handle = None
            return

        if self._eof:
            return

        next_index = self._part_index + 1
        if next_index >= len(self._parts):
            self._flush_layers()
            self._eof = True
            return

        name, url, size = self._parts[next_index]
        if self.bytes_downloaded + size > self._group_budget:
            self.stopped_early = True
            self._flush_layers()
            self._eof = True
            return
        destination = self._staging / name
        try:
            result = self._download(url, destination)
        except (HttpError, BudgetExceeded):
            self.stopped_early = True
            self._flush_layers()
            self._eof = True
            return
        self.bytes_downloaded += max(0, result.transferred_bytes)
        self._part_index = next_index
        self.parts_used += 1
        self._handle = destination.open("rb")


__all__ = ["SegmentedGzipReader"]
