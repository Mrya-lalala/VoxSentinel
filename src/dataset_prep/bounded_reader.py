"""Conservative pre-charging for uncached, bounded acquisition reads.

Charges requested payload bytes before I/O and retains charges on failure or
short reads. This is a conservative allowance, not an exact wire-byte counter.
The caller must disable upstream read-ahead and run one acquisition job at a time.
"""
import io

from .budget import BudgetExceeded


class BoundedReader(io.RawIOBase):
    def __init__(self, inner, ledger, source, allowance):
        super().__init__()
        self.inner = inner
        self.ledger = ledger
        self.source = source
        self.allowance = allowance  # shared mutable remaining allowance
        self.bytes = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.inner.tell()

    def seek(self, offset, whence=0):
        return self.inner.seek(offset, whence)

    def read(self, size=-1):
        if size is None or size < 0:
            raise ValueError('Acquisition reads must specify a bounded size')
        if size > self.allowance['remaining']:
            raise BudgetExceeded('Per-language acquisition allowance exhausted before read')
        if size == 0:
            return b''
        # Charge before any bytes can be transferred. Failed reads stay reserved.
        self.ledger.charge(self.source, size)
        self.allowance['remaining'] -= size
        self.bytes += size
        return self.inner.read(size)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def close(self):
        if not self.closed:
            self.inner.close()
        super().close()


class SeekableRangeReader(io.RawIOBase):
    """Seekable serial range reader over an ``HfFileSystem`` path.

    PyArrow needs a seekable file, while ``block_size=0`` HF streams only
    support forward no-op seeks.  This wrapper keeps a position pointer, and
    lazily (re)opens a streaming request at the exact requested offset
    (``Range: bytes=<pos>-``); each ``read`` charges the requested byte count
    to the shared ledger *before* any transfer and retains the charge on
    failure or short reads.  Reads are strictly bounded (no read-ahead); run
    one acquisition job at a time.
    """

    def __init__(self, fs, path, ledger, source, allowance):
        super().__init__()
        self._fs = fs
        self._path = path
        self._ledger = ledger
        self._source = source
        self._allowance = allowance  # shared mutable remaining allowance
        self._pos = 0
        self._size = None
        self._stream = None
        self.bytes = 0  # total charged (requested) bytes, retained on failure

    # -- positioning -------------------------------------------------------- #

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def size(self):
        if self._size is None:
            self._size = int(self._fs.info(self._path)["size"])
        return self._size

    def seek(self, loc, whence=0):
        if whence == 0:
            target = loc
        elif whence == 1:
            target = self._pos + loc
        elif whence == 2:
            target = self.size() + loc
        else:
            raise ValueError(f'unsupported whence {whence}')
        if target < 0:
            raise ValueError('negative seek position')
        self._pos = target
        return target

    # -- reading ------------------------------------------------------------ #

    def _ensure_stream(self):
        if self._stream is None or self._stream.loc != self._pos:
            if self._stream is not None:
                self._stream.close()
            stream = self._fs.open(self._path, 'rb', block_size=0, cache_type='none')
            stream.loc = self._pos  # _open_connection() uses loc for the Range header
            self._stream = stream

    def read(self, size=-1):
        if size is None or size < 0:
            raise ValueError('Acquisition reads must specify a bounded size')
        if size == 0:
            return b''
        if size > self._allowance['remaining']:
            raise BudgetExceeded('Per-language acquisition allowance exhausted before read')
        self._ensure_stream()
        # Charge before any bytes can be transferred. Failed reads stay reserved.
        self._ledger.charge(self._source, size)
        self._allowance['remaining'] -= size
        self.bytes += size
        data = self._stream.read(size)
        self._pos += len(data)
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def close(self):
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        super().close()
