import io

import pytest

from src.dataset_prep.bounded_reader import BoundedReader
from src.dataset_prep.budget import BudgetExceeded, DownloadLedger


class Observed(io.BytesIO):
    reads = 0

    def read(self, size=-1):
        self.reads += 1
        return super().read(size)


def test_combined_reads_cannot_overspend(tmp_path):
    ledger = DownloadLedger(tmp_path/'ledger.json', 10)
    allowance = {'remaining': 30}
    first, second = Observed(b'123456'), Observed(b'abcdef')
    assert BoundedReader(first, ledger, 'test', allowance).read(6) == b'123456'
    with pytest.raises(BudgetExceeded):
        BoundedReader(second, ledger, 'test', allowance).read(6)
    assert second.reads == 0
    assert ledger.total_bytes == 6


def test_failed_read_retains_reservation(tmp_path):
    class Failed(Observed):
        def read(self, size=-1):
            raise OSError('partial transfer failure')
    ledger = DownloadLedger(tmp_path/'ledger.json', 10)
    allowance = {'remaining': 10}
    reader = BoundedReader(Failed(), ledger, 'test', allowance)
    with pytest.raises(OSError):
        reader.read(6)
    assert ledger.total_bytes == reader.bytes == 6
    assert allowance['remaining'] == 4


def test_language_limit_and_unbounded_reads_stop_before_io(tmp_path):
    raw = Observed(b'12345')
    reader = BoundedReader(raw, DownloadLedger(tmp_path/'ledger.json', 100), 'test', {'remaining': 3})
    with pytest.raises(BudgetExceeded):
        reader.read(4)
    with pytest.raises(ValueError):
        reader.read()
    assert raw.reads == 0


def test_seek_readinto_and_short_read_are_conservative(tmp_path):
    ledger = DownloadLedger(tmp_path/'ledger.json', 10)
    reader = BoundedReader(Observed(b'abcdef'), ledger, 'test', {'remaining': 10})
    reader.seek(4)
    buf = bytearray(4)
    assert reader.readinto(buf) == 2
    assert buf[:2] == b'ef'
    assert reader.tell() == 6
    assert ledger.total_bytes == 4
