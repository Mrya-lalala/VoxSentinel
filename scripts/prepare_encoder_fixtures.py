"""Unpack two public speech fixtures from a local, hash-checked parquet archive.

No dataset splitting, training, or network access. Requires requirements-fixtures.txt.
"""
import argparse
import hashlib
from pathlib import Path
import pyarrow.parquet as pq

ARCHIVE_SHA256 = '4e69a06fa5edc90921e5e7e39a7084881f8b3ed9c805c574f4f39c6fde27c603'
FIXTURES = [
    ('1272-128104-0000', '4e25e22555cd16e90edb0a3b49fdcf1fe652b2a1250ab643634db33895c75b41'),
    ('1272-128104-0001', '46a9d58622b4675b29564da2d9ba73e702241c5fa969f12c387cad4aa984276a'),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parquet', required=True)
    parser.add_argument('--output-dir', default='artifacts/speech')
    args = parser.parse_args()
    path = Path(args.parquet)
    if hashlib.sha256(path.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        parser.error('speech archive SHA-256 mismatch')
    rows = {r['id']: r for r in pq.read_table(path).to_pylist()}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, digest in FIXTURES:
        data = rows[name]['audio']['bytes']
        if hashlib.sha256(data).hexdigest() != digest:
            parser.error(f'{name}: audio SHA-256 mismatch')
        target = output / f'{name}.flac'
        target.write_bytes(data)
        print(target)


if __name__ == '__main__':
    main()
