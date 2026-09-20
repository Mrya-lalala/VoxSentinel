"""Bind unchanged v2 train/dev manifests to existing audited training caches."""
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_split_contract(version_path, dataset_root):
    version_path, dataset_root = Path(version_path), Path(dataset_root)
    spec = json.loads(version_path.read_text())
    if spec.get('schema') != 'voxsentinel.dataset_version.v2':
        raise ValueError('Unsupported split-version schema')
    evidence = {'version_file': str(version_path), 'version_sha256': digest(version_path),
                'split_policy': spec['split_policy'],
                'strict_conversion_family_compliant': spec.get('strict_conversion_family_compliant'),
                'test_used': False, 'splits': {}}
    for split, cached_split in (('train', 'train'), ('dev', 'val')):
        entry = spec['manifests'][split]
        manifest = Path(entry['path'])
        cached_manifest = dataset_root/'manifests'/f'windows.core_{cached_split}.jsonl'
        actual = digest(manifest)
        if actual != entry['sha256'] or actual != digest(cached_manifest):
            raise ValueError(f'{split}: v2 manifest does not match pinned cache manifest')
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]
        if not rows or any(r.get('split') != cached_split for r in rows):
            raise ValueError(f'{split}: wrong or empty split assignment')
        evidence['splits'][split] = {'manifest': str(manifest), 'sha256': actual,
                                     'windows': len(rows), 'cache_split': cached_split}
    # Test identity is recorded only as declared metadata. Never open its rows,
    # audio, or features from the training entry point.
    evidence['reserved_test'] = dict(spec['manifests']['test'])
    return evidence
