import json
from pathlib import Path
import pytest
from scripts.training_split_contract import digest, verify_split_contract


def fixture(tmp_path):
    manifests = tmp_path/'manifests'
    manifests.mkdir()
    spec = {'schema': 'voxsentinel.dataset_version.v2', 'split_policy': 'fixture', 'manifests': {}}
    for split, old in [('train', 'train'), ('dev', 'val')]:
        p = manifests/f'windows.core_{old}.jsonl'
        p.write_text(json.dumps({'window_id': split, 'split': old})+'\n')
        q = tmp_path/f'{split}.jsonl'
        q.write_bytes(p.read_bytes())
        spec['manifests'][split] = {'path': str(q), 'sha256': digest(q)}
    spec['manifests']['test'] = {'path': str(tmp_path/'DOES_NOT_EXIST'), 'sha256': 'not_read'}
    version = tmp_path/'version.json'
    version.write_text(json.dumps(spec))
    return version, spec


def test_test_files_not_needed_or_read(tmp_path):
    version, _ = fixture(tmp_path)
    evidence = verify_split_contract(version, tmp_path)
    assert evidence['test_used'] is False
    assert evidence['splits']['dev']['cache_split'] == 'val'


def test_changed_manifest_rejected_even_with_updated_version_hash(tmp_path):
    version, spec = fixture(tmp_path)
    p = Path(spec['manifests']['train']['path'])
    p.write_text(json.dumps({'split': 'train', 'window_id': 'new'})+'\n')
    spec['manifests']['train']['sha256'] = digest(p)
    version.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='cache manifest'):
        verify_split_contract(version, tmp_path)


def test_test_rows_cannot_be_presented_as_training(tmp_path):
    version, spec = fixture(tmp_path)
    payload = json.dumps({'split': 'test'})+'\n'
    p = Path(spec['manifests']['train']['path']); p.write_text(payload)
    (tmp_path/'manifests/windows.core_train.jsonl').write_text(payload)
    spec['manifests']['train']['sha256'] = digest(p)
    version.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='split assignment'):
        verify_split_contract(version, tmp_path)
