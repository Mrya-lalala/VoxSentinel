"""Inventory coverage and reserve prior exposures; does not select or download v2."""
from collections import Counter
import json
from pathlib import Path

from scripts.gru_identity_audit import audit_identity, reference
from src.dataset_prep.materialize import sha256_file


def main():
    root = Path('artifacts/datasets')
    out = Path('artifacts/datasets-v2/planning')
    out.mkdir(parents=True, exist_ok=True)
    paths = [root/'manifests'/f'windows.core_{s}.jsonl' for s in ('train', 'val')]
    rows = [[json.loads(line) for line in p.read_text().splitlines() if line] for p in paths]
    audit = audit_identity(*rows)
    if not audit['ready']:
        raise ValueError('Existing identity evidence is not valid')
    speakers, refs, hashes = set(), set(), set()
    for row in rows[0] + rows[1]:
        lang = row['spoken_language'].lower()
        values = ([row['source_file']] if row['dataset_id'] == 'kathbath' else
                  [row['parent_refs'].get(f'{role}_reference') for role in ('source', 'target')])
        for value in values:
            parsed = reference(value)
            if parsed:
                speakers.add(('kathbath', lang, parsed['speaker']))
                refs.add(('kathbath', lang, parsed['canonical']))
        for kind in ('original_audio', 'prepared_audio'):
            if row.get(kind, {}).get('sha256'):
                hashes.add(row[kind]['sha256'])
    exclusions = {'scope': 'All v1 core train and development; extend for any other fitted or inspected data before test selection',
                  'speaker_keys': sorted(speakers), 'reference_keys': sorted(refs),
                  'audio_sha256': sorted(hashes),
                  'source_manifest_sha256': {str(p): sha256_file(p) for p in paths},
                  'limits': 'Language-scoped identity; cross-language identity and near duplicates still require review.'}
    (out/'prior_exposure_exclusions.json').write_text(json.dumps(exclusions, indent=2)+'\n')
    inventories = {}
    for path in sorted((root/'staging/joint').glob('kathbath_inventory.*.jsonl')):
        gender = Counter()
        for line in path.read_text().splitlines():
            gender[str(json.loads(line).get('gender', 'unknown'))] += 1
        inventories[path.name.split('.')[1]] = dict(gender)
    report = {'status': 'not_ready_to_train_or_release', 'existing_identity_audit_passed': audit['ready'],
              'existing_gender_coverage': audit['gender_coverage'], 'cached_genuine_inventory_gender': inventories,
              'prior_exposure_speakers': len(speakers), 'prior_exposure_references': len(refs),
              'budget': json.loads((root/'manifests/downloads.json').read_text()),
              'blockers': ['IndieFake access pending; no archive, manifest, terms or actual split audited',
                           'Cached genuine inventories have no documented male coverage; new balanced acquisition needed',
                           'No v2 connected-component selection or untouched test has been materialized',
                           'Cross-language identity and near-duplicate audit outstanding'],
              'mutations': 'Planning outputs only; frozen data, caches and weights unchanged'}
    (out/'readiness.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('existing_gender_coverage','cached_genuine_inventory_gender')}, indent=2))

if __name__ == '__main__':
    main()
