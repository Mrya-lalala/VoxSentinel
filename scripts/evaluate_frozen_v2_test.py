"""One fixed-checkpoint test evaluation; never trains or tunes thresholds."""
import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from scripts.validate_dataset_v2 import row_identities
from src.dataset_prep.materialize import sha256_file
from src.dataset_prep.splitting import UnionFind
from src.scoring.metrics import binary_metrics
from src.scoring.predict import FilePredictor


def summarize(rows):
    labels = np.array([r['label'] for r in rows])
    scores = np.array([r['synthetic_score'] for r in rows])
    margins = np.array([r['logits'][1]-r['logits'][0] for r in rows])
    m = asdict(binary_metrics(torch.tensor(scores), torch.tensor(labels), threshold=0.5))
    genuine, spoof = int((labels == 0).sum()), int((labels == 1).sum())
    m['genuine_false_alarm_rate'] = m['false_positive']/genuine if genuine else None
    m['spoof_miss_rate'] = m['false_negative']/spoof if spoof else None
    m['accuracy'] = (m['true_negative']+m['true_positive'])/len(rows)
    m['windows'], m['genuine'], m['spoof'] = len(rows), genuine, spoof
    if genuine and spoof:
        differences = margins[labels == 1, None] - margins[labels == 0][None, :]
        m['auc_margin'] = float(((differences > 0) + .5*(differences == 0)).mean())
        m['eer_margin'] = binary_metrics(torch.tensor(margins), torch.tensor(labels)).eer
        m['balanced_accuracy'] = 1-(m['genuine_false_alarm_rate']+m['spoof_miss_rate'])/2
    else:
        m['auc_margin'] = m['eer_margin'] = m['balanced_accuracy'] = None
    return m


def components(rows):
    uf = UnionFind(); first = {}
    for row in rows:
        ids = row_identities(row)
        keys = [('s', *x) for x in ids['speakers'].values()]+[('r', *x) for x in ids['references'].values()]
        if not keys:
            raise ValueError('Missing component identity')
        for key in keys:
            uf.union(keys[0], key)
        first[row['window_id']] = keys[0]
    return {wid: str(uf.find(key)) for wid, key in first.items()}


def bootstrap(rows, draws=2000, seed=20260919):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row['component']].append(i)
    blocks = list(groups.values()); rng = np.random.default_rng(seed)
    values = defaultdict(list)
    labels = np.array([r['label'] for r in rows]); pred = np.array([r['synthetic_score'] >= .5 for r in rows])
    for _ in range(draws):
        ix = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        y, p = labels[ix], pred[ix]
        values['accuracy'].append(float((y == p).mean()))
        if (y == 0).any(): values['genuine_false_alarm_rate'].append(float(p[y == 0].mean()))
        if (y == 1).any(): values['spoof_miss_rate'].append(float((~p[y == 1]).mean()))
    return {'method': '95% percentile bootstrap of admitted relationship components; exploratory small-sample intervals',
            'components': len(blocks), 'draws': draws, 'seed': seed,
            'intervals': {key: np.quantile(v, [.025, .975]).tolist() for key, v in values.items()}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=Path('artifacts/evaluations/gru-v2-test-epoch5'))
    args=p.parse_args(); out=args.out
    if out.exists(): raise FileExistsError('Evaluation destination already exists; do not repeat/select on test results')
    root=Path('artifacts/datasets-v2'); manifest=root/'manifests/windows.test.jsonl'
    version=json.loads((root/'manifests/dataset-version.v2.json').read_text())
    if sha256_file(manifest)!=version['manifests']['test']['sha256']: raise ValueError('Test manifest changed')
    audit_path=root/'reports/audit.json'; audit=json.loads(audit_path.read_text())
    if not audit['ready'] or audit.get('protocol')!='speaker_recording_disjoint_v2': raise ValueError('Invalid data audit/protocol')
    rows=[json.loads(line) for line in manifest.read_text().splitlines() if line]
    if len(rows)!=96 or len({r['window_id'] for r in rows})!=96 or any(r['split']!='test' for r in rows): raise ValueError('Unexpected test content')
    checkpoint=Path('artifacts/runs/gru-v2-split-reproduction/gru_best.pt')
    from src.detectors.checkpoints import load_checkpoint
    payload=load_checkpoint(checkpoint,model_name='gru')
    if payload['epoch']!=5: raise ValueError('Expected predetermined epoch 5')
    group_ids=components(rows)
    # Validate every input before any prediction; never skip failed examples.
    for row in rows:
        if sha256_file(root/row['prepared_audio']['path'])!=row['prepared_audio']['sha256']: raise ValueError('Audio hash mismatch')
    out.mkdir(parents=True)
    spec=json.loads(Path('artifacts/releases/indic-gru-pilot-v0.1/release.json').read_text())
    spec['head']={'file':str(checkpoint.resolve()),'sha256':sha256_file(checkpoint),'epoch':5}
    spec['release_id']='gru-v2-split-reproduction-epoch5-fixed-test'
    release=out/'release.json';release.write_text(json.dumps(spec,indent=2)+'\n')
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'checkpoint':str(checkpoint),'checkpoint_sha256':sha256_file(checkpoint),
          'manifest_sha256':sha256_file(manifest),'audit_sha256':sha256_file(audit_path),'epoch':5,'threshold':.5,
          'protocol':'speaker_recording_disjoint_v2','strict_conversion_family_compliant':False,
          'inputs':'All 96 manifest prepared windows; no failed example may be silently dropped',
          'selection':'Checkpoint and threshold fixed from development before test inference; no test-driven retuning',
          'metrics':['confusion','accuracy','false_alarm_rate','miss_rate','EER probability and logit-margin','AUC logit-margin'],
          'subgroups':['language','generator','genuine documented gender','synthetic target documented gender'],
          'uncertainty':{'unit':'admitted relationship component','draws':2000,'seed':20260919,'interval':'95% percentile for accuracy/FAR/miss'},
          'test_status':'This test becomes evaluated once predictions begin; do not reuse as untouched for adaptive tuning',
          'source_script_sha256':sha256_file(__file__)}
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    from scripts.train_gru_from_cache import _code_state
    (out/'code_state.json').write_text(json.dumps(_code_state(out),indent=2)+'\n')
    torch.set_num_threads(4)
    predictor=FilePredictor(release); results=[]
    try:
        with (out/'predictions.jsonl').open('x') as handle:
            for row in rows:
                result=predictor.predict(root/row['prepared_audio']['path'],language=row['spoken_language'])
                if result['status']!='ok': raise ValueError(f"Prepared test audio abstained: {row['window_id']}")
                if result['window']['start_sample']!=0 or result['window']['end_sample']!=row['window']['num_samples']:
                    raise ValueError('Prepared window was unexpectedly cropped')
                from scripts.gru_identity_audit import reference
                ref=reference(row['source_file'] if row['label']==0 else row['parent_refs']['target_reference'])
                item={**result,'window_id':row['window_id'],'label':row['label'],'language':row['spoken_language'],
                      'generator':row.get('generator') or 'genuine','gender':ref['gender'] if ref else 'unknown','component':group_ids[row['window_id']]}
                results.append(item);handle.write(json.dumps(item)+'\n');handle.flush()
                if len(results)%12==0: print(f'Evaluated {len(results)}/96',flush=True)
        subgroups=defaultdict(list)
        for r in results:
            subgroups['language/'+r['language']].append(r)
            subgroups['generator/'+r['generator']].append(r)
            subgroups[('genuine_gender/' if r['label']==0 else 'synthetic_target_gender/')+r['gender']].append(r)
        report={'status':'completed','plan_sha256':sha256_file(out/'plan.json'),'overall':summarize(results),
                'uncertainty':bootstrap(results),'subgroups':{k:summarize(v) for k,v in sorted(subgroups.items())},
                'checkpoint_unchanged':sha256_file(checkpoint)==plan['checkpoint_sha256'],
                'test_manifest_unchanged':sha256_file(manifest)==plan['manifest_sha256'],
                'limitations':['96 windows; eight per language; small subgroup counts',
                              'No cross-language person independence or strict conversion-family isolation established',
                              'Training genuine speech remains female only; English unvalidated',
                              'Selected-window classification, not whole-recording or streaming guarantee',
                              'Bootstrap components follow the admitted protocol, not full unused-conversion chains']}
        if not report['checkpoint_unchanged'] or not report['test_manifest_unchanged']:raise ValueError('Pinned inputs changed')
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps(report['overall'],indent=2))
    except BaseException as error:
        (out/'failure.json').write_text(json.dumps({'error':repr(error),'predictions_written':len(results)})+'\n')
        raise


if __name__=='__main__':main()
