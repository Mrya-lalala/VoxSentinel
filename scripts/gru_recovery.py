"""Bounded recovery audits/evaluation. Training remains in train_gru_from_cache/fit."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from scripts.gru_identity_audit import audit_identity
from scripts.diagnose_gru import auc_rank, eer_independent, confusion_report, distribution_stats, _forward_logits
from src.config import ModelConfig
from src.data.batch import EmbeddingExample, collate_examples
from src.dataset_prep.features import load_feature_bundle, validate_core_training_cache
from src.detectors.checkpoints import load_checkpoint, restore_checkpoint
from src.detectors.registry import create_detector

ROOT = Path('artifacts/datasets')
REFERENCE = Path('artifacts/runs/gru-core-v1')
CORRECTED = Path('artifacts/runs/gru-core-v1-diagnosis-corrected')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def manifests(root=ROOT):
    return {s:[json.loads(line) for line in (root/'manifests'/f'windows.core_{s}.jsonl').read_text().splitlines() if line.strip()] for s in ('train','val')}


def preservation_inventory():
    paths=[]
    for directory in [REFERENCE, Path('artifacts/runs/gru-core-v1-diagnosis'), ROOT/'features', ROOT/'manifests']:
        paths.extend(p for p in directory.rglob('*') if p.is_file())
    return {str(p):sha(p) for p in sorted(paths)}


def readiness(root=ROOT, reference_run=REFERENCE):
    rows=manifests(root)
    identity=audit_identity(rows['train'],rows['val'])
    from scripts.train_gru_from_cache import CORE_LANGUAGES
    cache=validate_core_training_cache(root/'features',train_manifest=root/'manifests/windows.core_train.jsonl',val_manifest=root/'manifests/windows.core_val.jsonl',expected_languages=CORE_LANGUAGES)
    original=json.loads((reference_run/'dataset_identity.json').read_text())
    failures=[]
    signatures={}
    for split in ('train','val'):
        manifest=root/'manifests'/f'windows.core_{split}.jsonl'; bundle_path=root/'features'/f'{split}.pt'
        b=load_feature_bundle(bundle_path); lookup={r['window_id']:r for r in rows[split]}
        signatures[split]={'manifest_sha256':sha(manifest),'bundle_sha256':sha(bundle_path),'cache_identity':b['identity'],'window_ids':[i['window_id'] for i in b['items']]}
        for key in ('manifest_sha256','bundle_sha256','cache_identity'):
            if signatures[split][key]!=original[split][key]: failures.append(f'{split}: original {key} mismatch')
        if len(lookup)!=len(rows[split]) or len({i['window_id'] for i in b['items']})!=len(b['items']) or set(lookup)!={i['window_id'] for i in b['items']}: failures.append(f'{split}: window IDs mismatch')
        for item in b['items']:
            row=lookup.get(item['window_id'])
            if row is None: continue
            x=item['features']
            if int(item['label'])!=row['label'] or item['prepared_sha256']!=row['prepared_audio']['sha256'] or item.get('spoken_language')!=row['spoken_language']: failures.append(f'{item["window_id"]}: item metadata mismatch')
            if x.dtype!=torch.float32 or x.shape!=(item['num_frames'],1024) or not torch.isfinite(x).all() or x.requires_grad: failures.append(f'{item["window_id"]}: feature contract')
            if sha(root/row['prepared_audio']['path'])!=item['prepared_sha256']: failures.append(f'{item["window_id"]}: audio hash mismatch')
        enc={k:b['identity'].get(k) for k in ('encoder_checkpoint_sha256','output_layer','embedding_dim','preprocessing_version')}
        import yaml
        cfg=yaml.safe_load(Path('configs/backbones.yaml').read_text())['indic_wav2vec']
        if enc['encoder_checkpoint_sha256']!=cfg['checkpoint_sha256'] or enc['output_layer'] is not None or enc['embedding_dim']!=1024 or enc['preprocessing_version']!='voxsentinel-prep-2': failures.append(f'{split}: encoder identity mismatch')
    old_code=json.loads((reference_run/'code_state.json').read_text())['changed_or_untracked_source_hashes']
    divergence={}
    # Compare original-run hashes to the reviewed commit, before our intentional changes.
    for path,digest in old_code.items():
        content=subprocess.run(['git','show',f'884fcd7381b8409307c3b239fe9db5a22bc4798d:{path}'],capture_output=True,check=True).stdout
        if hashlib.sha256(content).hexdigest()!=digest: divergence[path]='differs from original run at reviewed commit'
    unknown=[p for p in divergence if not (p.startswith('docs/') or p in ('README.md','.gitignore'))]
    failures.extend('unexpected historical code divergence: '+p for p in unknown)
    return {'ready':identity['ready'] and cache['ready'] and not cache['skipped'] and not failures,
            'failures':failures,'canonical_identity':identity,'cache_audit':cache,'signatures':signatures,
            'original_code_divergence_at_reviewed_commit':divergence,
            'audio_files_hash_checked':len(rows['train'])+len(rows['val'])}


def examples_and_items(split):
    b=load_feature_bundle(ROOT/'features'/f'{split}.pt')
    return [EmbeddingExample(features=i['features'],label=int(i['label'])) for i in b['items']],b['items']


def evaluate_model(model, examples):
    logits=_forward_logits(model,examples,8)
    labels=np.array([e.label for e in examples],dtype=np.int64)
    margins=logits[:,1]-logits[:,0]
    scores=torch.softmax(torch.from_numpy(logits),1)[:,1].numpy()
    result={'cross_entropy':float(torch.nn.functional.cross_entropy(torch.from_numpy(logits),torch.from_numpy(labels))),
            **confusion_report(scores,labels), 'eer':eer_independent(margins,labels)['eer'],'auc':auc_rank(margins,labels),
            'probability_eer':eer_independent(scores,labels)['eer'],'probability_auc':auc_rank(scores,labels),
            'ranking_basis':'actual logit margin; checkpoint selection still uses original probability EER',
            'score_stats':distribution_stats(scores),'margin_stats':distribution_stats(margins),
            'unique_probabilities':len(np.unique(scores)),'unique_margins':len(np.unique(margins)),
            'by_class':{str(c):{'scores':distribution_stats(scores[labels==c]),'margins':distribution_stats(margins[labels==c])} for c in (0,1)}}
    return result,logits,scores


def rebuild(path):
    p=load_checkpoint(path,model_name='gru')
    model=create_detector(ModelConfig('gru',dict(p['model_config'])))
    restore_checkpoint(path,model,model_name='gru')
    return model.eval(),p


def evaluate_run(run_dir, out):
    out.mkdir(parents=True,exist_ok=True)
    report={'run_dir':str(run_dir),'models':{},'checks':{}}
    for name in ('best','final'):
        model,payload=rebuild(run_dir/f'gru_{name}.pt')
        for split in ('train','val'):
            examples,items=examples_and_items(split)
            metrics,logits,scores=evaluate_model(model,examples)
            report['models'][name+'/'+split]=metrics
            with (out/f'{name}_{split}_predictions.csv').open('w',newline='') as f:
                writer=csv.writer(f);writer.writerow(['window_id','label','language','generator','logit_genuine','logit_synthetic','logit_margin','score'])
                writer.writerows((i['window_id'],i['label'],i.get('spoken_language'),i.get('generator'),float(l[0]),float(l[1]),float(l[1]-l[0]),float(s)) for i,l,s in zip(items,logits,scores))
            if split=='val':
                singles=_forward_logits(model,examples,1)
                error=float(np.abs(singles-logits).max())
                if error>1e-5: raise AssertionError(f'Batch parity failed: {error}')
                subset=[examples[int(np.argmin([e.num_frames for e in examples]))],examples[0]]
                batch=collate_examples(subset)
                extended={**batch,'features':torch.cat([batch['features'],torch.full((2,40,1024),77.)],1),
                          'padding_mask':torch.cat([batch['padding_mask'],torch.ones(2,40,dtype=torch.bool)],1)}
                with torch.no_grad():
                    pad_error=float((model(batch).logits-model(extended).logits).abs().max())
                    if pad_error>1e-6: raise AssertionError(f'Padding parity failed: {pad_error}')
                    second,_=rebuild(run_dir/f'gru_{name}.pt')
                    reload_error=float((model(batch).logits-second(batch).logits).abs().max())
                    if reload_error!=0: raise AssertionError('Reload parity failed')
                    transform=getattr(model.model,'standardizer',None)
                    if transform is not None:
                        valid=~extended['padding_mask'];z=transform(extended['features'],valid)
                        if torch.count_nonzero(z[~valid]): raise AssertionError('Transformed padding is nonzero')
                report['checks'][name]={'batch1_vs_batch8_max_logit_error':error,'masked_77_padding_max_logit_error':pad_error,'reload_max_logit_error':reload_error,
                                       'checkpoint_sha256':sha(run_dir/f'gru_{name}.pt'),'epoch':payload['epoch']}
        if getattr(model.model,'standardizer',None) is not None:
            report['transform']=model.model.standardizer.metadata
    dump(out/'evaluation.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['readiness','evaluate','preserve'])
    parser.add_argument('--run-dir',type=Path,default=REFERENCE)
    parser.add_argument('--out-dir',type=Path,default=CORRECTED)
    args=parser.parse_args();torch.set_num_threads(4)
    if args.command=='readiness':
        report=readiness();dump(args.out_dir/'readiness.json',report)
        print(json.dumps({'ready':report['ready'],'failures':report['failures'],'divergence':report['original_code_divergence_at_reviewed_commit']},indent=2))
        return 0 if report['ready'] else 1
    if args.command=='preserve':
        path=args.out_dir/'preserved_originals_sha256.json'
        if path.exists(): raise FileExistsError(path)
        dump(path,preservation_inventory());return 0
    report=evaluate_run(args.run_dir,args.out_dir)
    print(json.dumps({k:{x:v[x] for x in ('cross_entropy','accuracy','eer','auc','tn','fp','fn','tp')} for k,v in report['models'].items()},indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
