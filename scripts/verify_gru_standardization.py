"""Bounded transform/reconstruction checks; no optimizer updates or broad suite."""
import copy
import hashlib
import json
import subprocess
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.gru_recovery import CORRECTED, ROOT, dump, examples_and_items, sha
from src.config import load_config
from src.data.batch import EmbeddingExample, collate_examples
from src.detectors.gru import GruSpoofDetector
from src.detectors.registry import create_detector
from src.detectors.standardization import FrameStandardizer, fit_detector_standardizer
from src.detectors.checkpoints import save_checkpoint, restore_checkpoint


def main():
    torch.set_num_threads(4)
    report={}
    # Independent tiny arithmetic oracle: frame weighting must differ from window weighting.
    toy=[EmbeddingExample(torch.tensor([[1.,2.],[3.,2.]]),0),EmbeddingExample(torch.tensor([[8.,2.]]),1)]
    scaler=FrameStandardizer(2);scaler.fit(toy)
    assert torch.allclose(scaler.mean,torch.tensor([4.,2.]))
    assert torch.allclose(scaler.population_variance,torch.tensor([26/3,0.],dtype=torch.float64))
    assert scaler.metadata['valid_frames']==3 and scaler.metadata['clamped_dimensions']==1
    report['frame_population_arithmetic']='passed (unequal window lengths, constant dimension)'
    original_source=subprocess.run(['git','show','884fcd7381b8409307c3b239fe9db5a22bc4798d:src/detectors/gru.py'],capture_output=True,text=True,check=True).stdout
    original=types.ModuleType('src.detectors._historical_gru');original.__package__='src.detectors'
    # Omit registration: examining historical code must not replace the active factory.
    exec(original_source.replace('register_detector("gru", GruDetector.from_config)',''),original.__dict__)
    torch.manual_seed(0);historical=original.GruSpoofDetector();old_rng=torch.get_rng_state()
    torch.manual_seed(0);enabled=GruSpoofDetector(feature_standardization=True);new_rng=torch.get_rng_state()
    assert torch.equal(old_rng,new_rng)
    assert all(torch.equal(x,dict(enabled.named_parameters())[n]) for n,x in historical.named_parameters())
    report['initialization_matches_reviewed_historical_class']=True
    report['construction_rng_parity']=True
    examples,items=examples_and_items('train')
    rng=torch.get_rng_state().clone()
    enabled.standardizer.fit(examples,cache_signature=sha(ROOT/'features/train.pt'))
    assert torch.equal(rng,torch.get_rng_state())
    # Two-pass independent float64 oracle on representative dimensions.
    dims=[0,1,17,127,511,1023]
    x=torch.cat([e.features[:,dims].double() for e in examples])
    assert torch.allclose(enabled.standardizer.mean[dims].double(),x.mean(0),atol=2e-8,rtol=1e-7)
    assert torch.allclose(enabled.standardizer.population_variance[dims],x.var(0,unbiased=False),atol=1e-16,rtol=1e-10)
    report['training_statistics_vs_independent_two_pass']='passed on six dimensions'
    report['fitting_rng_parity']=True
    report['transform']=enabled.standardizer.metadata
    assert not list(enabled.standardizer.parameters())
    report['standardizer_has_no_optimizer_parameters']=True
    batch=collate_examples(examples[:2]);output=enabled(batch['features'],batch['valid_lengths'],batch['padding_mask'])
    output.sum().backward()
    assert all(buf.grad is None for buf in enabled.standardizer.buffers())
    report['transform_has_no_gradients']=True
    torch.manual_seed(0);disabled=GruSpoofDetector()
    with torch.no_grad():
        assert torch.equal(historical(batch['features'],batch['valid_lengths'],batch['padding_mask']),disabled(batch['features'],batch['valid_lengths'],batch['padding_mask']))
    report['default_path_exact_historical_logits']=True
    # Current iterator uses the same per-epoch local generator as historical code.
    from src.data.batch import batch_iterator
    old_batch_source=subprocess.run(['git','show','884fcd7381b8409307c3b239fe9db5a22bc4798d:src/data/batch.py'],capture_output=True,check=True).stdout
    assert hashlib.sha256(old_batch_source).hexdigest()==sha(Path('src/data/batch.py'))
    orders={}
    for epoch in range(1,11):
        a=torch.randperm(len(examples),generator=torch.Generator().manual_seed(epoch))
        # Verify actual emitted examples against the known permutation, not just seeds.
        offset=0
        for batch in batch_iterator(examples,8,shuffle=True,seed=epoch):
            for j,length in enumerate(batch['valid_lengths']):
                e=examples[int(a[offset+j])]
                assert torch.equal(batch['features'][j,:length],e.features) and int(batch['labels'][j])==e.label
            offset+=len(batch['labels'])
        orders[str(epoch)]=hashlib.sha256(a.numpy().tobytes()).hexdigest()
    report['epoch_batch_order_sha256']=orders
    config=load_config(['configs/base.yaml','configs/gru.yaml','configs/gru_standardized.yaml'])
    detector=create_detector(config.model);fit_detector_standardizer(detector,examples,cache_signature=sha(ROOT/'features/train.pt'))
    with tempfile.TemporaryDirectory(prefix='vox-transform-') as tmp:
        path=Path(tmp)/'model.pt';save_checkpoint(path,detector,'gru',config.model.parameters)
        restored=create_detector(config.model);restore_checkpoint(path,restored,model_name='gru')
        batch=collate_examples(examples[:2])
        with torch.no_grad(): assert torch.equal(detector(batch).logits,restored(batch).logits)
        payload=torch.load(path,weights_only=False);del payload['model_state_dict']['model.standardizer.mean'];torch.save(payload,path)
        try:
            restore_checkpoint(path,create_detector(config.model),strict=False)
        except RuntimeError: pass
        else: raise AssertionError('Missing required transform state was accepted')
    report['checkpoint_roundtrip_exact']=True;report['missing_transform_state_rejected_even_non_strict']=True
    # Exercise the generic runner's preparation without invoking fit or optimizer steps.
    from src.detectors.runner import run_training
    def inspect_runner(model,*args,**kwargs):
        assert bool(model.model.standardizer.fitted)
        assert torch.equal(model.model.standardizer.mean,enabled.standardizer.mean)
        return 'prepared'
    with patch('src.detectors.runner.fit',inspect_runner):
        assert run_training(config,examples,examples[:2])=='prepared'
    report['generic_runner_transform_preparation']='passed; fit intercepted, no training'
    # In-memory identity failure cases: never write the frozen manifests.
    from scripts.gru_identity_audit import audit_identity, numeric_id, reference
    from scripts.gru_recovery import manifests, rebuild
    rows=manifests()
    genuine=next(x for x in rows['train'] if x['dataset_id']=='kathbath')
    fake=copy.deepcopy(next(x for x in rows['val'] if x['dataset_id']=='indicsynth' and x['spoken_language']==genuine['spoken_language']))
    fake['speaker_ids']['target']=genuine['speaker_ids']['speaker']+'.0'
    fake['parent_refs']['target_reference']=genuine['source_file'].split('/')[-1].replace('.m4a','.wav')
    fake['parent_refs']['reference_evidence']['target']={'fname':genuine['source_file'].split('/')[-1], 'shard':genuine['parent_refs']['shard'],'method':'scanned_inventory'}
    audit=audit_identity(rows['train'],[fake])
    assert not audit['ready']
    assert audit['speaker_key_intersections']['train_speaker_vs_val_target']['count']>0
    assert audit['reference_recording_intersections']['train_speaker_vs_val_target']['count']>0
    fake['speaker_ids']['target']=None
    assert audit_identity(rows['train'],[fake])['unresolved_required_count']>0
    duplicate=copy.deepcopy(genuine);duplicate['window_id']='duplicate-probe'
    audit=audit_identity([genuine,duplicate],[])
    assert audit['prepared_duplicates_within_split']['train'] and not audit['prepared_duplicates_cross_split']
    assert audit_identity([genuine],[duplicate])['prepared_duplicates_cross_split']
    assert numeric_id('00089.0')=='89' and reference('00012-0089-F.WAV')['canonical']=='12-89-f'
    dump(CORRECTED/'identity_checks.json',{'cross_class_cross_role_injected_overlap_detected':True,'required_missing_identity_reported':True,'within_and_cross_split_hash_duplicates_distinguished':True,'numeric_and_extension_normalization':True,'no_manifest_changes':True})
    run=Path('artifacts/runs/gru-core-v2-standardized')
    if (run/'run_status.json').exists():
        saved=torch.load(run/'standardizer.pt',weights_only=False)
        final_checks={}
        for name in ('best','final'):
            model,payload=rebuild(run/f'gru_{name}.pt')
            state=model.model.standardizer.state_dict()
            assert state.keys()==saved.keys()
            assert all(torch.equal(v,saved[k]) if torch.is_tensor(v) else v==saved[k] for k,v in state.items())
            final_checks[name+'_transform_unchanged_during_training']=True
            from src.config import ModelConfig
            plain=create_detector(ModelConfig('gru',{k:v for k,v in payload['model_config'].items() if k!='feature_standardization'}))
            try: restore_checkpoint(run/f'gru_{name}.pt',plain,strict=False)
            except ValueError: pass
            else: raise AssertionError('Mismatched transform config accepted')
        final_checks['mismatched_transform_config_rejected']=True
        evaluation=json.loads((run/'evaluation/evaluation.json').read_text())
        final_checks['live_best_reload_exact']=evaluation['checks']['live_best_reload_exact']
        final_checks['live_final_reload_max_logit_error']=evaluation['checks']['live_final_reload_max_logit_error']
        final_checks['single_completed_standardized_run']=len(list(Path('artifacts/runs').glob('gru-core-v2-standardized*')))==1
        dump(CORRECTED/'final_checks.json',final_checks)
    dump(CORRECTED/'standardization_checks.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
