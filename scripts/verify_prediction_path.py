"""Raw audio -> production predictor -> restored score parity; no fitting."""
import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import soundfile as sf
import torch

from src.scoring.predict import FilePredictor
from src.dataset_prep.features import load_feature_bundle
from src.dataset_prep.materialize import sha256_file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',default='artifacts/releases/indic-gru-pilot-v0.1/release.json')
    p.add_argument('--output',type=Path,default=Path('artifacts/prediction-path/report.json'))
    args=p.parse_args();torch.set_num_threads(4)
    root=Path('artifacts/datasets');rows=[json.loads(x) for x in (root/'manifests/windows.core_val.jsonl').read_text().splitlines()]
    cached={i['window_id']:i for i in load_feature_bundle(root/'features/val.pt')['items']}
    selected=[next(r for r in rows if r['spoken_language']==language and r['label']==label) for language in ('Hindi','Tamil') for label in (0,1)]
    predictor=FilePredictor(args.release)
    head_before=sha256_file(predictor.head_path);checks=[];start=time.perf_counter()
    for row in selected:
        if row['dataset_id']=='kathbath': raw=root/'raw/kathbath'/row['source_file']
        else: raw=root/'raw/indicsynth'/row['spoken_language'].lower()/(row['window_id'].split('-')[-1]+'.wav')
        assert sha256_file(raw)==row['original_audio']['sha256']
        result=predictor.predict(raw,language=row['spoken_language'])
        assert result['status']=='ok'
        expected=cached[row['window_id']]['features']
        batch={'features':expected[None],'valid_lengths':torch.tensor([len(expected)]),'padding_mask':torch.zeros(1,len(expected),dtype=torch.bool)}
        with torch.no_grad(): logits=predictor.detector(batch).logits[0]
        delta=float((torch.tensor(result['logits'])-logits).abs().max())
        assert delta<1e-5,(row['window_id'],delta)
        assert result['window']['start_sample']==row['window']['start_sample']
        assert result['window']['end_sample']==row['window']['end_sample']
        assert result['valid_frames']==cached[row['window_id']]['num_frames']
        checks.append({'window_id':row['window_id'],'label':row['label'],'cache_vs_raw_logit_max_error':delta,'result':result})
    with tempfile.TemporaryDirectory(prefix='vox-predict-') as td:
        td=Path(td);prepared=root/selected[0]['prepared_audio']['path']
        mono,sr=sf.read(prepared,dtype='float32');sf.write(td/'stereo.wav',np.column_stack([mono,mono]),sr,subtype='FLOAT')
        first=predictor.predict(prepared);stereo=predictor.predict(td/'stereo.wav')
        assert first['logits']==stereo['logits']
        sf.write(td/'silence.wav',np.zeros(32000,dtype=np.float32),16000,subtype='FLOAT')
        silence=predictor.predict(td/'silence.wav');assert silence['status']=='insufficient_audio' and silence['prediction'] is None
        sf.write(td/'short.wav',mono[:8000],16000,subtype='FLOAT')
        short=predictor.predict(td/'short.wav');assert short['status']=='insufficient_audio'
        (td/'corrupt.wav').write_text('not audio')
        try: predictor.predict(td/'corrupt.wav')
        except (ValueError,RuntimeError): pass
        else: raise AssertionError('Invalid audio accepted')
        release=json.loads(Path(args.release).read_text());release['head']['file']=str(predictor.head_path);release['head']['sha256']='0'*64
        (td/'bad-release.json').write_text(json.dumps(release))
        try: FilePredictor(td/'bad-release.json')
        except ValueError: pass
        else: raise AssertionError('Corrupt release accepted')
    assert head_before==sha256_file(predictor.head_path)
    assert not predictor.detector.training and all(p.grad is None for p in predictor.detector.parameters())
    assert not predictor.encoder._model.training and all(not p.requires_grad and p.grad is None for p in predictor.encoder._model.parameters())
    report={'status':'passed','scope':'local audio file through decode/mono/resample/window/real frozen encoder/restored standardizer/GRU/score/label',
            'checks':checks,'stereo_identical_channels_exact':True,'silence_and_short_abstain':True,'corrupt_audio_rejected':True,
            'wrong_checkpoint_hash_rejected':True,'head_checkpoint_unchanged':True,'encoder_and_head_frozen':True,
            'seconds':time.perf_counter()-start,'classification_quality_claim':False,
            'limitations':['Known development examples used for path parity, not independent performance','No microphone, streaming, service or UI layer exists in this path','Default selects one window of long files','English scope unvalidated']}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))

if __name__=='__main__': main()
