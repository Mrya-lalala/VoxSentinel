"""Calibrate frozen epoch 3, then score the reserved held-out extension once."""
import importlib.util
import json
import time
from dataclasses import asdict
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('regional_run',HERE/'run.py')
r=importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
np,torch=r.np,r.torch
DATA=r.ROOT.parents[1]/'datasets-regional-heldout-v1'
OUT=HERE/'final-evaluation'
OUT.mkdir(exist_ok=True)

def report(scores,rows,threshold):
    labels=torch.tensor([x['label'] for x in rows])
    result={'overall':asdict(r.binary_metrics(scores,labels,threshold)), 'languages':{}}
    for lang in sorted({x['spoken_language'] for x in rows}):
        ix=[i for i,x in enumerate(rows) if x['spoken_language']==lang]
        result['languages'][lang]=dict(windows=len(ix),**asdict(r.binary_metrics(scores[ix],labels[ix],threshold)))
    return result

def threshold(scores,labels):
    values=np.unique(scores.numpy()).astype(np.float64)
    middle=((values[:-1]+values[1:])/2).astype(np.float32)
    # Float32 comparisons must place every midpoint strictly above its lower score.
    middle=np.where(middle<=values[:-1],values[1:],middle).astype(np.float32)
    candidates=np.unique(np.r_[np.float32(0),np.float32(.5),middle,
                                  np.nextafter(np.float32(values[-1]),np.float32(np.inf))]).astype(np.float32)
    y=np.array(labels); s=scores.numpy(); genuine=int((y==0).sum()); spoof=int((y==1).sum())
    rankings=[]
    for t in candidates:
        pred=s>=t
        fp=int((pred&(y==0)).sum()); fn=int((~pred&(y==1)).sum())
        rankings.append((fp*spoof+fn*genuine,abs(float(t)-.5),float(t)))
    return min(rankings)[2]

def main():
    assert not (OUT/'test-exposure.json').exists(), 'Final test already exposed; inspect saved results'
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    version=json.loads((DATA/'dataset-version.json').read_text())
    validation=json.loads((DATA/'validation.json').read_text())
    assert r.sha(DATA/'dataset-version.json')==validation['version_sha256']
    modelpath=HERE/'best.pt'; original_sha=r.sha(modelpath)
    accepted=json.loads((HERE/'result.json').read_text())
    assert original_sha==accepted['checkpoint_sha256']
    rows={}
    for split in ('calibration','test'):
        entry=version['manifests'][split]; path=DATA/entry['path']
        assert r.sha(path)==entry['sha256']
        rows[split]=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
        assert len(rows[split])==entry['rows']
        assert all(x['split']==split and x['preprocessing_version']=='voxsentinel-prep-2' for x in rows[split])
    r.write(OUT/'identity.json',dict(checkpoint_sha256=original_sha,
        dataset_version_sha256=r.sha(DATA/'dataset-version.json'),script_sha256=r.sha(__file__),
        manifests={s:version['manifests'][s] for s in rows},encoder_sha256=r.PIN))
    encoder=r.IndicWav2VecExtractor(r.IndicWav2VecConfig(str(r.BASE),device='cuda',
        checkpoint_sha256=r.PIN,expected_embedding_dim=1024,backbone_id='indicwav2vec-large')).load()
    model=r.GruSpoofDetector(**r.MODEL).cuda()
    saved=r.restore_checkpoint(modelpath,model,map_location='cuda')
    assert saved['epoch']==3
    model.eval(); initial={k:v.detach().clone() for k,v in model.state_dict().items() if torch.is_tensor(v)}
    frozen_threshold=None; results={}; clock=time.perf_counter()
    for split,rs in rows.items():
        if split=='test':
            assert frozen_threshold is not None
            r.write(OUT/'test-exposure.json',dict(status='evaluation_started',checkpoint_sha256=original_sha,
                threshold=frozen_threshold,manifest_sha256=version['manifests']['test']['sha256']))
        scores=[]
        for i in range(0,len(rs),16):
            chunk=rs[i:i+16]; audio=[]
            for row in chunk:
                path=(DATA/row['prepared_audio']['path']).resolve()
                assert path.is_relative_to(DATA.resolve())
                assert r.sha(path)==row['prepared_audio']['sha256']
                a,rate=r.sf.read(path,dtype='float32')
                assert rate==16000 and a.ndim==1 and len(a)==row['prepared_audio']['frames']
                assert 16000<=len(a)<=64000 and np.isfinite(a).all() and np.abs(a).max()<=1
                audio.append(torch.from_numpy(a))
            b=encoder.extract_batch(torch.nn.utils.rnn.pad_sequence(audio,batch_first=True),
                                    torch.tensor([len(a) for a in audio]))
            with torch.no_grad():
                p=model(b.features,b.valid_lengths).softmax(-1)[:,1].cpu()
            assert torch.isfinite(p).all(); scores.append(p)
            if i%512==0:
                print(json.dumps(dict(stage=split,completed=min(i+16,len(rs)),total=len(rs))),flush=True)
        scores=torch.cat(scores)
        np.save(OUT/f'{split}-scores.npy',scores.numpy())
        if split=='calibration':
            frozen_threshold=threshold(scores,[x['label'] for x in rs])
            r.write(OUT/'calibration-decision.json',dict(threshold=frozen_threshold,
                objective='minimum balanced error; tie closest to 0.5',checkpoint_sha256=original_sha,
                calibration_manifest_sha256=version['manifests']['calibration']['sha256']))
        results[split]=report(scores,rs,frozen_threshold)
        r.write(OUT/f'{split}-predictions.jsonl.json',[
            dict(window_id=row['window_id'],language=row['spoken_language'],label=row['label'],
                 score=float(s),predicted=int(float(s)>=frozen_threshold)) for row,s in zip(rs,scores)])
        r.write(OUT/f'{split}-result.json',results[split])
    assert r.sha(modelpath)==original_sha
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in initial.items())
    calibrated=r.torch.load(modelpath,map_location='cpu',weights_only=False)
    calibrated['metadata'].update(threshold=frozen_threshold,calibrated=True,
        calibration_manifest_sha256=version['manifests']['calibration']['sha256'])
    torch.save(calibrated,OUT/'gru_calibrated.pt')
    fresh=r.GruSpoofDetector(**r.MODEL).cuda()
    r.restore_checkpoint(OUT/'gru_calibrated.pt',fresh,map_location='cuda'); fresh.eval()
    assert all(torch.equal(v,fresh.state_dict()[k]) for k,v in initial.items())
    with torch.no_grad():
        replay=fresh(b.features,b.valid_lengths).softmax(-1)[:,1].cpu()
    torch.testing.assert_close(replay,scores[-len(replay):],rtol=0,atol=0)
    r.write(OUT/'result.json',dict(threshold=frozen_threshold,epoch=3,results=results,
        unchanged_parameters=True,checkpoint_reload_max_error=float((replay-scores[-len(replay):]).abs().max()),
        original_checkpoint_sha256=original_sha,calibrated_checkpoint_sha256=r.sha(OUT/'gru_calibrated.pt'),
        seconds=time.perf_counter()-clock,missing_languages=['Bengali','Malayalam','Odia']))
    r.write(OUT/'test-exposure.json',dict(status='evaluated_benchmark',threshold=frozen_threshold,
        checkpoint_sha256=original_sha,result_sha256=r.sha(OUT/'result.json')))
    print(json.dumps(dict(status='complete',threshold=frozen_threshold,results={s:x['overall'] for s,x in results.items()})),flush=True)

if __name__=='__main__': main()
