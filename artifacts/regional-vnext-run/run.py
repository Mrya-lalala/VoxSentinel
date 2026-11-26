"""Bounded local experiment; frozen train/dev manifests, disk-backed features."""
import argparse
import gc
import hashlib
import json
import random
import shutil
import sys
import time
import traceback
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import soundfile as sf
import torch
from src.backbones.indic_wav2vec import IndicWav2VecConfig, IndicWav2VecExtractor
from src.data.batch import collate_examples, EmbeddingExample
from src.detectors.gru import GruSpoofDetector
from src.detectors.checkpoints import save_checkpoint, restore_checkpoint
from src.scoring.metrics import binary_metrics

OUT = Path(__file__).resolve().parent
DATA = ROOT.parents[1] / 'datasets-regional-vnext'
BASE = ROOT.parents[2] / 'models/indicwav2vec_large.pt'
PIN = '26bb5ada18952fd7355f691d25927b34a0e46d6afda7658bf5c254621831ab59'
MODEL = dict(input_dim=1024, hidden_size=256, num_layers=1, dropout=0.,
             num_classes=2, feature_standardization=True)

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def write(path, value):
    tmp = Path(str(path)+'.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
    tmp.replace(path)

def status(stage, **kw):
    value = dict(stage=stage, time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw)
    write(OUT/'status.json', value)
    print(json.dumps(value), flush=True)

def manifests():
    version = json.loads((DATA/'dataset-version.json').read_text())
    assert sha(DATA/'validation.json') == version['validation_sha256']
    rows = {}
    for split in ('train', 'dev'):
        entry = version['manifests'][split]
        path = DATA/entry['path']
        assert sha(path) == entry['sha256'], split
        rows[split] = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        assert len(rows[split]) == entry['rows']
        assert all(r['split'] == split and r['preprocessing_version']=='voxsentinel-prep-2' for r in rows[split])
    assert set(r['window_id'] for r in rows['train']).isdisjoint(r['window_id'] for r in rows['dev'])
    return version, rows

def extract(rows, identity):
    paths = [OUT/f'{s}.ready.json' for s in rows]
    if all(p.exists() for p in paths):
        for p in paths: assert json.loads(p.read_text())['identity'] == identity
        return
    need = sum(len(rs)*199*1024*4 for s,rs in rows.items() if not (OUT/f'{s}.features.npy').exists())
    assert shutil.disk_usage(OUT).free > need + 5*2**30, 'Insufficient cache disk space'
    encoder = IndicWav2VecExtractor(IndicWav2VecConfig(str(BASE), device='cuda',
        checkpoint_sha256=PIN, expected_embedding_dim=1024, backbone_id='indicwav2vec-large')).load()
    for split, rs in rows.items():
        marker=OUT/f'{split}.progress.json'; ready=OUT/f'{split}.ready.json'
        if ready.exists():
            assert json.loads(ready.read_text())['identity']==identity
            continue
        start=0
        if marker.exists():
            p=json.loads(marker.read_text()); assert p['identity']==identity; start=p['completed']
        mode='r+' if marker.exists() else 'w+'
        x=np.lib.format.open_memmap(OUT/f'{split}.features.npy', mode=mode, dtype=np.float32, shape=(len(rs),199,1024))
        lengths=np.lib.format.open_memmap(OUT/f'{split}.lengths.npy', mode=mode, dtype=np.int64, shape=(len(rs),))
        clock=time.perf_counter()
        for i in range(start,len(rs),16):
            batch=rs[i:i+16]; audio=[]
            for r in batch:
                path=(DATA/r['prepared_audio']['path']).resolve()
                assert path.is_relative_to(DATA.resolve())
                assert sha(path)==r['prepared_audio']['sha256'], str(path)
                a,rate=sf.read(path,dtype='float32')
                assert rate==16000 and a.ndim==1 and 16000<=len(a)<=64000
                assert len(a)==r['prepared_audio']['frames'] and np.isfinite(a).all() and np.abs(a).max()<=1
                audio.append(torch.from_numpy(a))
            wave=torch.nn.utils.rnn.pad_sequence(audio,batch_first=True)
            b=encoder.extract_batch(wave,torch.tensor([len(a) for a in audio]))
            for j,n in enumerate(b.valid_lengths.tolist()):
                assert 0<n<=199
                x[i+j]=0; x[i+j,:n]=b.features[j,:n].cpu().numpy(); lengths[i+j]=n
            if (i//16)%32==0 or i+16>=len(rs):
                x.flush(); lengths.flush()
                done=min(i+16,len(rs)); write(marker,dict(identity=identity,completed=done))
                status('extracting', split=split, completed=done,total=len(rs),
                       clips_per_second=round((done-start)/(time.perf_counter()-clock),2))
        x.flush(); lengths.flush(); del x,lengths
        write(ready,dict(identity=identity,rows=len(rs),
            features_sha256=sha(OUT/f'{split}.features.npy'), lengths_sha256=sha(OUT/f'{split}.lengths.npy')))
    del encoder,b
    gc.collect(); torch.cuda.empty_cache()

class Examples:
    def __init__(self,split,rows):
        self.split=split
        self.x=None
        self.lengths=np.load(OUT/f'{split}.lengths.npy')
        self.rows=rows
    def release(self):
        if self.x is not None:
            self.x._mmap.close()
            self.x=None
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        if i>=len(self): raise IndexError(i)
        # Bound mapped resident pages during both sequential standardizer fitting
        # and shuffled training. Returned tensors own their data before unmapping.
        if i%256==0: self.release()
        if self.x is None:
            self.x=np.load(OUT/f'{self.split}.features.npy',mmap_mode='r')
        n=int(self.lengths[i]); assert 0<n<=199
        return EmbeddingExample(torch.from_numpy(self.x[i,:n].copy()),self.rows[i]['label'])

def batch(examples, indices):
    try:
        result=collate_examples([examples[int(i)] for i in indices])
    finally:
        examples.release()
    return {k:v.to('cuda') for k,v in result.items()}

@torch.no_grad()
def predict(model,examples):
    model.eval(); logits=[]
    for i in range(0,len(examples),32):
        b=batch(examples,range(i,min(i+32,len(examples))))
        logits.append(model(b['features'],b['valid_lengths']).cpu())
    return torch.cat(logits).softmax(-1)[:,1]

def metrics(scores,rows):
    labels=torch.tensor([r['label'] for r in rows])
    result=dict(overall=asdict(binary_metrics(scores,labels)))
    result['languages']={}
    for lang in sorted({r['spoken_language'] for r in rows}):
        ix=[i for i,r in enumerate(rows) if r['spoken_language']==lang]
        result['languages'][lang]=dict(windows=len(ix),**asdict(binary_metrics(scores[ix],labels[ix])))
    return result

def smoke():
    torch.manual_seed(0)
    examples=[EmbeddingExample(torch.randn(n,1024),i%2) for i,n in enumerate((9,5,7,3))]
    m=GruSpoofDetector(**MODEL); m.standardizer.fit(examples)
    b=collate_examples(examples); m.eval()
    a=m(b['features'],b['valid_lengths'])
    poisoned=b['features'].clone(); poisoned[b['padding_mask']]=123
    torch.testing.assert_close(a,m(poisoned,b['valid_lengths']),rtol=0,atol=0)
    torch.nn.functional.cross_entropy(a,b['labels']).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    path=OUT/'smoke.pt'; save_checkpoint(path,m,'gru',MODEL)
    fresh=GruSpoofDetector(**MODEL); restore_checkpoint(path,fresh); fresh.eval()
    torch.testing.assert_close(a,fresh(b['features'],b['valid_lengths']),rtol=0,atol=0)
    status('smoke_passed',padding_invariance=True,finite_gradients=True,checkpoint_roundtrip=True)

def main():
    torch.set_num_threads(8); torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if '--smoke' in sys.argv: smoke(); return
    resume='--resume' in sys.argv
    assert resume or not (OUT/'history.json').exists(), 'Run already trained; use --resume'
    version,rows=manifests()
    identity=dict(dataset_version_sha256=sha(DATA/'dataset-version.json'),encoder_sha256=PIN,
                  selected_layer=None,dtype='float32',preprocessing='voxsentinel-prep-2')
    settings=dict(identity=identity,model=MODEL,epochs=10,batch_size=32,seed=0,
        optimizer=dict(name='AdamW',lr=.001,weight_decay=0),threshold=.5,selection='earliest minimum dev EER',
        manifests={s:version['manifests'][s] for s in ('train','dev')},script_sha256=sha(__file__),
        heldout_policy='Calibration and test excluded from feature extraction and training')
    if resume:
        old=json.loads((OUT/'settings.json').read_text())
        assert old['identity']==identity and old['model']==MODEL
        write(OUT/'resume.json',dict(script_sha256=sha(__file__),reason='Bound mmap residency per batch',original_script_sha256=old['script_sha256']))
    else:
        write(OUT/'settings.json',settings)
    status('preflight_passed',train=len(rows['train']),dev=len(rows['dev']))
    extract(rows,identity)
    examples={s:Examples(s,rs) for s,rs in rows.items()}
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    model=GruSpoofDetector(**MODEL)
    if not resume:
        status('fitting_train_standardizer')
        model.standardizer.fit(examples['train'],cache_signature=json.loads((OUT/'train.ready.json').read_text())['features_sha256'])
        examples['train'].release()
    model=model.cuda(); opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0)
    history=[]; best=float('inf'); clock=time.perf_counter(); step=0
    first_epoch=1; previous_seconds=0
    if resume:
        saved=restore_checkpoint(OUT/'last.pt',model,optimizer=opt,map_location='cuda')
        first_epoch=saved['epoch']+1; step=saved['global_step']
        history=[h for h in json.loads((OUT/'history.json').read_text()) if h['epoch']<first_epoch]
        best=min(h['dev']['overall']['eer'] for h in history)
        previous_seconds=history[-1]['seconds']
        status('resumed',saved_epoch=saved['epoch'])
    for epoch in range(first_epoch,11):
        model.train(); total=0.; count=0
        order=np.random.default_rng(epoch).permutation(len(examples['train']))
        for i in range(0,len(order),32):
            b=batch(examples['train'],order[i:i+32]); opt.zero_grad(set_to_none=True)
            logits=model(b['features'],b['valid_lengths'])
            loss=torch.nn.functional.cross_entropy(logits,b['labels'])
            assert torch.isfinite(loss); loss.backward(); opt.step(); step+=1
            total+=float(loss.detach())*len(b['labels']); count+=len(b['labels'])
            if i%8192==0: status('training',epoch=epoch,completed=count,total=len(order))
        scores=predict(model,examples['dev']); report=metrics(scores,rows['dev'])
        entry=dict(epoch=epoch,train_loss=total/count,dev=report,seconds=previous_seconds+time.perf_counter()-clock)
        history.append(entry); write(OUT/'history.json',history)
        meta=dict(encoder=dict(checkpoint_sha256=PIN,selected_layer=None,embedding_dim=1024,
            preprocessing_version='voxsentinel-prep-2',dtype='float32'),dataset=identity,threshold=.5,
            label_mapping={'0':'genuine','1':'synthetic'},selection='minimum dev EER',calibrated=False)
        save_checkpoint(OUT/'last.pt',model,'gru',MODEL,optimizer=opt,epoch=epoch,global_step=step,
                        metrics=report['overall'],metadata=meta)
        eer=report['overall']['eer']
        if eer<best-1e-12:
            best=eer
            save_checkpoint(OUT/'best.pt',model,'gru',MODEL,epoch=epoch,global_step=step,
                            metrics=report['overall'],metadata=meta)
            np.save(OUT/'best_dev_scores.npy',scores.numpy())
        status('epoch_complete',epoch=epoch,train_loss=total/count,dev=report['overall'])
    fresh=GruSpoofDetector(**MODEL).cuda()
    checkpoint=restore_checkpoint(OUT/'best.pt',fresh,map_location='cuda')
    scores=predict(fresh,examples['dev'])
    expected=torch.from_numpy(np.load(OUT/'best_dev_scores.npy'))
    torch.testing.assert_close(scores,expected,rtol=0,atol=0)
    write(OUT/'result.json',dict(epoch=checkpoint['epoch'],dev=metrics(scores,rows['dev']),
        checkpoint_sha256=sha(OUT/'best.pt'),reload_max_error=float((scores-expected).abs().max()),
        seconds=time.perf_counter()-clock,calibration_and_test_scored=False))
    status('complete',best_epoch=checkpoint['epoch'],dev_eer=best)

if __name__=='__main__':
    try: main()
    except Exception:
        status('failed',error=traceback.format_exc()); raise
