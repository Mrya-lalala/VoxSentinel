"""Create a local, self-describing pilot release without publishing weights."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil

import yaml
from src.dataset_prep.materialize import sha256_file
from src.dataset_prep.config import load_prep_config
from src.detectors.checkpoints import load_checkpoint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=Path('artifacts/runs/gru-core-v2-standardized'))
    parser.add_argument('--out-dir',type=Path,default=Path('artifacts/releases/indic-gru-pilot-v0.1'))
    args=parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()): raise FileExistsError(args.out_dir)
    checkpoint=args.run_dir/'gru_best.pt';payload=load_checkpoint(checkpoint,model_name='gru')
    prep=load_prep_config();encoder=yaml.safe_load(Path('configs/backbones.yaml').read_text())['indic_wav2vec']
    dataset=json.loads((args.run_dir/'dataset_identity.json').read_text())
    if 'splits' in dataset:
        enc_identity=dataset['splits']['train']['cache_identity']['encoder']
        identity={'encoder_checkpoint_sha256':enc_identity['checkpoint_sha256'],
                  'output_layer':enc_identity['output_layer'],
                  'preprocessing_version':enc_identity['preprocessing_version']}
    else:
        identity=dataset['train']['cache_identity']
    if identity['encoder_checkpoint_sha256']!=encoder['checkpoint_sha256'] or identity['preprocessing_version']!=prep.preprocessing['config_version'] or identity['output_layer']!=encoder['output_layer']:
        raise ValueError('Current configuration differs from frozen run')
    expected_head_identity={'checkpoint_sha256':encoder['checkpoint_sha256'], 'selected_layer':encoder['output_layer'],
                            'embedding_dim':encoder['expected_embedding_dim'], 'dtype':'float32',
                            'preprocessing_version':identity['preprocessing_version']}
    if payload.get('metadata',{}).get('encoder') != expected_head_identity:
        raise ValueError('Head metadata is incompatible with the pinned inference identity')
    args.out_dir.mkdir(parents=True)
    shutil.copyfile(checkpoint,args.out_dir/'gru_best.pt')
    spec={'schema':'voxsentinel.inference_release.v1','release_id':args.out_dir.name,
          'validation_status':'pilot development only; not production validated',
          'head':{'file':'gru_best.pt','sha256':sha256_file(checkpoint),'epoch':payload['epoch']},
          'encoder':encoder,'preprocessing_version':identity['preprocessing_version'],
          'window':asdict(prep.window),'threshold':0.5,
          'training_languages':[s.lower() for s in prep.quotas['core']['languages']],
          'training_run':str(args.run_dir),'training_identity':dataset,
          'limits':['No Indian-English training yet','Single selected window, not whole-file localization','Uncalibrated score','Gender and source-condition coverage gaps'],
          'usage':'Research pilot; source datasets retain their terms, including IndicSynth CC BY-NC 4.0; no commercial clearance asserted'}
    (args.out_dir/'release.json').write_text(json.dumps(spec,indent=2)+'\n')
    print(args.out_dir/'release.json')

if __name__=='__main__': main()
