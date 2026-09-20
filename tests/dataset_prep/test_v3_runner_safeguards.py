import copy
from src.config import load_config
from scripts.train_gru_v3_from_cache import DEFAULT_CONFIGS, _resolved_settings, _initial_comparison_check, checkpoint_encoder_identity


def test_resolved_raw_config_uses_nested_merge():
    s=_resolved_settings(load_config(DEFAULT_CONFIGS),DEFAULT_CONFIGS)
    assert s['raw_merged_config']['model']==s['resolved']['model']
    assert s['raw_merged_config']['training']==s['resolved']['training']


def test_architecture_and_execution_changes_rejected():
    s=_resolved_settings(load_config(DEFAULT_CONFIGS),DEFAULT_CONFIGS)
    assert _initial_comparison_check(s)['matches_initial_comparison']
    for key,value in [('shuffle',False),('use_amp',True),('device','cuda')]:
        changed=copy.deepcopy(s);changed['resolved']['training'][key]=value
        assert not _initial_comparison_check(changed)['matches_initial_comparison']
    changed=copy.deepcopy(s);changed['resolved']['model']['parameters']['hidden_size']=128
    assert not _initial_comparison_check(changed)['matches_initial_comparison']


def test_checkpoint_identity_matches_predictor_contract():
    source={'checkpoint_sha256':'digest','output_layer':None,'embedding_dim':1024,
            'preprocessing_version':'voxsentinel-prep-2','dtype':'float32','backbone_id':'extra-cache-field'}
    assert checkpoint_encoder_identity(source)=={'checkpoint_sha256':'digest','selected_layer':None,
        'embedding_dim':1024,'preprocessing_version':'voxsentinel-prep-2','dtype':'float32'}
