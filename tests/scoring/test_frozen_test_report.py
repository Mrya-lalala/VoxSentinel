import pytest
from scripts.evaluate_frozen_v2_test import summarize, bootstrap


def row(label, score, component):
    return {'label':label,'synthetic_score':score,'logits':[0., 2*score-1], 'component':component}


def test_fixed_threshold_and_ranking():
    result=summarize([row(0,.1,'a'),row(0,.5,'b'),row(1,.4,'c'),row(1,.9,'d')])
    assert [result[k] for k in ('true_negative','false_positive','false_negative','true_positive')]==[1,1,1,1]
    assert result['accuracy']==.5
    assert result['auc_margin']==.75


def test_single_class_rates_not_invented():
    result=summarize([row(0,.8,'a')])
    assert result['genuine_false_alarm_rate']==1
    assert result['spoof_miss_rate'] is None
    assert result['eer_margin'] is None
    assert result['auc_margin'] is None


def test_cluster_bootstrap_is_deterministic():
    rows=[row(0,.1,'shared'),row(1,.9,'shared'),row(0,.2,'other')]
    a=bootstrap(rows,draws=100)
    assert a==bootstrap(rows,draws=100)
    assert a['components']==2
    assert a['intervals']['accuracy']==[1.,1.]
