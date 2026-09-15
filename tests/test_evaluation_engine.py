from voxsentinel.evaluation import evaluate


def test_evaluate_aggregates_scores(detector, batches):
    result = evaluate(detector, batches)
    assert result.scores.shape == result.labels.shape == (4,)
    assert 0 <= result.metrics.accuracy <= 1
