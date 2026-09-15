import torch

from voxsentinel.training import train_epoch, validate_epoch


def test_train_and_validate_epoch(detector, batches):
    optimizer = torch.optim.SGD(detector.parameters(), lr=0.1)
    trained = train_epoch(detector, batches, optimizer)
    validated = validate_epoch(detector, batches)
    assert trained.steps == validated.steps == 2
    assert trained.examples == validated.examples == 4
    assert trained.loss >= 0 and validated.loss >= 0
