import torch

from voxsentinel.checkpoints import load_checkpoint, restore_checkpoint, save_checkpoint


def test_checkpoint_round_trip(tmp_path, detector):
    optimizer = torch.optim.SGD(detector.parameters(), lr=0.1)
    path = tmp_path / "model.pt"
    save_checkpoint(path, detector, "dummy", {"width": 1}, optimizer=optimizer, epoch=3, global_step=12)
    loaded = load_checkpoint(path, model_name="dummy")
    restored = type(detector)()
    restore_checkpoint(path, restored, model_name="dummy")
    assert loaded["epoch"] == 3
    for before, after in zip(detector.parameters(), restored.parameters()):
        assert torch.equal(before, after)
