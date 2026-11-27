"""Test training-only BatchNorm recalibration of a saved RawNet3 checkpoint."""
import json

import torch
from model import create_model
from prepare import ROOT, digest, load_data, write_json
from train import evaluate, save

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
data, ready = load_data()
path = ROOT / "run-50/best.pt"
source_hash = digest(path)
state = torch.load(path, map_location="cpu", weights_only=True)
model = create_model().cuda().eval().requires_grad_(False)
model.load_state_dict(state["model"])
before_scores, before = evaluate(model, *data["dev"], 16)
torch.testing.assert_close(before_scores, state["scores"], atol=1e-6, rtol=0)
layers = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
for layer in layers:
    layer.reset_running_stats()
    layer.momentum = None
    layer.train()
indices = torch.randperm(len(data["train"][0]), generator=torch.Generator().manual_seed(42))
with torch.no_grad():
    for start in range(0, len(indices), 64):
        group = indices[start:start + 64]
        if len(group) < 2:
            break
        model(data["train"][0][group].cuda())
        if (start // 64 + 1) % 64 == 0:
            print("Training-only BN statistics", min(start + 64, len(indices)), "/", len(indices), flush=True)
after_scores, after = evaluate(model, *data["dev"], 16)
output = ROOT / "batchnorm-diagnostic"
output.mkdir(exist_ok=True)
write_json(output / "result.json", {
    "source_epoch": state["epoch"], "source_checkpoint_sha256": source_hash,
    "data_sha256": state["data_sha256"], "before": before, "after": after,
    "recalibration": "all training examples, shuffled, batch 64, cumulative BN statistics",
    "parameter_updates": False,
})
def rank(metrics):
    total = sum(metrics[key] for key in (
        "true_positive", "true_negative", "false_positive", "false_negative"
    ))
    errors = metrics["false_positive"] + metrics["false_negative"]
    eer = round(metrics["eer"], 12)
    error_rate = errors / total
    return max(eer / .061, error_rate / .06), eer, error_rate


if rank(after) < rank(before):
    save(output / "improved.pt", {
        "model": model.state_dict(), "scores": after_scores, "metrics": after,
        "data_sha256": state["data_sha256"], "source_checkpoint_sha256": source_hash,
    })
print(json.dumps({"before": before, "after": after}), flush=True)
