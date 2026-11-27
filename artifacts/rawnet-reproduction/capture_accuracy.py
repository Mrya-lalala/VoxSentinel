"""Retain the highest-accuracy completed checkpoint during the ongoing run."""
import torch
from prepare import ROOT
from train import save

directory = ROOT / "run-50"
state = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
metrics = state["history"][-1]["dev"]
assert state["epoch"] == state["history"][-1]["epoch"]
target = directory / "best_accuracy.pt"
previous = torch.load(target, map_location="cpu", weights_only=True) if target.exists() else None
correct = metrics["true_positive"] + metrics["true_negative"]
previous_correct = (
    previous["metrics"]["true_positive"] + previous["metrics"]["true_negative"]
    if previous is not None else -1
)
if correct > previous_correct:
    save(target, {
        "model": state["model"], "epoch": state["epoch"],
        "metrics": metrics, "data_sha256": state["data_sha256"],
    })
    print("Accuracy candidate retained", state["epoch"], metrics, flush=True)


def joint_rank(value):
    total = sum(value[key] for key in (
        "true_positive", "true_negative", "false_positive", "false_negative"
    ))
    error_rate = (value["false_positive"] + value["false_negative"]) / total
    eer = round(value["eer"], 12)
    return max(eer / .061, error_rate / .06), eer, error_rate


joint_path = directory / "best_joint.pt"
joint = torch.load(joint_path, map_location="cpu", weights_only=True) if joint_path.exists() else None
if joint is None or joint_rank(metrics) < joint_rank(joint["metrics"]):
    save(joint_path, {
        "model": state["model"], "epoch": state["epoch"],
        "metrics": metrics, "data_sha256": state["data_sha256"],
        "selection": "closest joint violation of IFD EER <= .061 and accuracy >= .94",
    })
    print("Joint candidate retained", state["epoch"], metrics, flush=True)
