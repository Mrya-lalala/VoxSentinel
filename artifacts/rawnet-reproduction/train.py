"""RawNet3 from-scratch IFD run following the published optimizer recipe."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
from time import perf_counter

import torch
from model import create_model
from prepare import CACHE, ROOT, digest, load_data, write_json

sys.path.insert(0, str(ROOT.parents[1]))
from src.scoring.metrics import binary_metrics


def save(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


@torch.no_grad()
def evaluate(model, waves, labels, microbatch):
    model.eval()
    scores = []
    for start in range(0, len(waves), microbatch):
        scores.append(model(waves[start:start + microbatch].cuda()).squeeze(1).sigmoid().cpu())
    scores = torch.cat(scores)
    assert torch.isfinite(scores).all()
    return scores, asdict(binary_metrics(scores, labels.long(), .5))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--microbatch", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.microbatch < 2 or args.microbatch > 64:
        raise ValueError("microbatch must be between 2 and 64")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data, ready = load_data()
    output = ROOT / "run-50"
    output.mkdir(exist_ok=True)
    identity = digest(CACHE / "ready.json")
    protocol = {
        "model": "official RawNet3, C=1024, scalar fc6 output",
        "upstream": json.loads((ROOT / "upstream-provenance.json").read_text()),
        "source_sha256": {p.name: digest(p) for p in (
            Path(__file__), ROOT / "model.py",
            ROOT / "upstream/python/RawNet3/models/RawNet3.py",
            ROOT / "upstream/python/RawNet3/models/RawNetBasicBlock.py",
        )},
        "data_sha256": identity, "seed": 42, "epochs": args.epochs,
        "optimizer": "Adam", "lr": .001, "weight_decay": .00005,
        "loss": "BCEWithLogitsLoss", "effective_batch": 64,
        "microbatch": args.microbatch, "precision": "FP32, TF32 disabled",
        "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(),
        "counts": {s: len(v[0]) for s, v in data.items()},
        "selection": "minimum dev EER, earliest epoch on ties; accuracy threshold .5",
        "differences": [
            "existing speaker-disjoint train/dev folders, 200 development examples",
            "published exact split files unavailable",
            "BatchNorm observes microbatches when microbatch < 64",
            "binary output projection nOut=1 is an implementation choice",
        ],
    }
    path = output / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == protocol
    else:
        write_json(path, protocol)
    model = create_model().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.00005, foreach=False)
    records, best_eer, start_epoch = [], float("inf"), 1
    last = output / "last.pt"
    if args.resume:
        state = torch.load(last, map_location="cpu", weights_only=True)
        assert state["data_sha256"] == identity
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        records, best_eer, start_epoch = state["history"], state["best_eer"], state["epoch"] + 1
        del state
    elif last.exists():
        raise RuntimeError("Use --resume for an existing run")
    train, labels = data["train"]
    started = perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        indices = list(range(len(train)))
        random.Random(42 + epoch).shuffle(indices)
        total_loss = 0.
        for start in range(0, len(indices), 64):
            group = indices[start:start + 64]
            optimizer.zero_grad(set_to_none=True)
            chunks = [group[i:i + args.microbatch] for i in range(0, len(group), args.microbatch)]
            if len(chunks[-1]) == 1:
                chunks[-2].extend(chunks.pop())
            for selected in chunks:
                logits = model(train[selected].cuda()).squeeze(1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, labels[selected].cuda(), reduction="sum"
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite RawNet3 loss")
                (loss / len(group)).backward()
                total_loss += float(loss.detach())
            optimizer.step()
            if (start // 64 + 1) % 64 == 0:
                print(f"epoch {epoch}: {min(start + 64, len(train))}/{len(train)}", flush=True)
        scores, metrics = evaluate(model, *data["dev"], args.microbatch)
        record = {
            "epoch": epoch, "train_loss": total_loss / len(train), "dev": metrics,
            "elapsed_seconds": perf_counter() - started,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        }
        records.append(record)
        if metrics["eer"] < best_eer:
            best_eer = metrics["eer"]
            save(output / "best.pt", {
                "model": model.state_dict(), "epoch": epoch, "metrics": metrics,
                "scores": scores, "data_sha256": identity,
            })
        write_json(output / "history.json", records)
        save(last, {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": epoch, "history": records, "best_eer": best_eer,
            "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
            "data_sha256": identity,
        })
        print(f"epoch {epoch}: loss {record['train_loss']:.5f}, "
              f"dev EER {metrics['eer']:.2%}, accuracy {metrics['accuracy']:.2%}", flush=True)
    selected = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(selected["model"])
    scores, metrics = evaluate(model, *data["dev"], args.microbatch)
    torch.testing.assert_close(scores, selected["scores"], atol=1e-6, rtol=0)
    write_json(output / "result.json", {
        "selected_epoch": selected["epoch"], "metrics": metrics,
        "checkpoint_sha256": digest(output / "best.pt"),
        "reload_verified": True, "protocol_sha256": digest(path),
    })
    print("Selected checkpoint reload verified", metrics, flush=True)


if __name__ == "__main__":
    main()
