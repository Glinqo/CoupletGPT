"""Paired fixed-budget SFT: one pretrained start vs random starts, three seeds."""

import hashlib, json, math, pathlib, random, time
from dataclasses import asdict
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import torch
from torch.nn import functional as F
from couplet_gpt.data.datasets import ROOT, CoupletDataset
from couplet_gpt.modeling.gpt import GPT, GPTConfig, loss_sum
from couplet_gpt.training.pretrain import seed_all, atomic_save, json_write

RUN = ROOT / "runs/couplet_comparison_v1"
SEEDS = [2026, 2027, 2028]
EPOCHS = 3
BATCH = 256


def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def batch(ds, idx, device):
    x = ds.x[idx]
    y = ds.y[idx]
    width = int(ds.lengths[idx].max())
    x = torch.from_numpy(x[:, :width].astype(np.int64)).pin_memory().to(device, non_blocking=True)
    y = torch.from_numpy(y[:, :width].astype(np.int64)).pin_memory().to(device, non_blocking=True)
    return x, y


@torch.inference_mode()
def evaluate(model, ds, per_sample=False):
    model.eval()
    device = next(model.parameters()).device
    nll = char_nll = 0.0
    count = chars = correct = 0
    records = []
    for start in range(0, len(ds), BATCH):
        idx = np.arange(start, min(start + BATCH, len(ds)))
        x, y = batch(ds, idx, device)
        mask = y != -100
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model(x)
        flat_z = z[mask].float()
        flat_y = y[mask]
        losses = F.cross_entropy(flat_z, flat_y, reduction="none")
        cm = flat_y != 3
        nll += float(losses.sum())
        count += len(losses)
        char_nll += float(losses[cm].sum())
        chars += int(cm.sum())
        correct += int((flat_z.argmax(-1)[cm] == flat_y[cm]).sum())
        if per_sample:
            row = mask.nonzero()[:, 0]
            b = len(idx)
            sums = torch.zeros(b, device=device).scatter_add_(0, row, losses)
            cs = torch.zeros(b, device=device).scatter_add_(0, row[cm], losses[cm])
            nc = torch.zeros(b, device=device).scatter_add_(0, row[cm], torch.ones_like(losses[cm]))
            nv = torch.zeros(b, device=device).scatter_add_(0, row, torch.ones_like(losses))
            for i, a, c, n, v in zip(
                idx, sums.cpu().tolist(), cs.cpu().tolist(), nc.cpu().tolist(), nv.cpu().tolist()
            ):
                records.append(
                    {
                        "index": int(i),
                        "nll": a,
                        "valid_tokens": int(v),
                        "char_nll": c,
                        "characters": int(n),
                        "char_ce": c / n,
                    }
                )
    result = {
        "ce": nll / count,
        "ppl": math.exp(nll / count),
        "char_ce": char_nll / chars,
        "char_ppl": math.exp(char_nll / chars),
        "teacher_forced_char_accuracy": correct / chars,
        "valid_tokens": count,
        "characters": chars,
    }
    return (result, records) if per_sample else result


def train_one(condition, seed, train, validation):
    folder = RUN / f"{condition}_seed{seed}"
    folder.mkdir(exist_ok=True)
    if (folder / "result.json").exists():
        r = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        if r.get("status") == "complete":
            print("REUSE COMPLETE", folder.name, flush=True)
            return r
    assert not (folder / "training.jsonl").exists(), "Partial run must not be silently overwritten"
    seed_all(seed)
    model = GPT().cuda()
    if condition == "pretrained":
        ck = torch.load(
            ROOT / "runs/pretrain_v1/artifacts/best_model.pt", map_location="cpu", weights_only=True
        )
        model.load_state_dict(ck["model"])
        del ck
    optimizer = model.optimizer()
    seed_all(seed)  # Identical dropout stream across conditions.
    total_steps = math.ceil(len(train) / BATCH) * EPOCHS
    warmup = math.ceil(total_steps * 0.03)
    config = {
        "condition": condition,
        "seed": seed,
        "epochs": EPOCHS,
        "batch_size": BATCH,
        "total_steps": total_steps,
        "warmup_steps": warmup,
        "peak_lr": 3e-4,
        "min_lr": 3e-5,
        "optimizer": "AdamW, betas=(.9,.95), eps=1e-8",
        "decay": ".1 for attention/FFN matrices only",
        "precision": "BF16 autocast, FP32 parameters/loss",
        "loss": "lower characters plus EOS only; normalized by valid target count",
        "model_config": asdict(model.config),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "initialization": "random"
        if condition == "scratch"
        else "runs/pretrain_v1/artifacts/best_model.pt",
        "vocab_sha256": digest(ROOT / "tokenizer/vocab.json"),
        "source_code_sha256": digest(pathlib.Path(__file__)),
        "model_source_sha256": digest(ROOT / "couplet_gpt/modeling/gpt.py"),
    }
    json_write(folder / "config.json", config)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    initial = evaluate(model, validation)
    best = float("inf")
    best_epoch = 0
    step = 0
    seen = 0
    supervised = 0
    evals = [{"epoch": 0, "step": 0, "validation": initial}]
    metrics = (folder / "training.jsonl").open("w", encoding="utf-8", buffering=1)
    order_hashes = []
    for epoch in range(1, EPOCHS + 1):
        order = np.random.default_rng(seed + epoch * 10000).permutation(len(train))
        order_hashes.append(hashlib.sha256(order.tobytes()).hexdigest())
        model.train()
        total_loss = 0.0
        total_valid = 0
        epoch_started = time.perf_counter()
        for start in range(0, len(order), BATCH):
            idx = order[start : start + BATCH]
            x, y = batch(train, idx, "cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                summed = loss_sum(logits, y)
                valid = int((y != -100).sum())
                loss = summed / valid
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("nonfinite gradient")
            step += 1
            if step <= warmup:
                lr = 3e-4 * step / warmup
            else:
                progress = (step - warmup) / (total_steps - warmup)
                lr = 3e-5 + 0.5 * (3e-4 - 3e-5) * (1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            seen += len(idx)
            supervised += valid
            v = float(summed.detach())
            total_loss += v
            total_valid += valid
            if step % 50 == 0 or start + len(idx) == len(order):
                row = {
                    "epoch": epoch,
                    "step": step,
                    "examples_seen": seen,
                    "supervised_tokens": supervised,
                    "batch_ce": v / valid,
                    "epoch_running_ce": total_loss / total_valid,
                    "lr": lr,
                    "gradient_norm": float(norm),
                    "elapsed_seconds": time.perf_counter() - started,
                }
                metrics.write(json.dumps(row) + "\n")
            if step % 400 == 0:
                print(
                    condition,
                    seed,
                    "epoch",
                    epoch,
                    "step",
                    step,
                    "ce",
                    round(total_loss / total_valid, 4),
                    flush=True,
                )
        val = evaluate(model, validation)
        row = {
            "epoch": epoch,
            "step": step,
            "validation": val,
            "train_epoch_ce": total_loss / total_valid,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        evals.append(row)
        json_write(folder / "evaluation.json", evals)
        if val["ce"] < best:
            best = val["ce"]
            best_epoch = epoch
            atomic_save(
                folder / "best_model.pt",
                {
                    "model": model.state_dict(),
                    "model_config": asdict(model.config),
                    "condition": condition,
                    "seed": seed,
                    "epoch": epoch,
                    "validation": val,
                    "vocab_sha256": config["vocab_sha256"],
                },
            )
        # Resume/inspection state, even though scheduled experiment is only three epochs.
        atomic_save(
            folder / "latest.pt",
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_config": asdict(model.config),
                "epoch": epoch,
                "step": step,
                "examples_seen": seen,
                "supervised_tokens": supervised,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "config": config,
            },
        )
        print(
            "EPOCH COMPLETE",
            condition,
            seed,
            epoch,
            "validation CE",
            round(val["ce"], 5),
            "seconds",
            round(time.perf_counter() - epoch_started, 1),
            flush=True,
        )
    metrics.close()
    result = {
        "status": "complete",
        "condition": condition,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_ce": best,
        "initial_validation": initial,
        "steps": step,
        "examples_seen": seen,
        "supervised_tokens": supervised,
        "order_hashes": order_hashes,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
    }
    json_write(folder / "result.json", result)
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def main():
    RUN.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    ready = json.loads((ROOT / "PRETRAIN_READY.json").read_text(encoding="utf-8"))
    assert ready["status"] == "READY"
    # Verify actual tensors to be used without altering frozen data-v1 files.
    data = {
        s: digest(ROOT / "data/tokenized" / f"couplet_{s}.npz")
        for s in ("train", "validation", "test")
    }
    json_write(
        RUN / "experiment.json",
        {
            "seeds": SEEDS,
            "epochs": EPOCHS,
            "batch": BATCH,
            "pretrained_checkpoint_sha256": digest(
                ROOT / "runs/pretrain_v1/artifacts/best_model.pt"
            ),
            "dataset_tensor_sha256": data,
            "model_parameters": 8154624,
            "selection": "validation CE only",
            "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        },
    )
    train = CoupletDataset("train")
    validation = CoupletDataset("validation")
    results = []
    for seed in SEEDS:
        for condition in ("pretrained", "scratch"):
            result = train_one(condition, seed, train, validation)
            results.append(result)
            json_write(
                RUN / "status.json",
                {"status": "training", "completed_runs": len(results), "runs": results},
            )
    for seed in SEEDS:
        a, b = [r for r in results if r["seed"] == seed]
        assert (
            a["order_hashes"] == b["order_hashes"]
            and a["steps"] == b["steps"]
            and a["supervised_tokens"] == b["supervised_tokens"]
        )
    json_write(
        RUN / "training_complete.json",
        {
            "status": "complete",
            "runs": results,
            "fairness_checks": "same order, updates, samples and supervised tokens per paired seed",
        },
    )
    print("ALL SIX TRAINING RUNS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
