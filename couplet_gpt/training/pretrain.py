"""Train the v1 GPT from random initialization; token-budgeted and resumable."""

from __future__ import annotations
import argparse, contextlib, hashlib, json, math, os, pathlib, platform, random, time
from dataclasses import asdict
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import torch
from couplet_gpt.data.datasets import ROOT, CharacterTokenizer, MixedPretrainStream, PretrainDataset
from couplet_gpt.modeling.gpt import GPT, GPTConfig, loss_sum, generate

WEIGHTS = {"tang": 0.3, "song": 0.3, "ci": 0.1, "classical": 0.3}
PROMPTS = ["春风", "山中", "江上", "明月", "昔者", "夫学者", "天下"]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_batch(rows, device):
    # Trim only trailing columns that are PAD for every row; positions and valid contexts unchanged.
    x = np.stack([r["input_ids"] for r in rows])
    y = np.stack([r["labels"] for r in rows])
    width = int(np.where((y != -100).any(axis=0))[0][-1]) + 1
    x = torch.from_numpy(x[:, :width].copy()).pin_memory().to(device, non_blocking=True)
    y = torch.from_numpy(y[:, :width].copy()).pin_memory().to(device, non_blocking=True)
    return x, y


def amp(device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def state_capture(
    model, optimizer, stream, body, valid, step, best, patience, config, next_probe, next_full
):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": asdict(model.config),
        "training_config": config,
        "body_tokens": body,
        "valid_tokens": valid,
        "step": step,
        "best_validation": best,
        "patience": patience,
        "next_probe": next_probe,
        "next_full": next_full,
        "sampler_rng": stream.rng.bit_generator.state,
        "source_body_tokens": stream.observed_body_tokens.copy(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }


def atomic_save(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.inference_mode()
def evaluate(model, split="validation", full=False, batch_size=32):
    model.eval()
    device = next(model.parameters()).device
    result = {}
    for source in WEIGHTS:
        ds = PretrainDataset(source, split)
        indices = (
            np.arange(len(ds))
            if full
            else np.sort(
                np.random.default_rng(2027).choice(len(ds), min(128, len(ds)), replace=False)
            )
        )
        total = 0.0
        count = 0
        for start in range(0, len(indices), batch_size):
            x, y = tensor_batch([ds[int(i)] for i in indices[start : start + batch_size]], device)
            with amp(device):
                logits = model(x)
            total += float(loss_sum(logits, y))
            count += int((y != -100).sum())
        result[source] = {
            "ce": total / count,
            "ppl": math.exp(total / count),
            "valid_tokens": count,
            "windows": len(indices),
        }
    result["weighted_ce"] = sum(WEIGHTS[s] * result[s]["ce"] for s in WEIGHTS)
    result["weighted_ppl"] = math.exp(result["weighted_ce"])
    return result


def samples(model):
    model.eval()
    tokenizer = CharacterTokenizer()
    result = []
    for p in PROMPTS:
        for temperature in (0.0, 0.8):
            ids = generate(model, [2] + tokenizer.encode(p), temperature=temperature)
            result.append(
                {
                    "prompt": p,
                    "temperature": temperature,
                    "top_k": 20 if temperature else None,
                    "seed": 2027,
                    "generated_ids": ids,
                    "continuation": tokenizer.decode(ids),
                }
            )
    return result


def train(args):
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.set_num_threads(4)
    seed_all(2026)
    device = torch.device("cuda")
    run = ROOT / "results/pretraining" / args.run
    run.mkdir(parents=True, exist_ok=True)
    (run / "checkpoints").mkdir(exist_ok=True)
    assert args.resume or not (run / "train.jsonl").exists(), "Use --resume for an existing run"
    cfg = GPTConfig()
    model = GPT(cfg).to(device)
    optimizer = model.optimizer()
    stream = MixedPretrainStream()
    body = valid = step = 0
    best = float("inf")
    patience = 0
    next_probe = 1_000_000
    next_full = 5_000_000
    ready = json.loads((ROOT / "data/pretraining/build_manifest.json").read_text(encoding="utf-8"))
    assert ready["status"] == "READY" and ready["vocab_size"] == cfg.vocab_size
    data_manifest = ROOT / "data/pretraining/output_manifest.json"
    assert sha(data_manifest) == ready["output_manifest_sha256"]
    config = {
        "budget_body_tokens": args.tokens,
        "microbatch": args.microbatch,
        "update_valid_tokens": 8192,
        "warmup_body_tokens": 1_000_000,
        "peak_lr": 3e-4,
        "min_lr": 3e-5,
        "seed": 2026,
        "precision": "FP32 parameters and AdamW state, BF16 autocast, FP32 CE; no GradScaler",
        "source_weights": WEIGHTS,
        "vocab_sha256": sha(ROOT / "tokenizer/vocab.json"),
        "dataset_manifest_sha256": sha(data_manifest),
        "model_config": asdict(cfg),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "python": platform.python_version(),
        "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_code_sha256": {
            p.name: sha(p)
            for p in [
                ROOT / "couplet_gpt/modeling/gpt.py",
                ROOT / "couplet_gpt/training/pretrain.py",
                ROOT / "couplet_gpt/data/datasets.py",
            ]
        },
    }
    if args.resume:
        ck = torch.load(run / "checkpoints/latest.pt", map_location="cpu", weights_only=False)
        assert ck["training_config"]["vocab_sha256"] == config["vocab_sha256"]
        assert ck["training_config"]["dataset_manifest_sha256"] == config["dataset_manifest_sha256"]
        assert ck["training_config"]["budget_body_tokens"] == args.tokens
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        body = ck["body_tokens"]
        valid = ck["valid_tokens"]
        step = ck["step"]
        best = ck["best_validation"]
        patience = ck["patience"]
        next_probe = ck["next_probe"]
        next_full = ck["next_full"]
        stream.rng.bit_generator.state = ck["sampler_rng"]
        stream.observed_body_tokens = ck["source_body_tokens"].copy()
        random.setstate(ck["python_rng"])
        np.random.set_state(ck["numpy_rng"])
        torch.set_rng_state(ck["torch_rng"])
        torch.cuda.set_rng_state_all(ck["cuda_rng"])
        config = ck["training_config"]
    else:
        json_write(run / "config.json", config)
    metrics = (run / "train.jsonl").open("a", encoding="utf-8", buffering=1)
    evaluations = (run / "evaluation.jsonl").open("a", encoding="utf-8", buffering=1)
    generations = (run / "generations.jsonl").open("a", encoding="utf-8", buffering=1)
    start = time.perf_counter()
    run_start_body = body
    latest_full = None
    stop_reason = "budget_reached"

    def snapshot(name):
        atomic_save(
            run / "checkpoints" / name,
            state_capture(
                model,
                optimizer,
                stream,
                body,
                valid,
                step,
                best,
                patience,
                config,
                next_probe,
                next_full,
            ),
        )

    def eval_record(kind):
        result = evaluate(model, full=kind == "full")
        row = {"kind": kind, "step": step, "body_tokens": body, "validation": result}
        if kind == "probe":
            row["train_probe"] = evaluate(model, split="train", full=False)
        evaluations.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("EVAL", kind, "tokens", body, "CE", round(result["weighted_ce"], 5), flush=True)
        return result

    if step == 0:
        latest_full = eval_record("full")
        best = latest_full["weighted_ce"]
        snapshot("best.pt")
        snapshot("initial.pt")
        generations.write(
            json.dumps(
                {"body_tokens": body, "step": step, "samples": samples(model)}, ensure_ascii=False
            )
            + "\n"
        )
    try:
        while body < args.tokens:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            update_valid = update_body = 0
            total_loss = 0.0
            slots = 0
            update_start = time.perf_counter()
            while update_valid < 8192:
                rows = [next(stream) for _ in range(args.microbatch)]
                x, y = tensor_batch(rows, device)
                n = int((y != -100).sum())
                nbody = int((y >= 6).sum())
                with amp(device):
                    loss = loss_sum(model(x), y)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                # Stable accumulation scale; after all microbatches correct to their actual valid count.
                (loss / 8192).backward()
                total_loss += float(loss.detach())
                update_valid += n
                update_body += nbody
                slots += y.numel()
            factor = 8192 / update_valid
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(factor)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("Non-finite gradient")
            progress = body + update_body
            if progress < 1_000_000:
                lr = 3e-4 * progress / 1_000_000
            else:
                r = min(1.0, (progress - 1_000_000) / max(1, args.tokens - 1_000_000))
                lr = 3e-5 + 0.5 * (3e-4 - 3e-5) * (1 + math.cos(math.pi * r))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            body += update_body
            valid += update_valid
            step += 1
            elapsed = time.perf_counter() - start
            row = {
                "step": step,
                "body_tokens": body,
                "valid_tokens": valid,
                "ce": total_loss / update_valid,
                "lr": lr,
                "grad_norm_before_clip": float(norm),
                "update_body_tokens": update_body,
                "update_valid_tokens": update_valid,
                "padding_utilization": update_valid / slots,
                "update_seconds": time.perf_counter() - update_start,
                "body_tokens_per_second": update_body / (time.perf_counter() - update_start),
                "elapsed_seconds_this_invocation": elapsed,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            }
            metrics.write(json.dumps(row) + "\n")
            if step % 25 == 0:
                print(
                    "TRAIN",
                    step,
                    "tokens",
                    body,
                    "CE",
                    round(row["ce"], 4),
                    "tok/s",
                    round((body - run_start_body) / elapsed),
                    "MiB",
                    round(row["peak_allocated_mib"]),
                    flush=True,
                )
            did_full = False
            if body >= next_full or body >= args.tokens:
                latest_full = eval_record("full")
                did_full = True
                score = latest_full["weighted_ce"]
                improved = best - score
                patience = 0 if improved >= 0.005 else patience + 1
                if score < best:
                    best = score
                    snapshot("best.pt")
                while next_full <= body:
                    next_full += 5_000_000
                if body >= 5_000_000 and body - update_body < 5_000_000:
                    snapshot("milestone_5m.pt")
                if body >= 15_000_000 and body - update_body < 15_000_000:
                    snapshot("milestone_15m.pt")
                if body >= 30_000_000 and body - update_body < 30_000_000:
                    snapshot("milestone_30m.pt")
            if body >= next_probe or body >= args.tokens:
                eval_record("probe")
                generations.write(
                    json.dumps(
                        {"body_tokens": body, "step": step, "samples": samples(model)},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                while next_probe <= body:
                    next_probe += 1_000_000
                snapshot("latest.pt")
                json_write(
                    run / "status.json",
                    {
                        "status": "running",
                        "step": step,
                        "body_tokens": body,
                        "best_validation_ce": best,
                        "source_body_tokens": stream.observed_body_tokens,
                        "elapsed_seconds_this_invocation": time.perf_counter() - start,
                    },
                )
            if did_full and patience >= 3:
                stop_reason = "validation_patience"
                break
        snapshot("latest.pt")
        # Verify best model loading and full validation once more only if best is not latest.
        best_ck = torch.load(run / "checkpoints/best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(best_ck["model"])
        model.eval()
        best_result = evaluate(model, full=True)
        json_write(
            run / "best_generations.json",
            {"body_tokens": best_ck["body_tokens"], "samples": samples(model)},
        )
        total_source = sum(stream.observed_body_tokens.values())
        result = {
            "status": "complete",
            "stop_reason": stop_reason,
            "body_tokens": body,
            "valid_tokens": valid,
            "steps": step,
            "best_checkpoint_body_tokens": best_ck["body_tokens"],
            "best_validation": best_result,
            "source_body_tokens": stream.observed_body_tokens,
            "source_body_ratios": {
                s: n / total_source for s, n in stream.observed_body_tokens.items()
            },
            "elapsed_seconds_this_invocation": time.perf_counter() - start,
            "parameter_count": config["parameter_count"],
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "finished_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        }
        json_write(run / "result.json", result)
        json_write(run / "status.json", result)
        print("COMPLETE", json.dumps(result, ensure_ascii=False), flush=True)
    except BaseException as exc:
        json_write(run / "error.json", {"error": repr(exc), "body_tokens": body, "step": step})
        raise
    finally:
        metrics.close()
        evaluations.close()
        generations.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="pretrain_v1")
    parser.add_argument("--tokens", type=int, default=30_000_000)
    parser.add_argument("--microbatch", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    train(parser.parse_args())
