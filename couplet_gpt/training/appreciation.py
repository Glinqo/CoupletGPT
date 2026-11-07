"""One exhaustive large-corpus SFT pass, no post-hoc couplet adaptation."""

import argparse, shutil
from couplet_gpt.training.appreciation_runtime import *


def preflight(data):
    from couplet_gpt.modeling.cached_decoding import cached_prefix, cached_step

    m = initialize().eval()
    ck = torch.load(BASE, map_location="cpu", weights_only=True)
    base = GPT(GPTConfig(**ck["model_config"])).cuda().eval()
    base.load_state_dict(ck["model"])
    x = torch.tensor([[2] + list(range(10, 60))], device="cuda")
    with torch.inference_mode():
        inherited = float((m(x)[..., : base.config.vocab_size] - base(x)).abs().max())
        logits = m(x)
        equiv = float((logits - m.head(hidden(m, x))).abs().max())
        alter = x.clone()
        alter[:, 30:] = 77
        causal = float((m(x)[:, :30] - m(alter)[:, :30]).abs().max())
        lens = torch.tensor([x.shape[1]], device="cuda")
        a, cache = cached_prefix(m, x, lens, 100)
        cache0 = float((a - logits[:, -1]).abs().max())
        t = a.argmax(-1)
        b = cached_step(m, t, lens, lens + 1, cache)
        cache1 = float((b - m(torch.cat([x, t[:, None]], 1))[:, -1]).abs().max())
        xx, yy = data.collate(data.train_indices[:2])
        mask = yy != -100
        full = F.cross_entropy(m(xx)[mask].float(), yy[mask], reduction="sum")
        masked = nll(m, xx, yy)
        lossdiff = float((full - masked).abs()) / int(mask.sum())
    assert max(inherited, equiv, causal, cache0, cache1, lossdiff) < 1e-4
    del base, ck, cache
    torch.cuda.empty_cache()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True)
    bench = []
    for length in [512, 1024, 1536]:
        candidates = data.train_indices[data.lengths[data.train_indices] <= length]
        ii = candidates[np.argsort(data.lengths[candidates])[-max(1, min(16, 6144 // length)) :]]
        x, y = data.collate(ii)
        m.train()
        opt.zero_grad(set_to_none=True)
        began = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = nll(m, x, y) / (y != -100).sum()
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        bench.append(
            {
                "microbatch": len(ii),
                "length": int(x.shape[1]),
                "seconds": time.perf_counter() - began,
                "peak_mib": torch.cuda.max_memory_allocated() / 2**20,
            }
        )
    result = {
        "status": "pass",
        "old_vocab_logits_difference": inherited,
        "hidden_head_equivalence": equiv,
        "future_leak_difference": causal,
        "cache_prefill_difference": cache0,
        "cache_step_difference": cache1,
        "response_only_loss_difference": lossdiff,
        "tied_embeddings": m.head.weight is m.tok_emb.weight,
        "parameters": sum(p.numel() for p in m.parameters()),
        "benchmarks": bench,
        "benchmark_weights_discarded": True,
        "program_tone_prefix_not_a_neural_target": True,
        "base_sha256": sha(BASE),
    }
    dump(OUT / "architecture_checks.json", result)
    print("PREFLIGHT", json.dumps(result), flush=True)
    del m, opt, x, y, loss
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true")
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.50)
    ready = json.loads((OUT / "DATA_READY.json").read_text(encoding="utf8"))
    assert ready["status"] == "ready_for_direct_full_poetry_sft"
    if (OUT / "TRAINING_COMPLETE.json").exists():
        print("ALREADY_COMPLETE")
        return
    data = Corpus()
    locked = json.loads((OUT / "evaluation_locked.json").read_text(encoding="utf8"))
    if not (OUT / "architecture_checks.json").exists():
        preflight(data)
    if args.check_only:
        return
    if not (OUT / "config.json").exists():
        snapshot = OUT / "source_snapshot"
        snapshot.mkdir(exist_ok=True)
        names = [
            "model.py",
            "appreciation_sft_v3_common.py",
            "model_comment_sft_v2.py",
            "direct_poetry_sft_v4_common.py",
            "survey_direct_poetry_sft_v4.py",
            "prepare_direct_poetry_sft_v4.py",
            "train_direct_poetry_sft_v4.py",
            "commentary_prosody_v3.py",
        ]
        for name in names:
            shutil.copy2(ROOT / "scripts" / name, snapshot / name)
        dump(
            OUT / "config.json",
            {
                "status": "locked_before_training",
                "epochs": 1,
                "training_examples": ready["split_counts"]["train"],
                "supervised_tokens_per_epoch": ready["target_tokens_per_pass_by_split"]["train"],
                "source": "full poetry appreciation only",
                "curated_couplet_examples": 0,
                "base_sha256": sha(BASE),
                "data_ready_sha256": sha(OUT / "DATA_READY.json"),
                "effective_batch": 128,
                "adaptive_microbatch_token_budget": 6144,
                "max_microbatch": 16,
                "lr": 1.5e-4,
                "minimum_lr": 1.5e-5,
                "warmup_fraction": 0.03,
                "AdamW_betas": [0.9, 0.95],
                "weight_decay": 0.01,
                "clip": 1.0,
                "seed": 20261106,
                "torch": torch.__version__,
                "source_sha256": {n: sha(snapshot / n) for n in names},
            },
        )
    m = initialize()
    opt = torch.optim.AdamW(
        m.parameters(), lr=1.5e-4, betas=(0.9, 0.95), weight_decay=0.01, fused=True
    )
    ix = order(data)
    batch = 128
    steps = math.ceil(len(ix) / batch)
    step = cursor = seen = tokens = 0
    best = 1e9
    history = []
    prior = 0.0
    latest = OUT / "latest.pt"
    readyhash = sha(OUT / "DATA_READY.json")
    began = time.perf_counter()
    if latest.exists():
        assert args.resume, "Use --resume for existing training"
        ck = torch.load(latest, map_location="cpu", weights_only=True)
        assert ck["data_ready_sha256"] == readyhash
        m.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step, cursor, seen, tokens, best, history, prior = [
            ck[k]
            for k in [
                "step",
                "cursor",
                "examples_seen",
                "supervised_tokens",
                "best_validation_ce",
                "history",
                "seconds",
            ]
        ]
        torch.set_rng_state(ck["torch_rng"])
        torch.cuda.set_rng_state_all(ck["cuda_rng"])
        # A crash can leave log rows newer than the recoverable checkpoint; retain them separately.
        lp = OUT / "training.jsonl"
        if lp.exists():
            saved = list(rows(lp))
            tail = [r for r in saved if r["step"] > step]
            if tail:
                jsonl(OUT / ("interrupted_unsaved_after_step" + str(step) + ".jsonl"), tail)
                jsonl(lp, [r for r in saved if r["step"] <= step])

    def checkpoint(path, with_opt=True):
        obj = {
            "model": m.state_dict(),
            "model_config": asdict(m.config),
            "step": step,
            "cursor": cursor,
            "examples_seen": seen,
            "supervised_tokens": tokens,
            "best_validation_ce": best,
            "history": history,
            "seconds": prior + time.perf_counter() - began,
            "data_ready_sha256": readyhash,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "training": "direct_full_poetry_no_couplet_stage",
        }
        if with_opt:
            obj["optimizer"] = opt.state_dict()
        save(path, obj)

    if not latest.exists():
        baseline = evaluate(m, data, locked["validation"])
        best = baseline["response_ce"]
        history.append({"step": 0, "validation": baseline})
        dump(OUT / "baseline_test.json", evaluate(m, data, locked["test"]))
        rr = list(rows(OUT / "test_locked.jsonl"))[:64]
        jsonl(OUT / "generation_test_locked.jsonl", rr)
        pred = decode(m, rr, data.tok)
        jsonl(OUT / "baseline_generations.jsonl", pred)
        dump(OUT / "baseline_generation_metrics.json", metrics(pred))
        checkpoint(OUT / "best.pt", False)
        checkpoint(latest)
        print("INITIAL", baseline, flush=True)
    log = (OUT / "training.jsonl").open("a", encoding="utf8", buffering=1)
    interval = max(1, steps // 4)
    try:
        m.train()
        for start in range(cursor, len(ix), batch):
            ii = ix[start : start + batch]
            denom = int(data.response_lengths[ii].sum())
            opt.zero_grad(set_to_none=True)
            loss_total = 0.0
            ts = time.perf_counter()
            warmup = max(20, int(steps * 0.03))
            lr = (
                1.5e-4 * (step + 1) / warmup
                if step < warmup
                else 1.5e-5
                + 0.5
                * (1.5e-4 - 1.5e-5)
                * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))
            )
            for g in opt.param_groups:
                g["lr"] = lr
            micro = max(1, min(16, 6144 // int(data.lengths[ii].max())))
            for j in range(0, len(ii), micro):
                x, y = data.collate(ii[j : j + micro])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = nll(m, x, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss")
                (loss / denom).backward()
                loss_total += float(loss.detach())
            norm = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite gradient")
            opt.step()
            step += 1
            seen += len(ii)
            tokens += denom
            cursor = start + len(ii)
            log.write(
                json.dumps(
                    {
                        "step": step,
                        "examples_seen": seen,
                        "supervised_tokens": tokens,
                        "response_ce": loss_total / denom,
                        "lr": lr,
                        "microbatch": micro,
                        "length": int(data.lengths[ii].max()),
                        "update_seconds": time.perf_counter() - ts,
                        "gradient_norm": float(norm),
                        "seconds": prior + time.perf_counter() - began,
                    }
                )
                + "\n"
            )
            if step % 50 == 0:
                print(
                    "DIRECT_TRAIN",
                    step,
                    "/",
                    steps,
                    "seen",
                    seen,
                    "CE",
                    round(loss_total / denom, 4),
                    "sec",
                    round(time.perf_counter() - ts, 3),
                    flush=True,
                )
            if step % 200 == 0:
                checkpoint(latest)
            if step % interval == 0 or cursor == len(ix):
                v = evaluate(m, data, locked["validation"])
                history.append({"step": step, "examples_seen": seen, "validation": v})
                if v["response_ce"] < best:
                    best = v["response_ce"]
                    checkpoint(OUT / "best.pt", False)
                dump(OUT / "history.json", history)
                print("DIRECT_VALIDATION", step, v, flush=True)
                vr = list(rows(OUT / "validation_locked.jsonl"))[:8]
                jsonl(OUT / f"validation_step{step}.jsonl", decode(m, vr, data.tok))
                checkpoint(latest)
                m.train()
        dump(
            OUT / "training_result.json",
            {
                "status": "complete",
                "epochs": 1,
                "steps": step,
                "examples_seen": seen,
                "supervised_tokens": tokens,
                "best_validation_ce": best,
                "seconds": prior + time.perf_counter() - began,
                "history": history,
                "parameters": sum(p.numel() for p in m.parameters()),
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "effective_batch": batch,
                "adaptive_microbatch": "min(16, floor(6144 / batch maximum length))",
                "max_lr": 1.5e-4,
                "curated_couplet_training_examples": 0,
            },
        )
    except BaseException as exc:
        dump(OUT / "error.json", {"error": repr(exc), "step": step, "cursor": cursor})
        raise
    finally:
        log.close()
    del m, opt
    torch.cuda.empty_cache()
    ck = torch.load(OUT / "best.pt", map_location="cpu", weights_only=True)
    m = initialize()
    m.load_state_dict(ck["model"])
    m.eval()
    test = evaluate(m, data, locked["test"])
    rr = list(rows(OUT / "generation_test_locked.jsonl"))
    pred = decode(m, rr, data.tok)
    jsonl(OUT / "test_generations.jsonl", pred)
    evaluation = {
        "status": "complete",
        "baseline_test": json.loads((OUT / "baseline_test.json").read_text(encoding="utf8")),
        "finetuned_test": test,
        "baseline_generation": json.loads(
            (OUT / "baseline_generation_metrics.json").read_text(encoding="utf8")
        ),
        "finetuned_generation": metrics(pred),
        "selected_step": ck["step"],
        "selection": "lowest full-poetry validation response CE",
        "no_couplet_adaptation": True,
        "expert_quality_not_certified": True,
    }
    dump(OUT / "evaluation.json", evaluation)
    artifact = OUT / "artifacts"
    artifact.mkdir(exist_ok=True)
    save(
        artifact / "best_model.pt",
        {
            "model": ck["model"],
            "model_config": ck["model_config"],
            "architecture": "self_built_character_GPT_direct_poetry_SFT",
            "data_ready_sha256": readyhash,
            "program_prosody_prefix_external": True,
        },
    )
    (artifact / "vocab.json").write_bytes((OUT / "vocab.json").read_bytes())
    dump(artifact / "model_config.json", ck["model_config"])
    dump(
        OUT / "TRAINING_COMPLETE.json",
        {
            "status": "complete",
            "date": "2026-11-06",
            "direct_full_poetry_only": True,
            "curated_couplet_training_examples": 0,
            "model_sha256": sha(artifact / "best_model.pt"),
            "evaluation_sha256": sha(OUT / "evaluation.json"),
            "program_tone_prefix_not_model_output": True,
        },
    )
    print("DIRECT_POETRY_SFT_COMPLETE", evaluation, flush=True)


if __name__ == "__main__":
    main()
