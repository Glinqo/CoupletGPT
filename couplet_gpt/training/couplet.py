"""Train expanded lexical final version and a same-budget old-lexicon control."""

import argparse, gc, random, time
from couplet_gpt.training.couplet_common import *
from couplet_gpt.training.initial_couplet import evaluate, structure
from couplet_gpt.modeling.gpt import loss_sum


def preflight(m, branch, tr):
    m.eval()
    chosen = [next(r for r in tr if len(r["upper"]) == n) for n in (4, 6, 7, 9, 18, 49)]
    x, y = collate(chosen)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        z = m(x)
    assert torch.isfinite(z).all()
    x, _ = collate([chosen[2]])
    with torch.no_grad():
        a = m(x)
        b = m(x, explicit=True)
        cut = int((x == 5).nonzero()[0, 1]) + 1
        changed = x.clone()
        changed[:, cut:] = 123
        c = m(changed)
        explicit = float((a - b).abs().max())
        causal = float((a[:, :cut] - c[:, :cut]).abs().max())
    assert explicit < 1e-4 and causal == 0 and m.head.weight is m.tok_emb.weight
    assert m.context_provider.encoder is None, (
        "All training uppers must reuse source-only frozen context cache"
    )
    dump(
        OUT / branch / "preflight.json",
        {
            "status": "pass",
            "causal_difference": causal,
            "explicit_difference": explicit,
            "lengths": [len(r["upper"]) for r in chosen],
            "parameters": sum(p.numel() for p in m.parameters()),
            "lexicon_words": len(m.entries),
            "max_word_length": m.matcher.max_word_length,
            "frozen_encoder_not_in_optimizer": True,
        },
    )
    print("PREFLIGHT", branch, len(m.entries), flush=True)


def run(branch):
    folder = OUT / branch
    folder.mkdir(exist_ok=True)
    if (folder / "result.json").exists():
        return
    p = read(OUT / "protocol.json")
    seed_all(p["seed"])
    m, ck = initialize(branch)
    tr = list(rows(DATA / "train.jsonl"))
    assert len(tr) == p["examples_per_epoch"]
    val = list(rows(LOCK / "validation_locked.jsonl"))
    preflight(m, branch, tr)
    optimizer = torch.optim.AdamW(
        m.parameters(), lr=p["lr"], weight_decay=p["weight_decay"], betas=tuple(p["betas"])
    )
    resume = folder / "resume.pt"
    start = time.perf_counter()
    elapsed = 0.0
    steps = 0
    best_epoch = 0
    first_epoch = 1
    first_offset = 0
    num = den = 0
    orders = []
    if resume.exists():
        r = torch.load(resume, map_location="cpu", weights_only=False)
        m.load_state_dict(r["model"])
        optimizer.load_state_dict(r["optimizer"])
        history = r["history"]
        best = r["best"]
        best_epoch = r["best_epoch"]
        steps = r["steps"]
        first_epoch = r["epoch"]
        first_offset = r["offset"]
        num = r["num"]
        den = r["den"]
        elapsed = r["seconds"]
        orders = r["orders"]
        torch.set_rng_state(r["torch_rng"])
        torch.cuda.set_rng_state(r["cuda_rng"])
        np.random.set_state(r["numpy_rng"])
        random.setstate(r["python_rng"])
        print("RESUME", branch, steps, first_epoch, first_offset, flush=True)
    else:
        assert not (folder / "training.jsonl").exists(), (
            "Partial run without resume: inspect manually"
        )
        initial = evaluate(m, val)
        history = [{"epoch": 0, "validation": initial}]
        best = initial["char_ce"]
        atomic_save(folder / "best.pt", checkpoint(m, 0, initial, branch))
    weights = np.array([1.0 if len(r["upper"]) in (5, 7) else 4.0 for r in tr])
    weights /= weights.sum()

    def save_resume(ep, offset):
        atomic_save(
            resume,
            {
                "model": m.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": ep,
                "offset": offset,
                "history": history,
                "best": best,
                "best_epoch": best_epoch,
                "steps": steps,
                "num": num,
                "den": den,
                "orders": orders,
                "seconds": elapsed + time.perf_counter() - start,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(),
                "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
            },
        )

    with (folder / "training.jsonl").open("a", encoding="utf8", buffering=1) as log:
        for ep in range(first_epoch, p["epochs"] + 1):
            order = np.random.default_rng(7200 + ep).choice(
                len(tr), len(tr), replace=True, p=weights
            )
            if len(orders) < ep:
                orders.append(stable(order.tobytes().hex()))
            for j in range(0, len(order), 4096):
                order[j : j + 4096] = sorted(order[j : j + 4096], key=lambda k: len(tr[k]["upper"]))
            offset = first_offset if ep == first_epoch else 0
            if not offset:
                num = den = 0
            m.train()
            for i in range(offset, len(order), p["batch"]):
                b = [tr[j] for j in order[i : i + p["batch"]]]
                x, y = collate(b)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z = m(x)
                    ce = loss_sum(z, y)
                    loss = ce / (y != -100).sum() + 0.5 * structure(z, x, y)
                assert torch.isfinite(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                optimizer.step()
                steps += 1
                num += float(ce.detach())
                den += int((y != -100).sum())
                if steps % 250 == 0:
                    v = {
                        "branch": branch,
                        "epoch": ep,
                        "step": steps,
                        "train_ce": num / den,
                        "seconds": elapsed + time.perf_counter() - start,
                    }
                    log.write(json.dumps(v) + "\n")
                    print(json.dumps(v), flush=True)
                    dump(folder / "progress.json", v)
                if steps % 1000 == 0:
                    save_resume(ep, i + p["batch"])
            score = evaluate(m, val)
            history.append(
                {"epoch": ep, "validation": score, "train_ce": num / den, "supervised_tokens": den}
            )
            if score["char_ce"] < best:
                best = score["char_ce"]
                best_epoch = ep
                atomic_save(folder / "best.pt", checkpoint(m, ep, score, branch))
            atomic_save(folder / "latest.pt", checkpoint(m, ep, score, branch))
            dump(folder / "history.json", history)
            num = den = 0
            save_resume(ep + 1, 0)
            print("EPOCH", branch, ep, score["char_ce"], flush=True)
    dump(
        folder / "result.json",
        {
            "status": "complete",
            "epochs": p["epochs"],
            "steps": steps,
            "examples_seen": len(tr) * p["epochs"],
            "best_epoch": best_epoch,
            "best_char_ce": best,
            "history": history,
            "sample_order_hashes": orders,
            "seconds": elapsed + time.perf_counter() - start,
            "parameters": sum(p.numel() for p in m.parameters()),
            "new_word_residual_nonzero_rows": int(
                (m.word_residual.weight[len(ck["lexicon_entries"]) + 1 :].abs().sum(-1) > 0).sum()
            )
            if branch != "expanded"
            else int((m.word_residual.weight[p["old_words"] + 1 :].abs().sum(-1) > 0).sum()),
        },
    )
    m._dynamic_match.cache_clear()
    del m, optimizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.set_num_threads(4)
    prepare()
    for branch in ("expanded", "old_lexicon_control"):
        run(branch)
