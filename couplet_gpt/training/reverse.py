"""Train symmetric reverse adapter and matched alignment-only control, with resumable logs."""

import random, sys
from couplet_gpt.training.reverse_common import *
from couplet_gpt.training.initial_couplet import evaluate
from couplet_gpt.modeling.gpt import loss_sum


def preflight(m, old, branch, tr):
    sample = [next(r for r in tr if len(r["lower"]) == n) for n in (4, 6, 7, 9, 18)]
    x, y = collate(sample, reverse=True)
    m.eval()
    base = load_model_v2(old)
    with torch.no_grad():
        initial = float((m(x) - base(x)).abs().max())
    assert initial < 1e-4, initial
    x, _ = collate(sample[2:3], reverse=True)
    cut = int((x == 5).nonzero()[0, 1]) + 1
    changed = x.clone()
    changed[:, cut:] = 123
    with torch.no_grad():
        z = m(x)
        causal = float((z[:, :cut] - m(changed)[:, :cut]).abs().max())
    assert causal == 0 and m.head.weight is m.tok_emb.weight
    if branch == "reverse_adapter":
        assert m.context_provider.encoder is None
        # Input-side metadata must be original lower, independent of target upper.
        altered = [{**sample[2], "upper": "春" * 7}]
        a, _ = collate(altered, reverse=True)
        assert torch.equal(m.metadata_for(x), m.metadata_for(a))
    dump(
        OUT / branch / "preflight.json",
        {
            "status": "pass",
            "initial_logits_equal_old_reverse_maxdiff": initial,
            "target_future_difference": causal,
            "input_side": "original lower only",
            "tied_character_weights": True,
            "parameters": sum(p.numel() for p in m.parameters()),
        },
    )
    del base
    torch.cuda.empty_cache()
    print("PREFLIGHT", branch, initial, flush=True)


def run(branch):
    folder = OUT / branch
    folder.mkdir(exist_ok=True)
    if (folder / "result.json").exists():
        return
    p = read(OUT / "protocol.json")
    seed_all(p["seed"])
    m, old = initialize_reverse(branch)
    tr = list(rows(OUT / "reverse_train.jsonl"))
    val = list(rows(LOCK / "validation_locked.jsonl"))
    preflight(m, old, branch, tr)
    opt = torch.optim.AdamW(
        m.parameters(), lr=p["lr"], weight_decay=p["weight_decay"], betas=tuple(p["betas"])
    )
    start = time.perf_counter()
    elapsed = 0.0
    steps = 0
    best_epoch = 0
    first_epoch = 1
    first_offset = 0
    num = den = 0
    orders = []
    resume = folder / "resume.pt"
    if resume.exists():
        r = torch.load(resume, map_location="cpu", weights_only=False)
        m.load_state_dict(r["model"])
        opt.load_state_dict(r["optimizer"])
        history = r["history"]
        best = r["best"]
        best_epoch = r["best_epoch"]
        steps = r["steps"]
        first_epoch = r["epoch"]
        first_offset = r["offset"]
        num = r["num"]
        den = r["den"]
        orders = r["orders"]
        elapsed = r["seconds"]
        torch.set_rng_state(r["torch_rng"])
        torch.cuda.set_rng_state(r["cuda_rng"])
        np.random.set_state(r["numpy_rng"])
        random.setstate(r["python_rng"])
    else:
        assert not (folder / "training.jsonl").exists(), (
            "Inspect interrupted run without checkpoint"
        )
        initial = evaluate(m, val, True)
        history = [{"epoch": 0, "validation": initial}]
        best = initial["char_ce"]
        save_model(m, folder / "best.pt", branch, 0, initial)
    weights = np.array([1.0 if len(r["upper"]) in (5, 7) else 4.0 for r in tr])
    weights /= weights.sum()

    def resume_save(ep, offset):
        atomic_save(
            resume,
            {
                "model": m.state_dict(),
                "optimizer": opt.state_dict(),
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
                order[j : j + 4096] = sorted(order[j : j + 4096], key=lambda k: len(tr[k]["lower"]))
            offset = first_offset if ep == first_epoch else 0
            if not offset:
                num = den = 0
            m.train()
            for i in range(offset, len(order), p["batch"]):
                b = [tr[j] for j in order[i : i + p["batch"]]]
                x, y = collate(b, reverse=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z = m(x)
                    ce = loss_sum(z, y)
                    loss = ce / (y != -100).sum()
                assert torch.isfinite(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), p["clip"])
                opt.step()
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
                    dump(folder / "progress.json", v)
                    print(json.dumps(v), flush=True)
                if steps % 1000 == 0:
                    resume_save(ep, i + p["batch"])
            score = evaluate(m, val, True)
            history.append(
                {"epoch": ep, "validation": score, "train_ce": num / den, "supervised_tokens": den}
            )
            if score["char_ce"] < best:
                best = score["char_ce"]
                best_epoch = ep
                save_model(m, folder / "best.pt", branch, ep, score)
            save_model(m, folder / "latest.pt", branch, ep, score)
            dump(folder / "history.json", history)
            num = den = 0
            resume_save(ep + 1, 0)
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
            "parameters": sum(p.numel() for p in m.parameters()),
            "sample_order_hashes": orders,
            "history": history,
            "seconds": elapsed + time.perf_counter() - start,
        },
    )
    if hasattr(m, "_dynamic_match"):
        m._dynamic_match.cache_clear()
    del m, opt
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf8")
    torch.set_num_threads(4)
    freeze()
    build_reverse_contexts()
    for branch in BRANCHES:
        run(branch)
