"""Train the chosen ChineseBERT baseline and reverse ranker on all available lengths."""

import math, time
import numpy as np
import torch
from torch.nn import functional as F
from dataclasses import asdict
from couplet_gpt.data.couplet_splits import (
    OUT,
    BASE,
    prepare,
    collate,
    read,
    rows,
    dump,
    jsonl,
    sha,
)
from couplet_gpt.modeling.couplet import initialize, DynamicChineseBERT
from couplet_gpt.modeling.lexical_adapter import ContextStore
from couplet_gpt.data.datasets import CharacterTokenizer, ROOT
from couplet_gpt.modeling.gpt import GPT, GPTConfig, loss_sum
from couplet_gpt.modeling.alignment import load_model_v2
from couplet_gpt.training.pretrain import seed_all, atomic_save


def contexts():
    if (OUT / "context_ready.json").exists():
        return
    tok = CharacterTokenizer()
    texts = list(
        dict.fromkeys(
            r["upper"] for s in ["train", "validation", "test"] for r in rows(OUT / f"{s}.jsonl")
        )
    )
    keys = [tuple(tok.encode(t)) for t in texts]
    total = sum(map(len, keys))
    v = np.lib.format.open_memmap(
        OUT / "contexts.npy", mode="w+", dtype=np.float16, shape=(total, 768)
    )
    mask = np.lib.format.open_memmap(
        OUT / "context_valid.npy", mode="w+", dtype=np.uint8, shape=(total,)
    )
    old = ContextStore()
    online = DynamicChineseBERT()
    online.index = {}
    index = []
    offset = 0
    start = time.perf_counter()
    for i in range(0, len(keys), 256):
        kk = keys[i : i + 256]
        missing = [k for k in kk if k not in old.index]
        if missing:
            online.add(missing)
        for k in kk:
            n = len(k)
            if k in old.index:
                ix = old.index[k]
                a, b = old.values[ix, :n], old.valid[ix, :n]
            else:
                a, b = online.online[k]
            v[offset : offset + n] = a
            mask[offset : offset + n] = b
            index.append({"ids": list(k), "offset": offset, "length": n})
            offset += n
        online.online.clear()
        if i % 25600 == 0:
            print(
                "CACHE", i, len(keys), "seconds", round(time.perf_counter() - start, 1), flush=True
            )
    v.flush()
    mask.flush()
    jsonl(OUT / "context_index.jsonl", index)
    dump(
        OUT / "context_ready.json",
        {
            "status": "complete",
            "sources": len(keys),
            "characters": total,
            "source_only": True,
            "max_length": max(map(len, keys)),
        },
    )


def structure(z, x, y):
    rr, qq = (y >= 6).nonzero(as_tuple=True)
    down = (x == 5).long().argmax(1)
    j = qq - down[rr]
    gold = y[rr, qq]
    upper = x[rr, j + 2]
    a = z[rr, qq].float()
    lse = a.logsumexp(-1)
    copy = (a.gather(1, upper[:, None]).squeeze(1) - lse).exp()
    same = gold != upper
    total = (-torch.log1p(-copy.clamp(max=1 - 1e-6)) * same).sum()
    count = same.sum()
    # Preserve equality patterns for any source length, not only seven positions.
    for k in range(int(j.max())):
        valid = j > k
        u = x[rr, k + 2]
        past = x[rr, (down[rr] + 1 + k).clamp(max=x.shape[1] - 1)]
        eq = valid & (u == upper) & (gold == past)
        total += (-(a.gather(1, past[:, None]).squeeze(1) - lse) * eq).sum()
        count += eq.sum()
    return total / count.clamp_min(1)


@torch.inference_mode()
def evaluate(m, rr, reverse=False):
    m.eval()
    loss = chars = 0.0
    by = {}
    for i in range(0, len(rr), 128):
        b = rr[i : i + 128]
        x, y = collate(b, reverse)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = m(x)
        ce = F.cross_entropy(z.transpose(1, 2).float(), y, ignore_index=-100, reduction="none")
        valid = (y != -100) & (y != 3)
        for j, r in enumerate(b):
            n = len(r["upper"])
            a = float(ce[j][valid[j]].sum())
            c = int(valid[j].sum())
            loss += a
            chars += c
            t = by.setdefault(str(n), [0.0, 0])
            t[0] += a
            t[1] += c
    return {
        "char_ce": loss / chars,
        "rows": len(rr),
        "by_length": {k: {"char_ce": v[0] / v[1], "characters": v[1]} for k, v in by.items()},
    }


def run(reverse=False):
    name = "reverse" if reverse else "forward"
    folder = OUT / name
    folder.mkdir(exist_ok=True)
    if (folder / "result.json").exists():
        return
    assert not (folder / "training.jsonl").exists(), "Inspect partial training before restarting"
    seed_all(2026)
    if reverse:
        ck = torch.load(
            ROOT / "runs/optimization_v4/artifacts/reverse_model.pt",
            map_location="cpu",
            weights_only=True,
        )
        m = load_model_v2(ck)
    else:
        m, ck = initialize()
    tr = list(rows(OUT / "train.jsonl"))
    va = list(rows(OUT / "validation.jsonl"))
    # Validation stratified by source length; all non-five/seven validation rows are retained.
    val = [
        r for r in va if len(r["upper"]) not in (5, 7) or int(sha_text(r["id"])[:8], 16) % 10 == 0
    ]
    opt = torch.optim.AdamW(m.parameters(), lr=2e-5, weight_decay=0.01, betas=(0.9, 0.95))
    initial = evaluate(m, val, reverse)
    best = initial["char_ce"]
    history = [{"epoch": 0, "validation": initial}]

    def save(path, ep, score):
        atomic_save(
            path,
            {
                "architecture": "variable_lexical_gpt_v7"
                if not reverse
                else ck.get("architecture", "gpt"),
                "model": m.state_dict(),
                "model_config": asdict(m.config),
                "lexicon_entries": getattr(m, "entries", None),
                "epoch": ep,
                "validation": score,
                "source_only": True,
                "base_checkpoint": str(
                    BASE
                    if not reverse
                    else ROOT / "runs/optimization_v4/artifacts/reverse_model.pt"
                ),
                "base_sha256": sha(
                    BASE
                    if not reverse
                    else ROOT / "runs/optimization_v4/artifacts/reverse_model.pt"
                ),
                "max_single_line_length": 62,
            },
        )

    save(folder / "best.pt", 0, initial)
    best_epoch = 0
    start = time.perf_counter()
    steps = 0
    orders = []
    weights = np.array([1.0 if len(r["upper"]) in (5, 7) else 4.0 for r in tr])
    weights /= weights.sum()
    log = (folder / "training.jsonl").open("w", encoding="utf8", buffering=1)
    for ep in range(1, 4):
        order = np.random.default_rng(7200 + ep).choice(len(tr), len(tr), replace=True, p=weights)
        orders.append(__import__("hashlib").sha256(order.tobytes()).hexdigest())
        # Local length sorting reduces padding without removing short/long examples.
        for i in range(0, len(order), 4096):
            order[i : i + 4096] = sorted(order[i : i + 4096], key=lambda j: len(tr[j]["upper"]))
        m.train()
        num = den = 0
        for i in range(0, len(order), 128):
            b = [tr[j] for j in order[i : i + 128]]
            x, y = collate(b, reverse)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = m(x)
                cs = loss_sum(z, y)
                loss = cs / (y != -100).sum()
                if not reverse:
                    loss = loss + 0.5 * structure(z, x, y)
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            steps += 1
            num += float(cs.detach())
            den += int((y != -100).sum())
            if steps % 200 == 0:
                v = {
                    "epoch": ep,
                    "step": steps,
                    "train_ce": num / den,
                    "seconds": time.perf_counter() - start,
                }
                log.write(__import__("json").dumps(v) + "\n")
                print(name, v, flush=True)
        score = evaluate(m, val, reverse)
        history.append({"epoch": ep, "validation": score, "train_ce": num / den})
        if score["char_ce"] < best:
            best = score["char_ce"]
            best_epoch = ep
            save(folder / "best.pt", ep, score)
        save(folder / "latest.pt", ep, score)
        dump(folder / "history.json", history)
        print("EPOCH", name, ep, score, flush=True)
    log.close()
    dump(
        folder / "result.json",
        {
            "status": "complete",
            "epochs": 3,
            "training_rows": len(tr),
            "examples_seen": len(tr) * 3,
            "steps": steps,
            "initial": initial,
            "best_char_ce": best,
            "best_epoch": best_epoch,
            "history": history,
            "length_extension_sampling_multiplier": 4,
            "order_hashes": orders,
            "seconds": time.perf_counter() - start,
            "loss": "CE + .5 variable-length structure (forward); CE (reverse)",
            "BMES_head_retained_not_supervised_this_stage": True,
        },
    )
    if hasattr(m, "_dynamic_match"):
        m._dynamic_match.cache_clear()
    del m, opt
    torch.cuda.empty_cache()


def sha_text(s):
    return __import__("hashlib").sha256(s.encode()).hexdigest()


if __name__ == "__main__":
    torch.set_num_threads(4)
    prepare()
    contexts()
    run()
    run(True)
