import collections, hashlib, json
from pathlib import Path
import numpy as np, torch
from couplet_gpt.data.couplet_splits import ROOT, read, rows, dump, jsonl, sha, collate
from couplet_gpt.modeling.couplet import initialize, DynamicChineseBERT
from couplet_gpt.data.datasets import CharacterTokenizer
from couplet_gpt.data.build_couplets import norm

DATA = ROOT / "data/processed/couplet_v8/single_clause_v7"
OUT = ROOT / "runs/data_continuation_v8"
PREVIOUS = ROOT / "runs/final_couplet_v7"


def stable(s):
    return hashlib.sha256(s.encode()).hexdigest()


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        return
    assert read(DATA.parent / "verification.json")["status"] == "pass"
    tr = list(rows(DATA / "train.jsonl"))
    va = list(rows(DATA / "validation.jsonl"))
    te = list(rows(DATA / "test.jsonl"))
    held = {r["upper"] for r in va + te}
    old = list(rows(PREVIOUS / "train.jsonl"))
    control = [r for r in old if r["upper"] not in held and norm(r["upper"]) not in held]
    oldseen = {u for r in old for u in (r["upper"], norm(r["upper"]))}
    gg = {}
    for r in te:
        if r["upper"] in oldseen:
            continue
        g = gg.setdefault(r["upper"], {"upper": r["upper"], "references": []})
        if r["lower"] not in g["references"]:
            g["references"].append(r["lower"])
    # Fixed length-stratified comparison, locked before seeing updated models.
    groups = []
    for n in sorted({len(u) for u in gg}):
        group = sorted(
            [g for g in gg.values() if len(g["upper"]) == n],
            key=lambda g: stable("evaluation:" + g["upper"]),
        )
        groups.extend(group[: (600 if n in (4, 5, 7) else 250 if n == 6 else 100)])
    val = []
    for n in sorted({len(r["upper"]) for r in va}):
        val.extend(
            sorted(
                [r for r in va if len(r["upper"]) == n],
                key=lambda r: stable("validation:" + r["id"]),
            )[:500]
        )
    jsonl(OUT / "validation_locked.jsonl", val)
    jsonl(OUT / "test_locked.jsonl", groups)
    jsonl(OUT / "control_train.jsonl", control)
    dump(
        OUT / "protocol.json",
        {
            "status": "locked_before_training",
            "initial_forward": str(PREVIOUS / "forward/best.pt"),
            "initial_reverse": str(PREVIOUS / "reverse/best.pt"),
            "initial_forward_sha256": sha(PREVIOUS / "forward/best.pt"),
            "initial_reverse_sha256": sha(PREVIOUS / "reverse/best.pt"),
            "vocab_sha256": sha(ROOT / "tokenizer/vocab.json"),
            "new_training_pairs": len(tr),
            "control_training_pairs": len(control),
            "control_removed_new_heldout_uppers": len(old) - len(control),
            "epochs": 3,
            "examples_per_epoch": len(tr),
            "batch": 128,
            "steps_per_branch": 3 * ((len(tr) + 127) // 128),
            "validation_pairs": len(val),
            "test_upper_groups": len(groups),
            "test_length_counts": dict(collections.Counter(len(g["upper"]) for g in groups)),
            "test_excludes_previous_v7_train_uppers": True,
            "historical_pretraining_exposure_not_ruled_out": True,
            "seed": 2026,
            "sampling": "replacement; non5/7 weight4; local4096 length sort",
            "lr": 2e-5,
            "optimizer": "AdamW betas(.9,.95), weight_decay .01, gradient_clip1",
            "forward_loss": "CE+.5 dynamic structure",
            "reverse_loss": "CE",
            "generation": "raw greedy; v4 beam16 + deterministic8 samples + forward/reverse score sum",
            "same_updates_and_examples_not_equal_tokens_or_FLOPs": True,
            "test_sha256": sha(OUT / "test_locked.jsonl"),
            "validation_sha256": sha(OUT / "validation_locked.jsonl"),
        },
    )
    print("PROTOCOL", read(OUT / "protocol.json"), flush=True)


class Contexts(DynamicChineseBERT):
    def __init__(self):
        super().__init__()
        self.extra = {}
        cache = ROOT / "data/cache/couplet_extended"
        if (cache / "context_ready.json").exists():
            self.extra_values = np.load(cache / "contexts.npy", mmap_mode="r")
            self.extra_valid = np.load(cache / "context_valid.npy", mmap_mode="r")
            self.extra = {
                tuple(r["ids"]): (r["offset"], r["length"])
                for r in rows(cache / "context_index.jsonl")
            }

    def __call__(self, ids):
        keys = [tuple(r[2 : r.index(5)]) for r in ids.detach().cpu().tolist()]
        missing = list(
            dict.fromkeys(
                k
                for k in keys
                if k not in self.index and k not in self.extra and k not in self.online
            )
        )
        if missing:
            self.add(missing)
        width = max(map(len, keys))
        v = np.zeros((len(keys), width, 768), np.float16)
        mask = np.zeros((len(keys), width), np.int64)
        for i, k in enumerate(keys):
            if k in self.index:
                p, n = self.index[k]
                a, b = self.values[p : p + n], self.valid[p : p + n]
            elif k in self.extra:
                p, n = self.extra[k]
                a, b = self.extra_values[p : p + n], self.extra_valid[p : p + n]
            else:
                a, b = self.online[k]
            v[i, : len(k)] = a
            mask[i, : len(k)] = b
        return torch.tensor(v, device=ids.device), torch.tensor(mask, device=ids.device)


def make_contexts():
    if (OUT / "context_ready.json").exists():
        return
    tok = CharacterTokenizer()
    old = DynamicChineseBERT()
    texts = dict.fromkeys(
        r["upper"]
        for p in [
            DATA / "train.jsonl",
            OUT / "control_train.jsonl",
            OUT / "validation_locked.jsonl",
            OUT / "test_locked.jsonl",
        ]
        for r in rows(p)
    )
    keys = [tuple(tok.encode(t)) for t in texts]
    missing = [k for k in keys if k not in old.index]
    total = sum(map(len, missing))
    assert all(1 not in k for k in keys)
    v = np.lib.format.open_memmap(
        OUT / "contexts.npy", mode="w+", dtype=np.float16, shape=(total, 768)
    )
    mask = np.lib.format.open_memmap(
        OUT / "context_valid.npy", mode="w+", dtype=np.uint8, shape=(total,)
    )
    index = []
    offset = 0
    for i in range(0, len(missing), 256):
        kk = missing[i : i + 256]
        old.add(kk)
        for k in kk:
            a, b = old.online[k]
            n = len(k)
            v[offset : offset + n] = a
            mask[offset : offset + n] = b
            index.append({"ids": list(k), "offset": offset, "length": n})
            offset += n
        old.online.clear()
        if i % 25600 == 0:
            print("SOURCE CACHE", i, len(missing), flush=True)
    v.flush()
    mask.flush()
    jsonl(OUT / "context_index.jsonl", index)
    dump(
        OUT / "context_ready.json",
        {
            "status": "complete",
            "new_sources": len(missing),
            "reused_v7_sources": len(keys) - len(missing),
            "characters": total,
            "source_only": True,
        },
    )
    del old
    torch.cuda.empty_cache()


def model(path):
    m, ck = initialize(path)
    m.context_provider = Contexts()
    return m, ck
